#!/usr/bin/env node
import { randomBytes } from "node:crypto";
import { appendFileSync, chmodSync, closeSync, existsSync, openSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import net from "node:net";
import { dirname, join } from "node:path";
import { spawn } from "node:child_process";

const args = process.argv.slice(2);
const value = (name) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
};
const statePath = value("--state");
const root = value("--root");
if (!statePath || !root) process.exit(2);
const lockPath = `${statePath}.lock`;
const logPath = `${statePath}.log`;
const log = (message) => {
  try { appendFileSync(logPath, `${new Date().toISOString()} ${message}\n`, { mode: 0o600 }); } catch {}
};
const pidAlive = (pid) => {
  try { process.kill(pid, 0); return true; } catch { return false; }
};

let lockFd;
try {
  lockFd = openSync(lockPath, "wx", 0o600);
  writeFileSync(lockFd, `${process.pid}\n`, "utf8");
} catch {
  let ownerPid = 0;
  try { ownerPid = Number.parseInt(readFileSync(lockPath, "utf8").trim(), 10); } catch {}
  if (Number.isInteger(ownerPid) && ownerPid > 1 && pidAlive(ownerPid)) process.exit(0);
  try { unlinkSync(lockPath); } catch {}
  try { unlinkSync(statePath); } catch {}
  try {
    lockFd = openSync(lockPath, "wx", 0o600);
    writeFileSync(lockFd, `${process.pid}\n`, "utf8");
  } catch { process.exit(0); }
}
closeSync(lockFd);

const token = randomBytes(32).toString("hex");
const clients = new Set();
const pending = new Map();
let owner = null;
let nextWorkerId = 1;
let workerReady = false;
let modelReady = false;
let shuttingDown = false;
let stdoutBuffer = "";

const uv = process.env.PTT_UV || "uv";
let workerCommand = uv;
let workerArgs = ["run", "--locked", "--project", root, "--python", "3.11", "python", join(root, "ptt_worker.py")];
if (process.env.NODE_ENV === "test" && process.env.PTT_TEST_WORKER_COMMAND) {
  const configured = JSON.parse(process.env.PTT_TEST_WORKER_COMMAND);
  if (!Array.isArray(configured) || configured.length === 0 || configured.some((item) => typeof item !== "string")) {
    throw new Error("PTT_TEST_WORKER_COMMAND must be a non-empty JSON argv array");
  }
  [workerCommand, ...workerArgs] = configured;
}
const worker = spawn(
  workerCommand,
  workerArgs,
  { cwd: root, env: { ...process.env, PTT_CLIENT_PID: String(process.pid), PYTHONUNBUFFERED: "1" }, stdio: ["pipe", "pipe", "pipe"] },
);
const writeSocket = (socket, message) => {
  if (!socket.destroyed) socket.write(`${JSON.stringify(message)}\n`);
};
const sendWorker = (socket, clientId, command) => {
  if (!workerReady || !worker.stdin.writable) {
    writeSocket(socket, { id: clientId, ok: false, error: "global speech worker is not ready" });
    return;
  }
  const id = nextWorkerId++;
  pending.set(id, { socket, clientId, command });
  worker.stdin.write(`${JSON.stringify({ id, command })}\n`);
};
const cancelOwner = () => {
  if (!owner) return;
  const previous = owner;
  owner = null;
  sendWorker(previous, null, "cancel");
  for (const socket of clients) {
    if (socket !== previous) writeSocket(socket, { event: "owner_released" });
  }
};

worker.stdout.setEncoding("utf8");
worker.stdout.on("data", (chunk) => {
  stdoutBuffer += chunk;
  while (stdoutBuffer.includes("\n")) {
    const index = stdoutBuffer.indexOf("\n");
    const line = stdoutBuffer.slice(0, index);
    stdoutBuffer = stdoutBuffer.slice(index + 1);
    if (!line.trim()) continue;
    let message;
    try { message = JSON.parse(line); } catch { continue; }
    if (message.event === "ready") {
      workerReady = true;
      for (const socket of clients) writeSocket(socket, { event: "worker_ready" });
    }
    if (message.event === "model_ready") {
      modelReady = true;
      for (const socket of clients) writeSocket(socket, message);
    }
    if (message.id !== undefined && message.id !== null) {
      const request = pending.get(message.id);
      if (request) {
        pending.delete(message.id);
        writeSocket(request.socket, { ...message, id: request.clientId });
        if (request.command === "cancel" || request.command === "stop" || message.ok === false) {
          if (owner === request.socket) owner = null;
        }
      }
      continue;
    }
    if (owner) writeSocket(owner, message);
    if (message.event === "model_error") {
      for (const socket of clients) writeSocket(socket, message);
    }
  }
});
worker.stderr.setEncoding("utf8");
worker.stderr.on("data", (chunk) => log(`worker: ${String(chunk).trimEnd()}`));

const server = net.createServer((socket) => {
  socket.setEncoding("utf8");
  let buffer = "";
  let authenticated = false;
  clients.add(socket);
  socket.on("data", (chunk) => {
    buffer += chunk;
    while (buffer.includes("\n")) {
      const index = buffer.indexOf("\n");
      const line = buffer.slice(0, index);
      buffer = buffer.slice(index + 1);
      if (!line.trim()) continue;
      let message;
      try { message = JSON.parse(line); } catch { socket.destroy(); return; }
      if (!authenticated) {
        if (message.type !== "auth" || message.token !== token) { socket.destroy(); return; }
        authenticated = true;
        writeSocket(socket, { type: "ready", worker_ready: workerReady, model_ready: modelReady });
        continue;
      }
      const id = message.id;
      const command = message.command;
      if (command === "warm") {
        writeSocket(socket, { id, ok: true, event: workerReady ? "ready" : "warming" });
      } else if (command === "start") {
        if (owner && owner !== socket) writeSocket(socket, { id, ok: false, error: "microphone is in use by another Prime client" });
        else { owner = socket; sendWorker(socket, id, command); }
      } else if (command === "shutdown") {
        writeSocket(socket, { id, ok: true, event: "shutdown" });
        shutdown();
      } else if (command === "devices") {
        sendWorker(socket, id, command);
      } else if (owner !== socket) {
        writeSocket(socket, { id, ok: false, error: "client does not own the active recording" });
      } else {
        sendWorker(socket, id, command);
      }
    }
  });
  socket.on("close", () => {
    clients.delete(socket);
    if (owner === socket) cancelOwner();
    for (const [id, request] of pending) {
      if (request.socket === socket) pending.delete(id);
    }
  });
  socket.on("error", () => {});
});

const cleanup = () => {
  try { unlinkSync(statePath); } catch {}
  try { unlinkSync(lockPath); } catch {}
};
const shutdown = () => {
  if (shuttingDown) return;
  shuttingDown = true;
  cancelOwner();
  try { worker.stdin.end(`${JSON.stringify({ id: nextWorkerId++, command: "shutdown" })}\n`); } catch {}
  for (const socket of clients) socket.end();
  server.close();
  setTimeout(() => { if (worker.exitCode === null) worker.kill(); cleanup(); process.exit(0); }, 1500).unref();
};
worker.on("exit", (code) => {
  log(`worker exited ${code}`);
  for (const socket of clients) writeSocket(socket, { event: "worker_exit", error: `speech worker exited (${code})` });
  cleanup();
  server.close(() => process.exit(code ?? 1));
});
worker.on("error", (error) => { log(`worker error: ${error.message}`); cleanup(); process.exit(1); });
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
process.on("exit", cleanup);

server.listen(0, "127.0.0.1", () => {
  const address = server.address();
  if (!address || typeof address === "string") process.exit(1);
  const temporary = `${statePath}.${process.pid}.tmp`;
  writeFileSync(temporary, `${JSON.stringify({
    version: 1,
    pid: process.pid,
    port: address.port,
    token,
    root,
    preset: process.env.PTT_PRESET || "balanced",
    device: process.env.PTT_DEVICE || "auto",
  })}\n`, { mode: 0o600 });
  renameSync(temporary, statePath);
  try { chmodSync(statePath, 0o600); } catch {}
  log(`listening on ${address.port}`);
});
