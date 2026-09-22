import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, watch } from "node:fs";
import net from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";
import test from "node:test";

type Message = Record<string, any>;

async function waitForState(path: string): Promise<any> {
  try { return JSON.parse(readFileSync(path, "utf8")); } catch {}
  return new Promise((resolveState, reject) => {
    const timer = setTimeout(() => { watcher.close(); reject(new Error("state timeout")); }, 10_000);
    const watcher = watch(join(path, ".."), () => {
      try {
        const state = JSON.parse(readFileSync(path, "utf8"));
        clearTimeout(timer);
        watcher.close();
        resolveState(state);
      } catch {}
    });
  });
}

async function connectClient(state: any) {
  const socket = net.createConnection({ host: "127.0.0.1", port: state.port });
  socket.setEncoding("utf8");
  let buffer = "";
  const messages: Message[] = [];
  const waiters: Array<{ predicate: (value: Message) => boolean; resolve: (value: Message) => void }> = [];
  socket.on("data", (chunk) => {
    buffer += chunk;
    while (buffer.includes("\n")) {
      const index = buffer.indexOf("\n");
      const value = JSON.parse(buffer.slice(0, index));
      buffer = buffer.slice(index + 1);
      const waiterIndex = waiters.findIndex((waiter) => waiter.predicate(value));
      if (waiterIndex >= 0) waiters.splice(waiterIndex, 1)[0]!.resolve(value);
      else messages.push(value);
    }
  });
  const next = (predicate: (value: Message) => boolean) => {
    const index = messages.findIndex(predicate);
    if (index >= 0) return Promise.resolve(messages.splice(index, 1)[0]!);
    return new Promise<Message>((resolve) => waiters.push({ predicate, resolve }));
  };
  await new Promise<void>((resolve, reject) => { socket.once("connect", resolve); socket.once("error", reject); });
  socket.write(`${JSON.stringify({ type: "auth", token: state.token })}\n`);
  const ready = await next((value) => value.type === "ready");
  return { socket, next, ready, request(id: number, command: string) {
    socket.write(`${JSON.stringify({ id, command })}\n`);
    return next((value) => value.id === id);
  }};
}

test("global service shares one worker and preserves service across client close", { timeout: 10_000 }, async () => {
  const directory = mkdtempSync(join(tmpdir(), "ptt-global-test-"));
  const statePath = join(directory, "endpoint.json");
  const root = resolve(".");
  const fake = resolve("tests-ts/fixtures/fake-worker.mjs");
  const server = spawn(process.execPath, ["scripts/voice-server.mjs", "--state", statePath, "--root", root], {
    cwd: root,
    env: {
      ...process.env,
      NODE_ENV: "test",
      PTT_TEST_WORKER_COMMAND: JSON.stringify([process.execPath, fake]),
      PTT_PRESET: "balanced",
      PTT_DEVICE: "auto",
    },
    stdio: ["ignore", "ignore", "pipe"],
  });
  let stderr = "";
  server.stderr.setEncoding("utf8");
  server.stderr.on("data", (chunk) => { stderr += chunk; });
  try {
    const state = await waitForState(statePath);
    const contender = spawn(process.execPath, ["scripts/voice-server.mjs", "--state", statePath, "--root", root], {
      cwd: root,
      env: {
        ...process.env,
        NODE_ENV: "test",
      PTT_TEST_WORKER_COMMAND: JSON.stringify([process.execPath, fake]),
        PTT_PRESET: "balanced",
        PTT_DEVICE: "auto",
      },
      stdio: "ignore",
    });
    assert.equal(await new Promise<number | null>((resolveExit) => contender.once("exit", resolveExit)), 0);

    const first = await connectClient(state);
    const second = await connectClient(state);
    if (!first.ready.model_ready) await first.next((value) => value.event === "worker_ready");

    assert.equal((await first.request(1, "start")).event, "recording");
    assert.equal((await second.request(1, "start")).ok, false);
    const level = await first.next((value) => value.event === "level");
    assert.equal(level.level, 0.5);

    first.socket.end();
    await second.next((value) => value.event === "owner_released");
    assert.equal((await second.request(2, "start")).event, "recording");
    second.socket.end();

    const third = await connectClient(state);
    assert.equal((await third.request(1, "warm")).ok, true);
    assert.equal((await third.request(2, "shutdown")).event, "shutdown");
    third.socket.end();
    const exitCode = await new Promise<number | null>((resolveExit) => server.once("exit", resolveExit));
    assert.equal(exitCode, 0, stderr);
  } finally {
    if (server.exitCode === null) server.kill();
    rmSync(directory, { recursive: true, force: true });
  }
});
