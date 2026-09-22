import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, statSync, watch, writeFileSync, type FSWatcher } from "node:fs";
import { homedir } from "node:os";
import { fileURLToPath } from "node:url";
import { basename, dirname, join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { CustomEditor } from "@earendil-works/pi-coding-agent";
import {
  isKeyRelease,
  Key,
  matchesKey,
} from "@earendil-works/pi-tui";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const BURST_GAP_MS = 120;
const WARMING_EVENT_COUNT = 2;
const COMMIT_EVENT_COUNT = 5;
const RELEASE_GAP_MS = 200;
const INTERIM_PAINT_MS = 150;
const FIRST_RELEASE_FALLBACK_MS = 2000;
const TAP_SILENCE_MS = 15_000;
const TAP_MAX_MS = 120_000;
const DOUBLE_TAP_MS = 300;
const DOUBLE_TAP_DISAMBIGUATE_MS = 120;

export type VoiceMode = "hold" | "tap";
export type VoiceSettings = { enabled: boolean; mode: VoiceMode; autoSubmit: boolean };

const RUNTIME_IDENTITY = `${process.execPath} ${process.argv[1] ?? ""}`.toLowerCase();
const IS_PRIME_RUNTIME = RUNTIME_IDENTITY.includes("prime-agent") || Boolean(process.env.PRIME_AGENT_LAUNCHER_PATH);
const DEFAULT_AGENT_DIR = join(homedir(), IS_PRIME_RUNTIME ? ".prime/agent" : ".pi/agent");
const SETTINGS_PATH = process.env.PTT_CONFIG ?? join(DEFAULT_AGENT_DIR, "push-to-talk.json");

function workerEnvironment(): NodeJS.ProcessEnv {
  const keys = [
    "PATH", "Path", "HOME", "USER", "LOGNAME", "TMPDIR", "TEMP", "TMP",
    "LANG", "LC_ALL", "SystemRoot", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
    "XDG_CACHE_HOME", "XDG_RUNTIME_DIR", "HF_HOME", "HUGGINGFACE_HUB_CACHE",
    "TORCH_HOME", "UV_CACHE_DIR", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
    "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
    "PULSE_SERVER", "PIPEWIRE_REMOTE", "ALSA_CONFIG_PATH", "OMP_WAIT_POLICY",
    "PTT_DEVICE", "PTT_INPUT_DEVICE", "PTT_STREAM_CHUNK_MS", "PTT_STREAM_RIGHT_MS",
    "PTT_ALLOW_TELEMETRY",
  ];
  const environment: NodeJS.ProcessEnv = { PYTHONUNBUFFERED: "1", PTT_CLIENT_PID: String(process.pid) };
  for (const key of keys) {
    const value = process.env[key];
    if (value !== undefined) environment[key] = value;
  }
  return environment;
}

function loadVoiceSettings(): VoiceSettings {
  try {
    if (existsSync(SETTINGS_PATH)) {
      const value = JSON.parse(readFileSync(SETTINGS_PATH, "utf8")) as Partial<VoiceSettings>;
      return { enabled: value.enabled !== false, mode: value.mode === "tap" ? "tap" : "hold", autoSubmit: value.autoSubmit === true };
    }
  } catch {}
  return { enabled: process.env.PTT_ENABLED !== "0", mode: process.env.PTT_MODE === "tap" ? "tap" : "hold", autoSubmit: process.env.PTT_AUTO_SUBMIT === "1" };
}

function debugEvent(event: object): void {
  const path = process.env.PTT_DEBUG_LOG;
  if (!path) return;
  try { appendFileSync(path, `${JSON.stringify({ time: Date.now(), ...event })}\n`, { mode: 0o600 }); } catch {}
}

function saveVoiceSettings(settings: VoiceSettings): void {
  const directory = dirname(SETTINGS_PATH);
  mkdirSync(directory, { recursive: true });
  const temporary = join(directory, `.${basename(SETTINGS_PATH)}.${process.pid}.tmp`);
  writeFileSync(temporary, `${JSON.stringify(settings, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
  renameSync(temporary, SETTINGS_PATH);
}

type Reply = {
  id?: number;
  ok?: boolean;
  event?: string;
  text?: string;
  error?: string;
  level?: number;
  provisional?: boolean;
};

type WorkerCommand = "start" | "stop" | "cancel" | "devices" | "shutdown" | "arm_release" | "disarm_release";

export interface VoiceWorker {
  onLevel?: (level: number) => void;
  onInterim?: (text: string) => void;
  onReleaseTimeout?: () => void;
  onFatalError?: (error: Error) => void;
  warm(): Promise<void>;
  request(command: WorkerCommand): Promise<Reply>;
  close(): void;
}


class WorkerClient implements VoiceWorker {
  private child?: ChildProcessWithoutNullStreams;
  private ready?: Promise<void>;
  private nextId = 1;
  private pending = new Map<number, {
    resolve: (value: Reply) => void;
    reject: (error: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  }>();
  private stderrTail = "";
  private closed = false;
  onLevel?: (level: number) => void;
  onInterim?: (text: string) => void;
  onReleaseTimeout?: () => void;
  onFatalError?: (error: Error) => void;

  private ensureStarted(): Promise<void> {
    if (this.closed) return Promise.reject(new Error("speech worker client is closed"));
    if (this.ready) return this.ready;
    this.ready = new Promise((resolve, reject) => {
      const uv = process.env.PTT_UV ?? "uv";
      const child = spawn(
        uv,
        ["run", "--locked", "--project", ROOT, "--python", "3.11", "python", join(ROOT, "ptt_worker.py")],
        {
          cwd: ROOT,
          env: workerEnvironment(),
          stdio: ["pipe", "pipe", "pipe"],
        },
      );
      this.child = child;
      let stdoutBuffer = "";
      let settled = false;
      const startupTimer = setTimeout(() => {
        if (settled) return;
        settled = true;
        child.kill();
        reject(new Error("speech worker startup timed out"));
      }, 30_000);
      startupTimer.unref();

      child.stdout.setEncoding("utf8");
      child.stdout.on("data", (chunk: string) => {
        stdoutBuffer += chunk;
        while (stdoutBuffer.includes("\n")) {
          const index = stdoutBuffer.indexOf("\n");
          const line = stdoutBuffer.slice(0, index);
          stdoutBuffer = stdoutBuffer.slice(index + 1);
          if (!line.trim()) continue;
          let message: Reply;
          try {
            message = JSON.parse(line) as Reply;
          } catch {
            continue;
          }
          if (message.event === "ready" && !settled) {
            settled = true;
            clearTimeout(startupTimer);
            resolve();
          }
          if (message.event === "level" && typeof message.level === "number") {
            this.onLevel?.(Math.max(0, Math.min(1, message.level)));
          }
          if (message.event === "interim" && typeof message.text === "string") {
            this.onInterim?.(message.text);
          }
          if (message.event === "release_timeout") this.onReleaseTimeout?.();
          if (message.event === "model_error") {
            const error = new Error(message.error ?? "speech model failed to load");
            this.failAll(error);
            this.onFatalError?.(error);
          }
          if (message.id !== undefined) {
            const waiter = this.pending.get(message.id);
            if (waiter) {
              this.pending.delete(message.id);
              clearTimeout(waiter.timer);
              if (message.ok === false) waiter.reject(new Error(message.error ?? "worker request failed"));
              else waiter.resolve(message);
            }
          }
        }
      });
      child.stderr.setEncoding("utf8");
      child.stderr.on("data", (chunk: string) => {
        this.stderrTail = (this.stderrTail + chunk).slice(-4000);
      });
      child.stdin.on("error", (error) => this.failAll(error));
      child.on("error", (error) => {
        clearTimeout(startupTimer);
        if (!settled) { settled = true; reject(error); }
        this.failAll(error);
      });
      child.on("exit", (code) => {
        clearTimeout(startupTimer);
        const detail = this.stderrTail.trim();
        const error = new Error(`speech worker exited (${code ?? "signal"})${detail ? `: ${detail}` : ""}`);
        if (!settled) { settled = true; reject(error); }
        this.failAll(error);
        this.child = undefined;
        this.ready = undefined;
      });
    });
    return this.ready;
  }

  private failAll(error: Error): void {
    for (const waiter of this.pending.values()) {
      clearTimeout(waiter.timer);
      waiter.reject(error);
    }
    this.pending.clear();
  }

  async warm(): Promise<void> {
    await this.ensureStarted();
  }

  async request(command: WorkerCommand): Promise<Reply> {
    if (this.closed) throw new Error("speech worker client is closed");
    await this.ensureStarted();
    const child = this.child;
    if (!child?.stdin.writable) throw new Error("speech worker is not available");
    const id = this.nextId++;
    const timeoutMs = command === "stop" ? 310_000 : 30_000;
    const reply = new Promise<Reply>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`speech worker ${command} request timed out`));
      }, timeoutMs);
      timer.unref();
      this.pending.set(id, { resolve, reject, timer });
    });
    child.stdin.write(`${JSON.stringify({ id, command })}\n`, (error) => {
      if (!error) return;
      const waiter = this.pending.get(id);
      this.pending.delete(id);
      if (waiter) clearTimeout(waiter.timer);
      waiter?.reject(error);
    });
    return reply;
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.failAll(new Error("speech worker client closed"));
    const child = this.child;
    if (!child) return;
    try {
      child.stdin.end(`${JSON.stringify({ id: this.nextId++, command: "shutdown" })}\n`);
    } catch {
      child.kill();
      return;
    }
    setTimeout(() => { if (child.exitCode === null) child.kill(); }, 1000).unref();
  }
}


/**
 * Daemon-backed Prime imports extension modules in the TTY client before it
 * constructs ProcessTerminal, but binds the factory in the worker. Capture the
 * single terminal data callback at registration time so hold-Space can remain
 * a package-only feature without patching Prime.
 */
export class ClientEditorVoiceBridge {
  private worker?: VoiceWorker;
  private readonly workerFactory: () => VoiceWorker;
  private settingsWatcher?: FSWatcher;
  private settingsMtimeMs = 0;
  private settings: VoiceSettings;
  private commandBuffer = "";
  private burstCount = 0;
  private leakedSpaces = 0;
  private lastSpaceAt = 0;
  private recording = false;
  private processing = false;
  private captureGeneration = 0;
  private hasInterim = false;
  private targetInterim = "";
  private displayedInterim = "";
  private lastInterimPaint = 0;
  private marker = "";
  private meterIndex = 1;
  private editor?: CustomEditor;
  private originalInput?: (data: string) => void;
  private anchor?: { original: string; expected: string; prefix: string; suffix: string };

  constructor(options: {
    settings?: VoiceSettings;
    workerFactory?: () => VoiceWorker;
    watchSettings?: boolean;
  } = {}) {
    this.settings = options.settings ?? loadVoiceSettings();
    this.workerFactory = options.workerFactory ?? (() => new WorkerClient());
    this.settingsMtimeMs = this.readSettingsMtime();
    this.applySettings(this.settings);
    if (options.watchSettings !== false) {
      try {
        this.settingsWatcher = watch(dirname(SETTINGS_PATH), (_event, filename) => {
          if (filename && filename.toString() !== basename(SETTINGS_PATH)) return;
          this.refreshSettingsFromDisk(true);
        });
        this.settingsWatcher.unref();
      } catch {}
    }
  }

  handleInput(editor: CustomEditor, data: string, original: (data: string) => void): void {
    this.editor = editor;
    this.originalInput = original;
    this.refreshSettingsFromDisk();
    if (!isKeyRelease(data)) this.observeVoiceCommand(data);

    if (!this.settings.enabled || this.settings.mode !== "hold") {
      if (!isKeyRelease(data)) original(data);
      return;
    }

    if (matchesKey(data, Key.space)) {
      if (isKeyRelease(data)) {
        this.resetCandidate();
        if (this.recording && !this.processing) this.releaseHold();
        return;
      }
      if (this.processing) {
        this.abortProcessing();
        original(data);
        return;
      }
      this.handleSpace(data, original);
      return;
    }

    if (isKeyRelease(data)) return;
    if (this.processing) this.abortProcessing();
    else if (this.recording) this.cancelRecording();
    else if (this.burstCount > 0) this.resetCandidate();
    original(data);
  }

  close(): void {
    this.settingsWatcher?.close();
    this.settingsWatcher = undefined;
    this.cancelRecording();
    this.stopWorker();
  }

  private handleSpace(data: string, original: (data: string) => void): void {
    const now = Date.now();
    if (!this.recording && this.lastSpaceAt > 0 && now - this.lastSpaceAt > BURST_GAP_MS) this.resetCandidate();
    this.lastSpaceAt = now;

    if (this.recording) {
      this.advanceInterimDisplay();
      this.armRelease();
      return;
    }

    const previous = this.burstCount;
    this.burstCount += 1;
    if (this.burstCount >= COMMIT_EVENT_COUNT) {
      for (let index = 0; index < this.leakedSpaces; index++) original("\x7f");
      this.leakedSpaces = 0;
      this.burstCount = 0;
      this.captureAnchor();
      const generation = ++this.captureGeneration;
      this.recording = true;
      this.processing = false;
      this.hasInterim = false;
      this.targetInterim = "";
      this.displayedInterim = "";
      this.lastInterimPaint = 0;
      this.meterIndex = 1;
      debugEvent({ event: "editor_bridge_commit" });
      this.replaceMarker("▁");
      void this.startWorker().request("start").catch(() => this.failRecording(generation));
      this.armRelease(generation);
      return;
    }

    if (previous === 0) {
      original(data);
      this.leakedSpaces = 1;
    }
  }

  private captureAnchor(): void {
    const editor = this.editor;
    if (!editor) return;
    const text = editor.getText();
    const cursor = editor.getCursor();
    const lines = editor.getLines();
    let offset = cursor.col;
    for (let line = 0; line < cursor.line; line++) offset += (lines[line]?.length ?? 0) + 1;
    this.anchor = { original: text, expected: text, prefix: text.slice(0, offset), suffix: text.slice(offset) };
  }

  private replaceMarker(next: string): boolean {
    const editor = this.editor;
    const original = this.originalInput;
    const anchor = this.anchor;
    if (!editor || !original || !anchor || editor.getText() !== anchor.expected) return false;
    const value = `${anchor.prefix}${next}${anchor.suffix}`;
    editor.setText(value);
    for (let index = 0; index < Array.from(anchor.suffix).length; index++) original("\x1b[D");
    anchor.expected = value;
    this.marker = next;
    return true;
  }

  private finishRecording(text: string, generation: number): void {
    if (generation !== this.captureGeneration) return;
    const anchor = this.anchor;
    if (!anchor) return this.clearCaptureState();
    const leading = anchor.prefix.length > 0 && !/\s$/.test(anchor.prefix) && text ? " " : "";
    const trailing = anchor.suffix.length > 0 && !/^\s/.test(anchor.suffix) && text ? " " : "";
    if (!this.replaceMarker(`${leading}${text}${trailing}`)) {
      this.clearCaptureState();
      return;
    }
    const shouldSubmit = this.settings.autoSubmit && text.trim().split(/\s+/).filter(Boolean).length >= 3;
    debugEvent({ event: "editor_bridge_final", textLength: text.length });
    this.clearCaptureState(false);
    this.anchor = undefined;
    if (shouldSubmit) this.originalInput?.("\r");
  }

  private cancelRecording(): void {
    this.captureGeneration += 1;
    this.restoreAnchor();
    const worker = this.worker;
    if (worker && (this.recording || this.processing)) {
      void worker.request("disarm_release").catch(() => {});
      void worker.request("cancel").catch(() => {});
    }
    this.clearCaptureState();
  }

  private abortProcessing(): void {
    this.captureGeneration += 1;
    this.restoreAnchor();
    debugEvent({ event: "editor_bridge_processing_aborted" });
    this.clearCaptureState();
  }

  private restoreAnchor(): void {
    const anchor = this.anchor;
    if (!anchor) return;
    if (this.editor?.getText() === anchor.expected) {
      this.replaceMarker("");
    }
    this.anchor = undefined;
  }

  private clearCaptureState(clearAnchor = true): void {
    this.recording = false;
    this.processing = false;
    this.hasInterim = false;
    this.targetInterim = "";
    this.displayedInterim = "";
    this.lastInterimPaint = 0;
    this.marker = "";
    this.burstCount = 0;
    this.leakedSpaces = 0;
    this.lastSpaceAt = 0;
    if (clearAnchor) this.anchor = undefined;
  }

  private armRelease(generation = this.captureGeneration): void {
    void this.startWorker().request("arm_release").catch(() => this.failRecording(generation));
  }

  private releaseHold(): void {
    if (!this.recording || this.processing) return;
    const generation = this.captureGeneration;
    this.processing = true;
    if (this.hasInterim && this.targetInterim) {
      this.replaceMarker(this.targetInterim);
      this.displayedInterim = this.targetInterim;
    } else {
      this.replaceMarker("");
    }
    void this.startWorker().request("stop").then((reply) => {
      if (generation !== this.captureGeneration) return;
      this.finishRecording(reply.text?.trim() ?? "", generation);
    }).catch(() => this.failRecording(generation));
  }

  private failRecording(generation: number): void {
    if (generation !== this.captureGeneration) return;
    this.captureGeneration += 1;
    this.restoreAnchor();
    this.clearCaptureState();
  }

  private advanceInterimDisplay(force = false): void {
    if (!this.hasInterim || !this.targetInterim) return;
    const now = Date.now();
    if (!force && now - this.lastInterimPaint < INTERIM_PAINT_MS) return;
    const displayed = Array.from(this.displayedInterim);
    const target = Array.from(this.targetInterim);
    let common = 0;
    while (common < displayed.length && common < target.length && displayed[common] === target[common]) common++;
    const remaining = target.slice(common);
    if (!remaining.length && common === displayed.length) return;
    const take = Math.max(1, Math.ceil(remaining.length / 2));
    this.displayedInterim = [...target.slice(0, common), ...remaining.slice(0, take)].join("");
    this.lastInterimPaint = now;
    this.replaceMarker(this.displayedInterim);
  }

  private resetCandidate(): void {
    this.burstCount = 0;
    this.leakedSpaces = 0;
    this.lastSpaceAt = 0;
  }

  private observeVoiceCommand(data: string): void {
    if (data.length !== 1) return;
    const code = data.charCodeAt(0);
    if (code === 0x03) this.commandBuffer = "";
    else if (code === 0x7f || code === 0x08) this.commandBuffer = this.commandBuffer.slice(0, -1);
    else if (code === 0x0d || code === 0x0a) {
      const command = this.commandBuffer.trim().toLowerCase();
      this.commandBuffer = "";
      if (!command.startsWith("/voice")) return;
      const option = command.slice("/voice".length).trim();
      if (option === "status") return;
      if (option) return;
      const next = { ...this.settings, enabled: !this.settings.enabled, mode: "hold" as const };
      this.applySettings(next);
    } else if (code >= 0x20 && code <= 0x7e) {
      this.commandBuffer += data;
      if (this.commandBuffer.length > 256) this.commandBuffer = this.commandBuffer.slice(-256);
    }
  }

  private readSettingsMtime(): number {
    try { return statSync(SETTINGS_PATH).mtimeMs; } catch { return 0; }
  }

  private refreshSettingsFromDisk(force = false): void {
    const mtime = this.readSettingsMtime();
    if (!force && mtime === this.settingsMtimeMs) return;
    this.settingsMtimeMs = mtime;
    this.applySettings(loadVoiceSettings());
  }

  private applySettings(next: VoiceSettings): void {
    const modeChanged = this.settings.mode !== next.mode;
    this.settings = next;
    if (!next.enabled) {
      this.cancelRecording();
      this.stopWorker();
    } else {
      this.startWorker();
      if (modeChanged) this.cancelRecording();
    }
  }

  private startWorker(): VoiceWorker {
    if (this.worker) return this.worker;
    const worker = this.workerFactory();
    this.worker = worker;
    worker.onReleaseTimeout = () => this.releaseHold();
    worker.onFatalError = () => this.failRecording(this.captureGeneration);
    worker.onInterim = (text) => {
      const value = text.trim();
      if (!this.recording || !value) return;
      this.hasInterim = true;
      this.targetInterim = value;
      this.advanceInterimDisplay(true);
    };
    worker.onLevel = (level) => {
      if (!this.recording || this.processing || this.hasInterim) return;
      const blocks = " ▁▂▃▄▅▆▇█";
      const next = Math.max(1, Math.min(blocks.length - 1, Math.round(Math.min(level * 1.8, 1) * (blocks.length - 1))));
      if (next === this.meterIndex) return;
      this.meterIndex = next;
      this.replaceMarker(blocks[next]!);
    };
    void worker.warm().catch(() => {});
    return worker;
  }

  private stopWorker(): void {
    const worker = this.worker;
    this.worker = undefined;
    worker?.close();
  }
}

function installDaemonClientEditorBridge(): void {
  if (!process.stdin.isTTY || !process.stdout.isTTY || process.argv.includes("--in-process") || !IS_PRIME_RUNTIME) return;
  const key = Symbol.for("push-to-talk-prime.editor-bridge");
  const globals = globalThis as typeof globalThis & { [key]?: ClientEditorVoiceBridge };
  if (globals[key]) return;
  const bridge = new ClientEditorVoiceBridge();
  globals[key] = bridge;
  const prototype = CustomEditor.prototype as CustomEditor & { wantsKeyRelease?: boolean };
  prototype.wantsKeyRelease = true;
  const original = CustomEditor.prototype.handleInput;
  CustomEditor.prototype.handleInput = function (data: string): void {
    bridge.handleInput(this, data, (value) => original.call(this, value));
  };
  process.once("exit", () => bridge.close());
}

installDaemonClientEditorBridge();

type State = "idle" | "candidate" | "recording" | "transcribing";

class RecordingController {
  state: State = "idle";
  private generation = 0;
  private animationTimer?: ReturnType<typeof setInterval>;
  private targetLevel = 0;
  private smoothLevel = 0;
  private toggleHint = false;
  private lastDisplay = "";
  captureMode: VoiceMode = "hold";
  private silenceTimer?: ReturnType<typeof setTimeout>;
  private maxTimer?: ReturnType<typeof setTimeout>;
  private insertTranscript?: (text: string) => boolean;
  private insertInterim?: (text: string) => boolean;

  constructor(private worker: WorkerClient, private ctx: ExtensionContext) {
    worker.onFatalError = (error) => this.fail(error);
    worker.onLevel = (level) => {
      this.targetLevel = level;
      if (this.captureMode === "tap" && this.state === "recording" && level > 0.03) this.armTapSilence();
    };
    worker.onInterim = (text) => {
      this.insertInterim?.(text);
      if (this.captureMode === "tap" && this.state === "recording") this.armTapSilence();
    };
  }

  setInsertHandlers(interim: (text: string) => boolean, final: (text: string) => boolean): void {
    this.insertInterim = interim;
    this.insertTranscript = final;
  }

  private clearCaptureTimers(): void {
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    if (this.maxTimer) clearTimeout(this.maxTimer);
    this.silenceTimer = this.maxTimer = undefined;
  }

  private armTapSilence(): void {
    if (this.captureMode !== "tap" || this.state !== "recording") return;
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    this.silenceTimer = setTimeout(() => this.stop(), TAP_SILENCE_MS);
  }

  private clearDisplay(): void {
    if (this.animationTimer) clearInterval(this.animationTimer);
    this.animationTimer = undefined;
    this.targetLevel = 0;
    this.smoothLevel = 0;
    this.lastDisplay = "";
    this.ctx.ui.setStatus("push-to-talk", undefined);
    this.ctx.ui.setWidget("push-to-talk", undefined);
  }

  showWarmup(): void {
    if (this.state !== "idle") return;
    const line = "keep holding…";
    this.ctx.ui.setStatus("push-to-talk", line);
    this.ctx.ui.setWidget("push-to-talk", [line], { placement: "belowEditor" });
  }

  clearWarmup(): void {
    if (this.state === "idle") this.clearDisplay();
  }

  private renderListening(): void {
    if (this.captureMode === "tap") {
      const line = "● REC · tap to send";
      if (line === this.lastDisplay) return;
      this.lastDisplay = line;
      this.ctx.ui.setStatus("push-to-talk", line);
      this.ctx.ui.setWidget("push-to-talk", [line], { placement: "belowEditor" });
      return;
    }
    const blocks = " ▁▂▃▄▅▆▇█";
    this.smoothLevel = this.smoothLevel * 0.7 + Math.min(this.targetLevel * 1.8, 1) * 0.3;
    const index = Math.max(1, Math.min(blocks.length - 1, Math.round(this.smoothLevel * (blocks.length - 1))));
    const meter = blocks[index];
    const action = this.toggleHint ? "press F8 to stop" : "release space to transcribe";
    this.ctx.ui.setStatus("push-to-talk", `${meter} listening… · ${action}`);
    this.ctx.ui.setWidget("push-to-talk", [`${meter} listening…`], { placement: "belowEditor" });
  }

  private startMeter(toggleHint = false): void {
    this.clearDisplay();
    this.toggleHint = toggleHint;
    this.renderListening();
    if (this.captureMode === "hold") this.animationTimer = setInterval(() => this.renderListening(), 50);
  }

  private showTranscribing(): void {
    this.clearDisplay();
    const line = "Voice: processing…";
    this.ctx.ui.setStatus("push-to-talk", line);
    this.ctx.ui.setWidget("push-to-talk", [line], { placement: "belowEditor" });
  }

  private fail(error: unknown): void {
    this.clearCaptureTimers();
    this.state = "idle";
    this.clearDisplay();
    const message = error instanceof Error ? error.message : String(error);
    this.ctx.ui.notify(`Push-to-talk: ${message}. Run /ptt-doctor for setup help.`, "error");
  }

  begin(toggleHint = false, mode: VoiceMode = "hold"): void {
    if (this.state !== "idle") return;
    const generation = ++this.generation;
    this.captureMode = mode;
    this.state = "recording";
    this.startMeter(toggleHint);
    if (mode === "tap") {
      this.armTapSilence();
      this.maxTimer = setTimeout(() => this.stop(), TAP_MAX_MS);
    } else if (toggleHint) {
      // Manual Prime fallback has no key release event. Never leave the
      // microphone open indefinitely if the second toggle is missed.
      this.maxTimer = setTimeout(() => this.stop(), TAP_MAX_MS);
    }
    void this.worker.request("start").catch((error) => {
      if (generation === this.generation) this.fail(error);
    });
  }

  cancel(): void {
    if (this.state === "idle") {
      this.clearDisplay();
      return;
    }
    ++this.generation;
    this.clearCaptureTimers();
    const wasCapturing = this.state === "recording";
    this.state = "idle";
    this.clearDisplay();
    if (wasCapturing) void this.worker.request("cancel").catch(() => {});
  }

  stop(): void {
    if (this.state !== "recording") return;
    const generation = ++this.generation;
    this.clearCaptureTimers();
    this.state = "transcribing";
    this.showTranscribing();
    void this.worker.request("stop").then((reply) => {
      if (generation !== this.generation) return;
      this.state = "idle";
      this.clearDisplay();
      const text = reply.text?.trim();
      if (text) {
        const inserted = this.insertTranscript?.(text) ?? false;
        if (!inserted && !this.insertTranscript) this.ctx.ui.pasteToEditor(text);
        else if (!inserted) this.ctx.ui.notify("Prompt changed while transcribing; transcript was not inserted", "warning");
      } else this.ctx.ui.notify("No speech detected", "warning");
    }).catch((error) => {
      if (generation === this.generation) this.fail(error);
    });
  }

  toggle(mode: VoiceMode = "hold"): void {
    if (this.state === "idle") this.begin(true, mode);
    else if (this.state === "recording") this.stop();
    else this.ctx.ui.notify("Transcription is still running", "info");
  }
}

class PushToTalkEditor extends CustomEditor {
  wantsKeyRelease = true;
  private disposed = false;
  private burstCount = 0;
  private leakedSpaces = 0;
  private holdCommitted = false;
  private burstTimer?: ReturnType<typeof setTimeout>;
  private releaseTimer?: ReturnType<typeof setTimeout>;
  private firstReleaseTimer?: ReturnType<typeof setTimeout>;
  private awaitingSubmitDoubleTap = false;
  private firstSubmitTapAt = 0;
  private submitTapTimer?: ReturnType<typeof setTimeout>;
  private anchor?: {
    original: string;
    expected: string;
    prefix: string;
    suffix: string;
  };

  constructor(
    tui: any,
    theme: any,
    keybindings: any,
    private recording: RecordingController,
    private settings: VoiceSettings,
  ) {
    super(tui, theme, keybindings);
    recording.setInsertHandlers(
      (text) => this.applyAnchoredTranscript(text, false),
      (text) => this.applyAnchoredTranscript(text, true),
    );
  }

  prepareAnchor(): void {
    const text = this.getText();
    const cursor = this.getCursor();
    const lines = this.getLines();
    let offset = cursor.col;
    for (let line = 0; line < cursor.line; line++) offset += (lines[line]?.length ?? 0) + 1;
    this.anchor = {
      original: text,
      expected: text,
      prefix: text.slice(0, offset),
      suffix: text.slice(offset),
    };
  }

  private setTextWithCursorBeforeSuffix(value: string, suffix: string): void {
    this.setText(value);
    for (let index = 0; index < Array.from(suffix).length; index++) super.handleInput("\x1b[D");
  }

  private clearCandidate(): void {
    if (this.burstTimer) clearTimeout(this.burstTimer);
    this.burstTimer = undefined;
    this.burstCount = 0;
    this.leakedSpaces = 0;
    this.recording.clearWarmup();
  }

  private clearReleaseTimers(): void {
    if (this.releaseTimer) clearTimeout(this.releaseTimer);
    if (this.firstReleaseTimer) clearTimeout(this.firstReleaseTimer);
    this.releaseTimer = this.firstReleaseTimer = undefined;
  }

  private removeLeakedSpaces(): void {
    for (let index = 0; index < this.leakedSpaces; index++) super.handleInput("\x7f");
    this.leakedSpaces = 0;
  }

  private armRelease(short = true): void {
    this.clearReleaseTimers();
    const delay = short ? RELEASE_GAP_MS : FIRST_RELEASE_FALLBACK_MS + RELEASE_GAP_MS;
    const timer = setTimeout(() => {
      this.clearReleaseTimers();
      this.holdCommitted = false;
      this.recording.stop();
    }, delay);
    if (short) this.releaseTimer = timer;
    else this.firstReleaseTimer = timer;
  }

  private commitHold(): void {
    if (this.burstTimer) clearTimeout(this.burstTimer);
    this.burstTimer = undefined;
    this.removeLeakedSpaces();
    this.burstCount = 0;
    this.holdCommitted = true;
    this.prepareAnchor();
    this.recording.clearWarmup();
    this.recording.begin(false, "hold");
    // If a terminal emits the commit burst but no later repeat, keep the same
    // first-event grace period Claude Code uses before inferring release.
    this.armRelease(false);
  }

  private cursorIsAtEnd(): boolean {
    const cursor = this.getCursor();
    const lines = this.getLines();
    return cursor.line === lines.length - 1 && cursor.col === (lines.at(-1)?.length ?? 0);
  }

  private clearDoubleTapSubmit(): void {
    if (this.submitTapTimer) clearTimeout(this.submitTapTimer);
    this.submitTapTimer = undefined;
    this.firstSubmitTapAt = 0;
    this.awaitingSubmitDoubleTap = false;
  }

  private handleDoubleTapSubmit(count: number): boolean {
    if (!this.awaitingSubmitDoubleTap || !this.cursorIsAtEnd() || this.recording.state !== "idle") return false;
    const now = Date.now();
    if (this.firstSubmitTapAt > 0 && now - this.firstSubmitTapAt <= DOUBLE_TAP_MS) {
      this.clearDoubleTapSubmit();
      super.handleInput("\x7f");
      setTimeout(() => { if (!this.disposed) super.handleInput("\r"); }, DOUBLE_TAP_DISAMBIGUATE_MS);
      return true;
    }
    this.firstSubmitTapAt = now;
    super.handleInput(" ");
    if (this.submitTapTimer) clearTimeout(this.submitTapTimer);
    this.submitTapTimer = setTimeout(() => this.clearDoubleTapSubmit(), DOUBLE_TAP_MS);
    return true;
  }

  private handleTapMode(count: number): void {
    if (this.recording.state === "idle") {
      if (this.getText().length !== 0) {
        for (let index = 0; index < count; index++) super.handleInput(" ");
        return;
      }
      this.prepareAnchor();
      this.recording.begin(false, "tap");
      return;
    }
    if (this.recording.state === "recording") {
      this.recording.stop();
      return;
    }
    for (let index = 0; index < count; index++) super.handleInput(" ");
  }

  private handleSpaceEvents(count: number): void {
    if (this.holdCommitted && this.recording.state === "recording") {
      this.armRelease(true);
      return;
    }
    if (this.recording.state !== "idle") {
      // Toggle-mode recording and processing do not steal normal typing.
      for (let index = 0; index < count; index++) super.handleInput(" ");
      return;
    }

    const previous = this.burstCount;
    this.burstCount += count;
    if (this.burstCount >= COMMIT_EVENT_COUNT) {
      this.commitHold();
      return;
    }

    // Match Claude Code's leak-safe warmup: at most the first two repeat
    // events reach the editor. They remain normal spaces if the burst stops.
    const passThrough = Math.max(0, Math.min(count, WARMING_EVENT_COUNT - previous));
    for (let index = 0; index < passThrough; index++) super.handleInput(" ");
    this.leakedSpaces += passThrough;
    if (this.burstCount >= WARMING_EVENT_COUNT) this.recording.showWarmup();

    if (this.burstTimer) clearTimeout(this.burstTimer);
    this.burstTimer = setTimeout(() => this.clearCandidate(), BURST_GAP_MS);
  }

  private applyAnchoredTranscript(transcript: string, final: boolean): boolean {
    const anchor = this.anchor;
    if (!anchor || this.getText() !== anchor.expected) return false;

    const leading = anchor.prefix.length > 0 && !/\s$/.test(anchor.prefix) && transcript.length > 0 ? " " : "";
    const trailing = anchor.suffix.length > 0 && !/^\s/.test(anchor.suffix) && transcript.length > 0 ? " " : "";
    const tail = `${trailing}${anchor.suffix}`;
    const value = `${anchor.prefix}${leading}${transcript}${tail}`;
    this.setTextWithCursorBeforeSuffix(value, tail);
    anchor.expected = value;
    if (final) {
      this.anchor = undefined;
      const wordCount = transcript.trim().split(/\s+/).filter(Boolean).length;
      const autoSubmit = wordCount >= 3 && (this.recording.captureMode === "tap" || this.settings.autoSubmit);
      if (autoSubmit) setTimeout(() => { if (!this.disposed) super.handleInput("\r"); }, 0);
      else if (this.recording.captureMode === "hold" && anchor.suffix.length === 0) {
        this.awaitingSubmitDoubleTap = true;
      }
    }
    return true;
  }

  private restoreAnchor(): void {
    const anchor = this.anchor;
    this.anchor = undefined;
    if (!anchor || this.getText() !== anchor.expected) return;
    this.setTextWithCursorBeforeSuffix(anchor.original, anchor.suffix);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.clearCandidate();
    this.clearReleaseTimers();
    this.clearDoubleTapSubmit();
    this.holdCommitted = false;
    this.restoreAnchor();
    this.recording.cancel();
  }

  cancelFromOutside(): void {
    this.clearCandidate();
    this.clearReleaseTimers();
    this.clearDoubleTapSubmit();
    this.holdCommitted = false;
    this.restoreAnchor();
    this.recording.cancel();
  }

  handleInput(data: string): void {
    if (isKeyRelease(data)) return;
    if (!this.settings.enabled) {
      super.handleInput(data);
      return;
    }

    const isPrintableSpaces = data.length > 0 && /^ +$/.test(data);
    if (isPrintableSpaces || matchesKey(data, Key.space)) {
      const count = isPrintableSpaces ? data.length : 1;
      if (this.settings.mode === "tap") this.handleTapMode(count);
      else if (!this.handleDoubleTapSubmit(count)) this.handleSpaceEvents(count);
      return;
    }

    if (matchesKey(data, Key.escape) && this.recording.state === "recording") {
      this.clearCandidate();
      this.clearReleaseTimers();
      this.holdCommitted = false;
      this.restoreAnchor();
      this.recording.cancel();
      return;
    }

    // Any other key ends an uncommitted candidate without changing the spaces
    // already typed. During processing, edits are allowed but invalidate the
    // anchored transaction so a late transcript cannot corrupt the prompt.
    if (this.burstCount > 0) this.clearCandidate();
    if (this.awaitingSubmitDoubleTap) this.clearDoubleTapSubmit();
    super.handleInput(data);
  }
}

export default function pushToTalk(pi: ExtensionAPI): void {
  const worker = new WorkerClient();
  const settings = loadVoiceSettings();
  let recording: RecordingController | undefined;
  let editor: PushToTalkEditor | undefined;

  const reportStatus = (ctx: ExtensionContext) => {
    const state = recording?.state ?? "idle";
    ctx.ui.notify(`Voice: ${settings.enabled ? settings.mode : "off"}. State: ${state}. Hold detection: 5 events / 120 ms burst / 200 ms release. Worker: uv + Python 3.11. Device: ${process.env.PTT_DEVICE ?? "auto"}.`, "info");
  };

  pi.on("session_start", (_event, ctx) => {
    debugEvent({ event: "session_start", hasUI: ctx.hasUI });
    if (!ctx.hasUI) return;
    recording = new RecordingController(worker, ctx);
    const editorFactory = (tui: any, theme: any, keybindings: any) => {
      debugEvent({ event: "editor_factory" });
      editor?.dispose();
      editor = new PushToTalkEditor(tui, theme, keybindings, recording!, settings);
      return editor;
    };
    ctx.ui.setEditorComponent(editorFactory);
    if (
      settings.enabled &&
      process.env.PTT_SKIP_WARM !== "1" &&
      ctx.ui.getEditorComponent() !== undefined
    ) {
      // In-process pi owns the editor locally. Daemon-backed Prime is warmed
      // by ClientStdinVoiceBridge instead, avoiding a duplicate Photon model.
      void worker.warm().catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        ctx.ui.notify(`Push-to-talk could not warm up: ${message}`, "warning");
      });
    }
    if (process.env.PRIME_AGENT_LAUNCHER_PATH) {
      setTimeout(() => {
        if (!settings.enabled || !editor) return;
        // Prime finalizes its prompt UI after session_start. Reinstall once
        // after that reset; the editor snapshot preserves any prompt text.
        debugEvent({ event: "editor_factory_reattach" });
        ctx.ui.setEditorComponent(editorFactory);
      }, 2000);
    }
    if (settings.enabled) {
      const instruction = settings.mode === "tap" ? "tap Space to record" : "hold Space to speak";
      ctx.ui.notify(`Voice mode enabled (${settings.mode}): ${instruction}`, "info");
    }
  });

  pi.registerCommand("voice", {
    description: "Toggle hold-Space voice input; use /voice status to inspect it",
    handler: async (args, ctx) => {
      const option = args.trim().toLowerCase();
      if (option === "status") {
        reportStatus(ctx);
        return;
      }
      if (option) {
        ctx.ui.notify("Usage: /voice [status]", "warning");
        return;
      }
      settings.enabled = !settings.enabled;
      settings.mode = "hold";
      if (!settings.enabled) {
        if (editor) editor.cancelFromOutside();
        else recording?.cancel();
      }
      saveVoiceSettings(settings);
      ctx.ui.notify(
        settings.enabled ? "Voice enabled. Hold Space to record." : "Voice disabled.",
        "info",
      );
    },
  });

  pi.on("session_shutdown", async () => {
    debugEvent({ event: "session_shutdown" });
    editor?.dispose();
    worker.close();
  });
}
