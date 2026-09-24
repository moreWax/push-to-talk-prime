import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";
import type { CustomEditor } from "@earendil-works/pi-coding-agent";
import {
  ClientEditorVoiceBridge,
  SharedWorkerClient,
  audioIndicatorRgb,
  decorateAudioIndicator,
  decorateSpeculativeSuffix,
  type VoiceSettings,
  type VoiceWorker,
} from "../extensions/push-to-talk.ts";

class FakeWorker implements VoiceWorker {
  commands: string[] = [];
  warmed = 0;
  closed = false;
  stopResult: Promise<any> = Promise.resolve({ ok: true, event: "transcript", text: "world" });
  onLevel?: (level: number) => void;
  onInterim?: (text: string) => void;
  onReleaseTimeout?: () => void;
  onFatalError?: (error: Error) => void;
  async warm() { this.warmed++; }
  async request(command: any) {
    this.commands.push(command);
    if (command === "stop") return this.stopResult;
    return { ok: true, event: command };
  }
  close() { this.closed = true; }
}

class FakeEditor {
  text: string;
  cursor: number;
  renders = 0;
  tui = { requestRender: () => { this.renders++; } };
  constructor(text = "") { this.text = text; this.cursor = text.length; }
  getText() { return this.text; }
  getLines() { return this.text.split("\n"); }
  getCursor() { return { line: 0, col: this.cursor }; }
  setText(value: string) { this.text = value; this.cursor = value.length; }
  input(data: string) {
    if (data === "\x7f") {
      if (this.cursor > 0) {
        this.text = this.text.slice(0, this.cursor - 1) + this.text.slice(this.cursor);
        this.cursor--;
      }
    } else if (data === "\x1b[D") {
      this.cursor = Math.max(0, this.cursor - 1);
    } else if (data.length === 1 && data >= " ") {
      this.text = this.text.slice(0, this.cursor) + data + this.text.slice(this.cursor);
      this.cursor += data.length;
    }
  }
}

const holdSettings: VoiceSettings = { enabled: true, mode: "hold", autoSubmit: false, preset: "balanced", device: "auto" };
const releaseSpace = "\x1b[32;1:3u";
const tick = () => new Promise<void>((resolve) => setImmediate(resolve));

function setup(text = "", settings = holdSettings) {
  const worker = new FakeWorker();
  const bridge = new ClientEditorVoiceBridge({ settings, workerFactory: () => worker, watchSettings: false });
  const editor = new FakeEditor(text);
  const send = (data: string) => bridge.handleInput(editor as unknown as CustomEditor, data, (value) => editor.input(value));
  return { worker, bridge, editor, send };
}

test("asynchronous settings watcher errors are handled and closed", () => {
  class FakeWatcher extends EventEmitter {
    closed = false;
    unref() { return this; }
    close() { this.closed = true; }
  }
  const watcher = new FakeWatcher();
  const bridge = new ClientEditorVoiceBridge({
    settings: holdSettings,
    workerFactory: () => new FakeWorker(),
    watchFactory: (() => watcher) as any,
  });
  assert.doesNotThrow(() => watcher.emit("error", Object.assign(new Error("too many files"), { code: "EMFILE" })));
  assert.equal(watcher.closed, true);
  bridge.close();
});

test("synchronous settings watcher failures do not escape construction", () => {
  assert.doesNotThrow(() => {
    const bridge = new ClientEditorVoiceBridge({
      settings: holdSettings,
      workerFactory: () => new FakeWorker(),
      watchFactory: (() => { throw Object.assign(new Error("too many files"), { code: "EMFILE" }); }) as any,
    });
    bridge.close();
  });
});

test("model warm-up timeout clears stale connection for retry", async () => {
  const client = new SharedWorkerClient("balanced", "cpu", 1, false) as any;
  let destroyed = false;
  client.socket = { destroy: () => { destroyed = true; } };
  client.ready = Promise.resolve();
  await assert.rejects(client.waitForModel(), /warm-up timed out/);
  assert.equal(destroyed, true);
  assert.equal(client.socket, undefined);
  assert.equal(client.ready, undefined);
  assert.equal(client.modelReadyPromise, undefined);
  assert.equal(client.rejectModelReady, undefined);
});

test("normal Space typing is preserved", () => {
  const { bridge, editor, send } = setup();
  send(" ");
  send(releaseSpace);
  send("x");
  assert.equal(editor.text, " x");
  bridge.close();
});

test("hold removes the candidate Space and inserts an anchored final transcript", async () => {
  const { worker, bridge, editor, send } = setup("hello");
  for (let i = 0; i < 5; i++) send(" ");
  assert.equal(editor.text, "hello▁");
  await tick();
  assert.deepEqual(worker.commands.slice(0, 2), ["start", "arm_release"]);
  send(releaseSpace);
  await tick();
  assert.equal(editor.text, "hello world");
  bridge.close();
});

test("meter remains beside interim text without storing ANSI", async () => {
  const { worker, bridge, editor, send } = setup();
  worker.stopResult = Promise.resolve({ ok: true, event: "transcript", text: "hello world" });
  for (let i = 0; i < 5; i++) send(" ");
  await tick();
  worker.onInterim?.("hello wor", "hello");
  assert.equal(editor.text, "hello wor▁");
  assert.equal(editor.text.includes("\x1b"), false);
  const renders = editor.renders;
  worker.onLevel?.(0.9);
  assert.equal(editor.text, "hello wor▁");
  assert.equal(editor.renders, renders + 1);
  worker.onLevel?.(0.9);
  assert.ok(editor.renders >= renders + 1);
  send(releaseSpace);
  await tick();
  assert.equal(editor.text, "hello world");
  assert.equal(editor.text.includes("\x1b"), false);
  bridge.close();
});

test("render decoration colors one owned placeholder without changing width", () => {
  const input = ["before ▁ and owned ▂ after"];
  const output = decorateAudioIndicator(input, "▂", 6, [224, 82, 82]);
  assert.equal(input[0], "before ▁ and owned ▂ after");
  assert.match(output[0]!, /before ▁ and owned \x1b\[38;2;224;82;82m▆ after\x1b\[39m/);
  assert.equal(output[0]!.replace(/\x1b\[[0-9;]*m/g, "").length, input[0]!.length);
});

test("render decoration suppresses Prime's adjacent software cursor", () => {
  const cursor = "\x1b_pi:c\x07\x1b[7m \x1b[27m";
  const output = decorateAudioIndicator([`prompt ▁${cursor} rest`], "▁", 4, [82, 224, 224])[0]!;
  assert.equal(output.includes(cursor), false);
  assert.match(output, /prompt \x1b\[38;2;82;224;224m▄  rest\x1b\[39m/);
});

test("indicator color follows Claude's gray threshold and 90-degree hue rotation", () => {
  assert.deepEqual(audioIndicatorRgb(0.149, 9000), [128, 128, 128]);
  assert.deepEqual(audioIndicatorRgb(0.15, 0), [224, 82, 82]);
  assert.deepEqual(audioIndicatorRgb(0.15, 1000), [153, 224, 82]);
  assert.deepEqual(audioIndicatorRgb(0.15, 2000), [82, 224, 224]);
  assert.deepEqual(audioIndicatorRgb(0.15, 3000), [153, 82, 224]);
});

test("render decoration dims only the speculative suffix", () => {
  const input = [" > hello speculative▁"];
  const output = decorateSpeculativeSuffix(input, " speculative");
  assert.equal(output[0], " > hello\x1b[2m speculative\x1b[22m▁");
  assert.equal(output[0]!.replace(/\x1b\[[0-9;]*m/g, ""), input[0]);
  assert.equal(input[0], " > hello speculative▁");
});

test("capture chooses a placeholder absent from existing prompt text", () => {
  const { bridge, editor, send } = setup("existing ▁ block");
  for (let i = 0; i < 5; i++) send(" ");
  assert.equal(editor.text, "existing ▁ block▂");
  assert.equal(editor.text.includes("\x1b"), false);
  bridge.close();
});

test("typing while finalization is pending preserves edits and rejects stale final", async () => {
  const { worker, bridge, editor, send } = setup("hello");
  let resolveStop!: (value: any) => void;
  worker.stopResult = new Promise((resolve) => { resolveStop = resolve; });
  for (let i = 0; i < 5; i++) send(" ");
  send(releaseSpace);
  send("x");
  assert.equal(editor.text, "hellox");
  resolveStop({ ok: true, event: "transcript", text: "world" });
  await tick();
  assert.equal(editor.text, "hellox");
  bridge.close();
});

test("device command restarts the worker", () => {
  const first = new FakeWorker();
  const second = new FakeWorker();
  const workers = [first, second];
  const bridge = new ClientEditorVoiceBridge({ settings: holdSettings, workerFactory: () => workers.shift()!, watchSettings: false });
  const editor = new FakeEditor();
  const send = (data: string) => bridge.handleInput(editor as unknown as CustomEditor, data, (value) => editor.input(value));
  for (const char of "/voice device gpu\r") send(char);
  assert.equal(first.closed, true);
  assert.equal(second.warmed, 1);
  bridge.close();
});

test("preset command restarts the worker with persisted selection", () => {
  const first = new FakeWorker();
  const second = new FakeWorker();
  const workers = [first, second];
  const bridge = new ClientEditorVoiceBridge({ settings: holdSettings, workerFactory: () => workers.shift()!, watchSettings: false });
  const editor = new FakeEditor();
  const send = (data: string) => bridge.handleInput(editor as unknown as CustomEditor, data, (value) => editor.input(value));
  for (const char of "/voice preset smooth\r") send(char);
  assert.equal(first.closed, true);
  assert.equal(second.warmed, 1);
  bridge.close();
});

test("bare voice command toggles the warm worker", () => {
  const first = new FakeWorker();
  const second = new FakeWorker();
  const workers = [first, second];
  const bridge = new ClientEditorVoiceBridge({ settings: holdSettings, workerFactory: () => workers.shift()!, watchSettings: false });
  const editor = new FakeEditor();
  const send = (data: string) => bridge.handleInput(editor as unknown as CustomEditor, data, (value) => editor.input(value));
  for (const char of "/voice\r") send(char);
  assert.equal(first.closed, true);
  for (const char of "/voice\r") send(char);
  assert.equal(second.warmed, 1);
  bridge.close();
});
