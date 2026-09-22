import assert from "node:assert/strict";
import test from "node:test";
import type { CustomEditor } from "@earendil-works/pi-coding-agent";
import {
  ClientEditorVoiceBridge,
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
const tick = async () => { await Promise.resolve(); await Promise.resolve(); };

function setup(text = "", settings = holdSettings) {
  const worker = new FakeWorker();
  const bridge = new ClientEditorVoiceBridge({ settings, workerFactory: () => worker, watchSettings: false });
  const editor = new FakeEditor(text);
  const send = (data: string) => bridge.handleInput(editor as unknown as CustomEditor, data, (value) => editor.input(value));
  return { worker, bridge, editor, send };
}

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
  assert.deepEqual(worker.commands.slice(0, 2), ["start", "arm_release"]);
  send(releaseSpace);
  await tick();
  assert.equal(editor.text, "hello world");
  bridge.close();
});

test("interim text replaces the level marker and final text replaces interim", async () => {
  const { worker, bridge, editor, send } = setup();
  worker.stopResult = Promise.resolve({ ok: true, event: "transcript", text: "hello world" });
  for (let i = 0; i < 5; i++) send(" ");
  worker.onInterim?.("hello wor");
  assert.match(editor.text, /^hello/);
  send(releaseSpace);
  await tick();
  assert.equal(editor.text, "hello world");
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
