#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { requireSupportedPlatform } from "./platform.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
requireSupportedPlatform();
const command = process.env.PTT_UV || "uv";
const check = spawnSync(command, ["--version"], { encoding: "utf8" });
if (check.error || check.status !== 0) {
  console.error("uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/");
  process.exit(1);
}
console.log(check.stdout.trim());
const result = spawnSync(command, ["sync", "--locked", "--project", root, "--python", "3.11"], { stdio: "inherit" });
process.exit(result.status ?? 1);
