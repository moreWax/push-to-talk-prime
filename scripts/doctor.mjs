#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { requireSupportedPlatform } from "./platform.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
requireSupportedPlatform();
const command = process.env.PTT_UV || "uv";
const modelCheck = process.argv.includes("--model");
const workerFlag = modelCheck ? "--doctor-model" : "--doctor";
const result = spawnSync(command, ["run", "--locked", "--project", root, "--python", "3.11", "python", join(root, "ptt_worker.py"), workerFlag], { stdio: "inherit" });
if (result.error) {
  console.error(`Could not run uv: ${result.error.message}`);
  console.error("Install uv from https://docs.astral.sh/uv/getting-started/installation/");
}
process.exit(result.status ?? 1);
