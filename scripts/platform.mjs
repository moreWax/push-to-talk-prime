import { platform, arch, release } from "node:os";

export function validatePlatform() {
  const os = platform();
  const cpu = arch();
  if (os === "darwin") {
    if (cpu !== "arm64") return { ok: false, error: `Unsupported macOS architecture: ${cpu}. Apple silicon is required.` };
    const darwinMajor = Number.parseInt(release().split(".")[0] || "0", 10);
    if (darwinMajor < 23) return { ok: false, error: "macOS 14 or newer is required by the locked native wheel set." };
    return { ok: true, platform: "macos-arm64" };
  }
  if (os === "win32") {
    return cpu === "x64"
      ? { ok: true, platform: "windows-x64" }
      : { ok: false, error: `Unsupported Windows architecture: ${cpu}. x64 is required.` };
  }
  if (os === "linux") {
    return cpu === "x64" || cpu === "arm64"
      ? { ok: true, platform: `linux-${cpu}` }
      : { ok: false, error: `Unsupported Linux architecture: ${cpu}. x64 or arm64 is required.` };
  }
  return { ok: false, error: `Unsupported operating system: ${os}.` };
}

export function requireSupportedPlatform() {
  const result = validatePlatform();
  if (!result.ok) {
    console.error(result.error);
    process.exit(1);
  }
  return result;
}
