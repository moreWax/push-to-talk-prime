# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |
| Earlier development snapshots | No |

Only the latest release line receives security fixes.

## Reporting a vulnerability

Do not open a public issue. Use [GitHub private vulnerability reporting](https://github.com/moreWax/push-to-talk-prime/security/advisories/new).

Include the affected version, OS, Prime/pi version, reproduction steps, impact, and a proposed fix if available. Remove tokens, credentials, usernames, audio, transcripts, settings, and unredacted debug logs.

Maintainers aim to acknowledge reports within three business days and coordinate disclosure. No fixed remediation date is guaranteed.

Relevant security areas include process spawning, IPC authentication, settings/debug-file permissions, microphone/audio disclosure, environment forwarding, model/dependency supply chain, and stale transcript mutation.

For ordinary setup and usage help, see [SUPPORT.md](SUPPORT.md).
