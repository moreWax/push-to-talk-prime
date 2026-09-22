# Releasing

- [ ] `main` is clean and CI is green.
- [ ] Third-party model/engine terms and notices are reviewed.
- [ ] `package.json` and `pyproject.toml` versions match SemVer release.
- [ ] `CHANGELOG.md` includes the release date and user-visible changes.
- [ ] Lockfiles changed only when dependency inputs changed.
- [ ] `npm ci` succeeds.
- [ ] `uv sync --locked --python 3.11` succeeds.
- [ ] `npm test` passes.
- [ ] `npm run doctor` passes on claimed platforms.
- [ ] `npm run doctor:model` passes on each claimed accelerator class where practical.
- [ ] Clean Prime install validates `/voice status`, hold/release/cancel, restart, update, and rollback.
- [ ] Two Prime clients connect to one global worker; owner disconnect releases capture.
- [ ] `npm pack --dry-run` contains no secrets, settings, logs, recordings, model weights, `.venv`, or `node_modules`.
- [ ] README, notices, architecture, troubleshooting, and support docs are current.
- [ ] Create an annotated `vX.Y.Z` tag and GitHub release from the changelog.
- [ ] If publishing npm, use 2FA/provenance and verify exact-version install from a clean directory.
- [ ] Document rollback, yank, or deprecation action if a release is broken.
