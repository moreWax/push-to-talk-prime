# Third-Party Notices

## moondream/parakeet-redux model

This project downloads `moondream/parakeet-redux` on first model warm-up. The model is published by Moondream and is based on NVIDIA's `nvidia/parakeet-tdt-0.6b-v3`.

- Model source: https://huggingface.co/moondream/parakeet-redux
- Base model: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- Model license: Creative Commons Attribution 4.0 International (CC BY 4.0)
- License text: https://creativecommons.org/licenses/by/4.0/legalcode

This package does not redistribute model weights. The weights are downloaded by the runtime. Local configuration changes the streaming preview window and does not modify the downloaded weights.

## Moondream, Kestrel, and Kestrel kernels

The Python environment installs `moondream`, `kestrel`, and platform-specific `kestrel-kernels` from their package indexes. They are not bundled in this source package.

The currently installed `kestrel-kernels` license states that use requires a separate written agreement with M87 Labs and restricts copying/distribution. A public release of this extension must not claim that arbitrary downstream users are licensed to run that engine. Confirm end-user terms with M87 Labs/Moondream before public distribution or replace the serving engine.

Project source outside these third-party components is licensed under the repository's MIT License.
