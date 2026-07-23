# Setup and Verification Runbook

This is the authoritative setup procedure for STT-MCP setup agents and contributors. Run commands
from the repository root. Do not claim a platform or backend works unless its applicable runtime
checks and real MCP acceptance test pass on that machine.

## 1. Supported deployment paths

| Environment | Backend | Status | Execution |
| --- | --- | --- | --- |
| Windows 10/11 x86-64, bare metal | Parakeet | Supported and locally verified | CPU |
| Windows 10/11 x86-64, bare metal | Granite | Supported and locally verified | CUDA 12.8 with SDPA |
| Linux x86-64, bare metal | Parakeet | Implemented, not locally verified | CPU |
| Linux x86-64, bare metal | Granite | Implemented, not locally verified | CUDA 12.8 with FlashAttention 2 |
| Linux arm64, bare metal | Parakeet | Implemented, not locally verified | CPU |
| WSL2 x86-64 | Parakeet or Granite | Implemented, not locally verified | CPU or CUDA |
| macOS arm64 | Parakeet | Implemented, not locally verified | Metal |
| macOS x86-64 | Parakeet | Implemented, not locally verified | CPU |
| Containers | Any | Unsupported and unverified | Undefined |

Granite remains fail-closed: exactly one NVIDIA GPU must be visible and NVML must report at least
8 GiB globally free before startup. Do not set or remap `CUDA_VISIBLE_DEVICES`. The 8 GiB floor is
conservative startup headroom over a measured 4.607 GiB peak, not intrinsic model consumption.

Parakeet does not use CUDA in STT-MCP. The selected release executable and GGUF must already exist
locally; runtime code never downloads either asset.

## 2. Install common prerequisites

Every path requires:

- [uv](https://docs.astral.sh/uv/) with managed CPython 3.13;
- `ffmpeg` and `ffprobe` on `PATH`;
- enough storage for the selected runtime and model;
- network access while installing dependencies, runtimes, and models.

```console
uv --version
uv python install 3.13
ffmpeg -version
ffprobe -version
uv sync --python 3.13 --locked
uv run stt-mcp setup inspect
```

Use `uv` only. Do not use Conda, direct `pip`, or a system Python environment.
The final command is a common-install smoke check: it must work before a backend is selected and
must not require the Granite extra.

If `uv python install 3.13` reports that an unmanaged `python3.13.exe` already exists, review that
executable before replacing it. Use `uv python install 3.13 --force` only when replacing that shim
is intentional, then rerun the commands above. Do not continue with a different Python version.

## 3. Inspect hardware and confirm the backend

Run the machine-readable inspection before installing a backend:

```console
uv run stt-mcp setup inspect
```

The JSON reports platform, architecture, NVIDIA GPU count, maximum free VRAM, a concrete
`recommended_backend`, an optional `parakeet_device`, and a stable reason code.

The setup agent must:

1. show the recommendation and the hardware facts that produced it;
2. ask the user to confirm Granite or Parakeet, while offering the supported alternative;
3. install and verify only the confirmed backend;
4. persist that concrete choice with `stt-mcp setup configure`.

Do not silently substitute the recommendation. A missing configuration defaults to Granite only
for compatibility with releases that predate backend selection; new setups must persist a choice.

Default configuration locations are `%LOCALAPPDATA%\stt-mcp\config.json` on Windows and
`${XDG_CONFIG_HOME:-~/.config}/stt-mcp/config.json` on POSIX. `STT_MCP_CONFIG` can point CLI and MCP
processes at an alternate configuration for isolated acceptance testing.

## 4. Install Parakeet

STT-MCP pins:

```text
parakeet.cpp release=v0.4.0
parakeet.cpp source commit=1da853421de9710cbe894a0110711de5a0516486
model repository=mudler/parakeet-cpp-gguf
model revision=399aa8eab0c12f128f2cc562277c30d99cfd7bdc
model file=tdt-0.6b-v3-q4_k.gguf
model SHA-256=993d73feb4206dadda865ab25bd64b50c48dc4d013c3bf6126a721f28b1d5ee8
```

The parakeet.cpp runtime is MIT licensed. The pinned model is CC-BY-4.0; preserve its attribution
when redistributing it. Review both upstream licenses before redistribution.

### Release archives

| Platform | Archive | SHA-256 | Device |
| --- | --- | --- | --- |
| Windows x64 | `parakeet-v0.4.0-bin-win-cpu-x64.zip` | `2880150a1bad2944baed46f2e6bb9f1bc55263a9f2bb85573785a7ec4fa35f27` | `cpu` |
| Linux x64 | `parakeet-v0.4.0-bin-linux-cpu-x64.tar.gz` | `0846509eeb64fcb40e0ad28cd16b5bec5387e4799e08c85fb600b428bb306240` | `cpu` |
| Linux arm64 | `parakeet-v0.4.0-bin-linux-cpu-arm64.tar.gz` | `6634487a4cdbd3185e7a127aa4f22fbc49ec56421f7bfb14f450400260597773` | `cpu` |
| macOS x64 | `parakeet-v0.4.0-bin-macos-cpu-x64.tar.gz` | `6f985e7a7185646e97a2d4fa7953b2019327ad56ad677f0602c666745d036a8d` | `cpu` |
| macOS arm64 | `parakeet-v0.4.0-bin-macos-metal-arm64.tar.gz` | `e607d8700bec29c5bf8fa2e8155adfbf92d4433d98608a9dd866633ea7d01767` | `metal` |

### Windows PowerShell

```powershell
$Root = Join-Path $env:LOCALAPPDATA "stt-mcp\parakeet"
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$Archive = Join-Path $Root "parakeet-v0.4.0-bin-win-cpu-x64.zip"
$ArchiveUrl = "https://github.com/mudler/parakeet.cpp/releases/download/v0.4.0/parakeet-v0.4.0-bin-win-cpu-x64.zip"
$ArchiveSha = "2880150a1bad2944baed46f2e6bb9f1bc55263a9f2bb85573785a7ec4fa35f27"
Invoke-WebRequest $ArchiveUrl -OutFile $Archive
if ((Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ArchiveSha) { throw "Parakeet archive checksum mismatch" }
Expand-Archive $Archive -DestinationPath $Root -Force

$Model = Join-Path $Root "tdt-0.6b-v3-q4_k.gguf"
$ModelUrl = "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/399aa8eab0c12f128f2cc562277c30d99cfd7bdc/tdt-0.6b-v3-q4_k.gguf?download=true"
$ModelSha = "993d73feb4206dadda865ab25bd64b50c48dc4d013c3bf6126a721f28b1d5ee8"
Invoke-WebRequest $ModelUrl -OutFile $Model
if ((Get-FileHash $Model -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ModelSha) { throw "Parakeet model checksum mismatch" }

$Executable = (Get-ChildItem $Root -Recurse -Filter "parakeet-cli.exe" | Select-Object -First 1).FullName
if (-not $Executable) { throw "parakeet-cli.exe was not found in the release archive" }
uv run stt-mcp setup configure --backend parakeet --parakeet-executable $Executable --parakeet-model $Model --parakeet-device cpu
```

### Linux

Linux requires `curl`, `tar`, and `sha256sum`. The following command selects the pinned archive for
the current supported architecture:

```bash
ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/stt-mcp/parakeet"
mkdir -p "$ROOT"

case "$(uname -m)" in
  x86_64|amd64)
    ARCHIVE='parakeet-v0.4.0-bin-linux-cpu-x64.tar.gz'
    ARCHIVE_SHA='0846509eeb64fcb40e0ad28cd16b5bec5387e4799e08c85fb600b428bb306240'
    ;;
  aarch64|arm64)
    ARCHIVE='parakeet-v0.4.0-bin-linux-cpu-arm64.tar.gz'
    ARCHIVE_SHA='6634487a4cdbd3185e7a127aa4f22fbc49ec56421f7bfb14f450400260597773'
    ;;
  *)
    echo "Unsupported Linux architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

curl -fL "https://github.com/mudler/parakeet.cpp/releases/download/v0.4.0/$ARCHIVE" -o "$ROOT/$ARCHIVE"
printf '%s  %s\n' "$ARCHIVE_SHA" "$ROOT/$ARCHIVE" | sha256sum -c -
tar -xzf "$ROOT/$ARCHIVE" -C "$ROOT"

MODEL="$ROOT/tdt-0.6b-v3-q4_k.gguf"
MODEL_SHA='993d73feb4206dadda865ab25bd64b50c48dc4d013c3bf6126a721f28b1d5ee8'
curl -fL 'https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/399aa8eab0c12f128f2cc562277c30d99cfd7bdc/tdt-0.6b-v3-q4_k.gguf?download=true' -o "$MODEL"
printf '%s  %s\n' "$MODEL_SHA" "$MODEL" | sha256sum -c -

EXECUTABLE="$(find "$ROOT" -type f -name parakeet-cli -print -quit)"
[ -n "$EXECUTABLE" ] || { echo "parakeet-cli was not found in the release archive" >&2; exit 1; }
uv run stt-mcp setup configure --backend parakeet \
  --parakeet-executable "$EXECUTABLE" \
  --parakeet-model "$MODEL" \
  --parakeet-device cpu
```

### macOS

macOS requires `curl`, `tar`, and the built-in `shasum`. Intel Macs use CPU; Apple Silicon uses the
Metal archive and device:

```bash
ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/stt-mcp/parakeet"
mkdir -p "$ROOT"

case "$(uname -m)" in
  x86_64)
    ARCHIVE='parakeet-v0.4.0-bin-macos-cpu-x64.tar.gz'
    ARCHIVE_SHA='6f985e7a7185646e97a2d4fa7953b2019327ad56ad677f0602c666745d036a8d'
    DEVICE='cpu'
    ;;
  arm64)
    ARCHIVE='parakeet-v0.4.0-bin-macos-metal-arm64.tar.gz'
    ARCHIVE_SHA='e607d8700bec29c5bf8fa2e8155adfbf92d4433d98608a9dd866633ea7d01767'
    DEVICE='metal'
    ;;
  *)
    echo "Unsupported macOS architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

curl -fL "https://github.com/mudler/parakeet.cpp/releases/download/v0.4.0/$ARCHIVE" -o "$ROOT/$ARCHIVE"
[ "$(shasum -a 256 "$ROOT/$ARCHIVE" | awk '{print $1}')" = "$ARCHIVE_SHA" ] || {
  echo "Parakeet archive checksum mismatch" >&2
  exit 1
}
tar -xzf "$ROOT/$ARCHIVE" -C "$ROOT"

MODEL="$ROOT/tdt-0.6b-v3-q4_k.gguf"
MODEL_SHA='993d73feb4206dadda865ab25bd64b50c48dc4d013c3bf6126a721f28b1d5ee8'
curl -fL 'https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/399aa8eab0c12f128f2cc562277c30d99cfd7bdc/tdt-0.6b-v3-q4_k.gguf?download=true' -o "$MODEL"
[ "$(shasum -a 256 "$MODEL" | awk '{print $1}')" = "$MODEL_SHA" ] || {
  echo "Parakeet model checksum mismatch" >&2
  exit 1
}

EXECUTABLE="$(find "$ROOT" -type f -name parakeet-cli -print -quit)"
[ -n "$EXECUTABLE" ] || { echo "parakeet-cli was not found in the release archive" >&2; exit 1; }
uv run stt-mcp setup configure --backend parakeet \
  --parakeet-executable "$EXECUTABLE" \
  --parakeet-model "$MODEL" \
  --parakeet-device "$DEVICE"
```

Do not use a model alias or `parakeet-server`; those paths may download models at runtime.

## 5. Install Granite

Granite supports Windows x86-64, Linux x86-64, and WSL2 x86-64 only.

### Install dependencies

```console
uv sync --python 3.13 --locked --extra granite
uv run stt-mcp setup configure --backend granite
```

### Validate CUDA

Install a current NVIDIA driver compatible with CUDA 12.8. Linux and WSL2 also require the CUDA
Toolkit, a C/C++ toolchain, and `nvcc` because the locked environment builds FlashAttention 2.
WSL2 must use the Windows host's WSL-capable NVIDIA driver, not a Linux display driver inside WSL.

Verify exactly one GPU and the global free-memory floor:

```console
nvidia-smi --list-gpus
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
```

Verify PyTorch:

```console
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Linux and WSL2 must also run:

```console
nvcc --version
uv run python -c "import flash_attn; print(flash_attn.__version__)"
```

### Download and verify the model

The reviewed Granite model boundary is:

```text
MODEL_ID=ibm-granite/granite-speech-4.1-2b-nar
MODEL_REVISION=99a4df9007ac5682f9daa093fb7008ff606e9a5d
```

Download and verify the exact snapshot:

```console
uv run hf download ibm-granite/granite-speech-4.1-2b-nar --revision 99a4df9007ac5682f9daa093fb7008ff606e9a5d --quiet
uv run hf cache verify ibm-granite/granite-speech-4.1-2b-nar --revision 99a4df9007ac5682f9daa093fb7008ff606e9a5d --fail-on-missing-files
```

`GraniteTranscriber.load()` uses `trust_remote_code=True`. Review Python files and upstream diffs
before changing the revision. Never replace the commit with `main`, a branch, or an unreviewed tag.

## 6. Run static and default test gates

Contributors should install the Granite extra because the complete test suite covers both engines:

```console
uv sync --python 3.13 --locked --extra granite
uv run ruff check .
uv run basedpyright
uv run ty check src tests
uv run mypy
uv run pytest
uv build
```

The normal pytest run skips the real backend acceptance unless `STT_MCP_TEST_MEDIA` is set.
If a Windows tool invocation reports `uv trampoline failed to canonicalize script path`, repair only
the affected locked launcher and rerun its exact gate:

```console
uv sync --python 3.13 --locked --reinstall-package <tool-name>
```

For example, use `basedpyright`, `mypy`, or `pytest` as `<tool-name>`; do not skip the gate or
substitute a globally installed tool.

## 7. Run real CLI and MCP acceptance

Use 5 to 30 seconds of clear speech, not silence or a pure tone. Longer recordings are accepted,
but make the feedback cycle and acceptance run proportionally slower.

### Repository reference media

`assets/test-audio.mp3` is the tracked reference fixture for repeatable local acceptance. Its
SHA-256 is `dfb6ef4cc9ad03ba54e24026ab734a56bf7e3251751e634e88f6f384402bff45` and its duration is
256.068 seconds. The ordinary pytest run still skips real inference by default; use this fixture
explicitly when a full-length reference acceptance run is appropriate:

```powershell
$Media = (Resolve-Path "assets\test-audio.mp3").Path
$env:STT_MCP_TEST_MEDIA = $Media
uv run pytest tests/test_mcp_acceptance.py -q -s
Remove-Item Env:STT_MCP_TEST_MEDIA
```

### CLI smoke test

```powershell
$Media = (Resolve-Path "D:\path\to\speech.wav").Path
uv run stt-mcp transcribe $Media --output ".artifacts\cli-smoke" --format json --format srt --format vtt
```

Inspect the JSON for the confirmed backend and useful recognition. Verify that the CLI published
JSON, SRT, and VTT. Timing remains coarse 30-second source-window timing.

### FastMCP acceptance

```powershell
$env:STT_MCP_TEST_MEDIA = $Media
uv run pytest tests/test_mcp_acceptance.py -q -s
Remove-Item Env:STT_MCP_TEST_MEDIA
```

The acceptance test must use the configured backend and publish SRT and VTT artifacts.

### Process cleanup

After MCP teardown, Windows must show no Parakeet CLI or Granite worker:

```powershell
$SelfPid = $PID
Get-CimInstance Win32_Process |
  Where-Object {
    $_.ProcessId -ne $SelfPid -and
    ($_.CommandLine -like "*parakeet-cli*" -or $_.CommandLine -like "*stt_mcp.worker*")
  }
```

POSIX systems must show no matching process:

```bash
pgrep -af 'parakeet-cli|stt_mcp\.worker' || true
```

## 8. Register an MCP client

After acceptance passes:

```console
uv run stt-mcp register opencode
uv run stt-mcp register claude-desktop
```

Register only installed clients. Restart the client, confirm the `transcribe` tool is listed, and
call it with the same absolute speech path. Registration launches `-m stt_mcp.server`; backend
selection remains in the persisted STT-MCP configuration, not in each client configuration.

### Codex

The `stt-mcp register` command updates JSON-configured clients. Codex manages its own TOML MCP
configuration, so register it through the installed Codex CLI after acceptance passes:

```powershell
$Python = (Resolve-Path ".venv\Scripts\python.exe").Path
codex mcp add stt-mcp -- $Python -m stt_mcp.server
codex mcp get stt-mcp
```

This creates a global Codex stdio-server entry. Restart the Codex CLI, app, or IDE extension, then
use `/mcp` or the MCP server settings to confirm that `stt-mcp` is enabled. To remove it later,
run `codex mcp remove stt-mcp`.

## 9. Cleanup

```console
uv run stt-mcp unregister opencode
uv run stt-mcp unregister claude-desktop
```

The repository workflow installs STT-MCP into `.venv` through `uv sync`; it does not create a uv
tool installation. Delete `.venv` only when you intentionally want to remove that project
environment. Transcript artifacts, configuration, downloaded Parakeet assets, and the Hugging Face
cache remain until explicitly removed.

## 10. Required completion report

An agent declaring setup complete must report:

- OS, architecture, and bare-metal or WSL2 status;
- inspection JSON, recommended backend, user-confirmed backend, and persisted config path;
- exact runtime/model revisions and successful SHA-256 or Hugging Face cache verification;
- FFmpeg/FFprobe versions and Granite CUDA details when applicable;
- static gates, default pytest, build, CLI smoke, and MCP acceptance outcomes;
- transcript sanity observations and recognition errors;
- zero surviving backend processes;
- every unverified platform or backend path.
