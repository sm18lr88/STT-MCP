# Agent Setup and Verification Runbook

This is the authoritative setup procedure for STT-MCP. Run commands from the repository root.
Do not claim a platform works unless its runtime checks and the real MCP acceptance test pass on
that platform.

## 1. Choose the supported deployment path

| Environment                           | Status                                | CUDA backend     | Notes                                                           |
| ------------------------------------- | ------------------------------------- | ---------------- | --------------------------------------------------------------- |
| Windows 10/11 x86-64, bare metal      | Supported and locally verified        | PyTorch SDPA     | Verified on an RTX 4090 Laptop GPU                              |
| Linux x86-64, bare metal              | Implemented, not yet locally verified | FlashAttention 2 | Requires a CUDA build toolchain                                 |
| WSL2 with a Linux x86-64 distribution | Implemented, not yet locally verified | FlashAttention 2 | Uses the Windows host's WSL-capable NVIDIA driver               |
| macOS                                 | Unsupported                           | None             | Apple MPS and CPU fallbacks are intentionally absent            |
| Docker or other containers            | Not supported or verified             | Undefined        | Use native bare metal or WSL2 until a container contract exists |

STT-MCP fails closed unless exactly one NVIDIA GPU is visible. Do not set or remap
`CUDA_VISIBLE_DEVICES`. At model startup, NVIDIA NVML must satisfy the server's conservative 8 GiB
globally free safety floor.

### What the 8 GiB floor means

It is a startup policy, not a demonstrated model minimum. On the verified Windows RTX 4090 Laptop
GPU path, a real 30-second transcription measured:

- 4.075 GiB additional global VRAM after model load;
- 4.454 GiB additional global VRAM during inference;
- 4.607 GiB peak PyTorch reserved memory.

The 8 GiB floor deliberately leaves roughly 3.4 GiB beyond that observed peak because Windows
native model startup previously crashed without a Python exception while a competing Topaz CUDA
workload was active. GPUs or shared workloads with less than 8 GiB free may be technically capable
of inference, but the current server rejects them because that configuration has not passed a
constrained-VRAM MCP acceptance test. Do not describe 8 GiB as intrinsic model consumption.

## 2. Install common prerequisites

All platforms need:

- [uv](https://docs.astral.sh/uv/) with a managed CPython 3.13 installation;
- `ffmpeg` and `ffprobe` on `PATH`;
- enough disk space for the CUDA PyTorch environment and Granite model cache;
- network access to Hugging Face for the initial pinned snapshot download.

Verify the common tools:

```console
uv --version
uv python install 3.13
ffmpeg -version
ffprobe -version
```

Use `uv` only. Do not create a Conda environment, run `pip`, or use a system Python installation.

## 3. Configure CUDA by operating system

### Windows bare metal

1. Install a current NVIDIA Windows driver that supports CUDA 12.8, then reboot if requested.
   Use NVIDIA's driver installer or your managed enterprise driver channel.
2. Do **not** install the full CUDA Toolkit solely for STT-MCP on Windows. The pinned PyTorch wheel
   includes its CUDA runtime, and the Windows path uses SDPA rather than compiling FlashAttention.
3. Install FFmpeg and uv through their official installers or a trusted package manager.
4. Verify the driver and global memory:

```powershell
nvidia-smi
nvidia-smi --list-gpus
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
```

Exactly one GPU must be listed. The current server additionally enforces its 8192 MiB globally free
startup safety floor; this is conservative headroom over the measured 4.607 GiB peak, not the
model's minimum VRAM requirement. Pause video enhancers, local LLMs, games, and other CUDA
workloads if necessary. On Windows WDDM, do not use `torch.cuda.mem_get_info()` as the global-memory
authority; STT-MCP intentionally uses NVML.

### Linux bare metal

1. Install a current NVIDIA Linux driver and the CUDA Toolkit from NVIDIA's instructions for the
   exact distribution: <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/>.
2. Install a C/C++ build toolchain and FFmpeg. On Debian/Ubuntu this normally includes
   `build-essential` and `ffmpeg`.
3. Verify both the driver and compiler:

```bash
nvidia-smi
nvidia-smi --list-gpus
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
nvcc --version
gcc --version
ffmpeg -version
```

The CUDA Toolkit and compiler are required because `uv sync` builds the pinned FlashAttention 2
dependency for the Linux path.

### WSL2

1. Install WSL2 and an x86-64 Linux distribution.
2. Install an NVIDIA Windows host driver with WSL CUDA support. Follow
   <https://docs.nvidia.com/cuda/wsl-user-guide/>.
3. Do **not** install a Linux NVIDIA display/kernel driver inside WSL. Install only the WSL/Linux
   CUDA Toolkit and normal Linux build prerequisites inside the distribution.
4. Keep the checkout in the WSL Linux filesystem rather than under `/mnt/c` for build performance.
5. Run the Linux verification commands above inside WSL. `nvidia-smi` must see exactly one GPU.

## 4. Create the locked project environment

```console
uv sync --python 3.13 --locked
```

Verify the selected runtime:

```console
uv run python -c "import torch; print('torch=', torch.__version__); print('cuda_build=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Required results:

- PyTorch reports a CUDA 12.8 build.
- `cuda_available=True`.
- `device_count=1`.
- The expected NVIDIA GPU name is printed.

On Linux and WSL2, also verify the compiled backend:

```bash
uv run python -c "import flash_attn; print(flash_attn.__version__)"
```

## 5. Download and verify the pinned Hugging Face model

The code authority is `MODEL_ID` and `MODEL_REVISION` in `src/stt_mcp/model.py`. At the time of
this runbook, they are:

```text
MODEL_ID=ibm-granite/granite-speech-4.1-2b-nar
MODEL_REVISION=99a4df9007ac5682f9daa093fb7008ff606e9a5d
```

First confirm that the checked-out code still has those values:

```console
uv run python -c "from stt_mcp.model import MODEL_ID, MODEL_REVISION; print(MODEL_ID); print(MODEL_REVISION)"
```

Download and verify every file against the exact remote revision.

PowerShell:

```powershell
$ModelId = "ibm-granite/granite-speech-4.1-2b-nar"
$Revision = "99a4df9007ac5682f9daa093fb7008ff606e9a5d"
$env:PYTHONUTF8 = "1"
$Snapshot = uv run hf download $ModelId --revision $Revision --quiet
uv run hf cache verify $ModelId --revision $Revision --fail-on-missing-files
$Snapshot
Remove-Item Env:PYTHONUTF8
```

`PYTHONUTF8=1` prevents the Hugging Face CLI's Unicode verification status symbols from failing on
Windows installations whose active console code page cannot encode them.

Bash:

```bash
MODEL_ID='ibm-granite/granite-speech-4.1-2b-nar'
REVISION='99a4df9007ac5682f9daa093fb7008ff606e9a5d'
SNAPSHOT="$(uv run hf download "$MODEL_ID" --revision "$REVISION" --quiet)"
uv run hf cache verify "$MODEL_ID" --revision "$REVISION" --fail-on-missing-files
printf '%s\n' "$SNAPSHOT"
```

`GraniteTranscriber.load()` uses `trust_remote_code=True`, so Python files from this exact snapshot
execute locally. Before changing the revision, inspect the snapshot's Python files and review the
upstream diff. Never replace the commit hash with `main`, a branch, or an unreviewed tag. Checksum
verification proves cache integrity against the selected Hub revision; it does not make unreviewed
remote code trustworthy.

## 6. Run deterministic project gates

```console
uv run ruff check .
uv run basedpyright
uv run ty check src tests
uv run mypy
uv run pytest
uv build
```

The normal pytest suite skips the expensive real-model MCP acceptance test unless a media path is
explicitly supplied.

## 7. Run a real CLI transcription

Choose a local file containing 5 to 30 seconds of clear speech. Avoid synthetic silence or a pure
tone because that does not validate speech recognition.

PowerShell:

```powershell
$Media = (Resolve-Path "D:\path\to\speech.wav").Path
uv run stt-mcp transcribe $Media --output ".artifacts\cli-smoke" --format json --format srt
```

Bash:

```bash
MEDIA="$(realpath '/path/to/speech.wav')"
uv run stt-mcp transcribe "$MEDIA" --output '.artifacts/cli-smoke' --format json --format srt
```

Inspect the JSON and SRT. The transcript should match the audible speech well enough to be useful.
Timing is intentionally coarse 30-second source-window timing, not word timing.

## 8. Run the real MCP acceptance test

This test drives the FastMCP `transcribe` tool through an in-memory MCP client/server session,
loads the real CUDA model, transcribes the configured media, publishes SRT and VTT, and lets the
server lifespan close the worker.

PowerShell:

```powershell
$env:STT_MCP_TEST_MEDIA = (Resolve-Path "D:\path\to\speech.wav").Path
uv run pytest tests/test_mcp_acceptance.py -q -s
Remove-Item Env:STT_MCP_TEST_MEDIA
```

Bash:

```bash
STT_MCP_TEST_MEDIA="$(realpath '/path/to/speech.wav')" \
  uv run pytest tests/test_mcp_acceptance.py -q -s
```

Acceptance requires:

- pytest exits successfully;
- both SRT and VTT files are published in pytest's isolated temporary directory;
- no partial artifact is reported;
- no `stt_mcp.worker` process survives after pytest exits.

On Windows, check worker cleanup with:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -like "python*" -and $_.CommandLine -like "*stt_mcp.worker*" }
```

On Linux/WSL2:

```bash
pgrep -af 'stt_mcp\.worker' || true
```

No process rows are expected.

## 9. Register and test an external MCP client

After the in-memory acceptance test passes:

```console
uv run stt-mcp register opencode
uv run stt-mcp register claude-desktop
```

Register only clients installed on the machine. Restart the client, inspect that the `transcribe`
tool is listed, and call it with the same absolute speech-file path. Client registration details and
the manual stdio command are in [`../README.md`](../README.md#register-an-mcp-client).

## 10. Cleanup and uninstall

Temporary normalized audio is deleted automatically. Transcript artifacts and the Hugging Face
model cache are persistent by design.

```console
uv run stt-mcp unregister opencode
uv run stt-mcp unregister claude-desktop
uv tool uninstall stt-mcp
```

Remove `.artifacts/` when its transcripts are no longer needed. To reclaim model storage, remove
the Hugging Face cache entry for `ibm-granite/granite-speech-4.1-2b-nar` only after confirming that
no STT-MCP worker is running.

## 11. Evidence to report

An agent declaring setup complete must report:

- OS and whether it is bare metal or WSL2;
- GPU model, driver version, global free VRAM, PyTorch version, and CUDA build;
- exact model ID and revision plus successful `hf cache verify` output;
- FFmpeg/FFprobe versions;
- static gates, default pytest, build, CLI smoke, and MCP acceptance outcomes;
- transcript sanity observations and known recognition errors;
- zero worker processes after MCP teardown;
- any unverified path, especially Linux/WSL2 FlashAttention runtime behavior.
