# STT-MCP

Let your AI agents understand audio and video. Normalizes audio with FFMPEG -> translates -> cleans up after itself.

Chosen backend implementation depends on your hardware, and your agent will help you choose: Parakeet or Granite.
That selection is persisted for both the CLI and MCP server.

## Backends

| Backend  | Best for                                             | Runtime                                                     |
| -------- | ---------------------------------------------------- | ----------------------------------------------------------- |
| Parakeet | CPU-only Windows/Linux, Intel Mac, and Apple Silicon | Pinned `parakeet.cpp` v0.4.0 plus a local multilingual GGUF |
| Granite  | Highest-quality verified CUDA path                   | Pinned IBM Granite Speech model on exactly one NVIDIA GPU   |

Windows x86-64 is verified with Parakeet CPU and Granite CUDA SDPA. Linux, WSL2, Intel macOS, and
Apple Silicon paths are implemented but remain runtime-unverified. Containers are unsupported.

## Quick start

Follow [`docs/SETUP.md`](docs/SETUP.md) for the authoritative installation and acceptance procedure.
Before continuing, install:

- [uv](https://docs.astral.sh/uv/) with managed CPython 3.13;
- `ffmpeg` and `ffprobe` on `PATH`;
- enough storage for the selected runtime and model.

Dependency and model installation also require network access. Prepare the project and inspect the
machine:

```console
uv --version
uv python install 3.13
ffmpeg -version
ffprobe -version
uv sync --python 3.13 --locked
uv run stt-mcp setup inspect
```

Review the JSON recommendation, then confirm the backend before installing or configuring it.
`setup inspect` never changes configuration.

### Granite

Granite requires a supported NVIDIA/CUDA environment and the optional dependency group:

```console
uv sync --python 3.13 --locked --extra granite
uv run stt-mcp setup configure --backend granite
```

### Parakeet

Download and verify the exact executable and model from the
[Parakeet setup instructions](docs/SETUP.md#4-install-parakeet), then persist their local paths:

```console
uv run stt-mcp setup configure --backend parakeet \
  --parakeet-executable "/path/to/parakeet-cli" \
  --parakeet-model "/path/to/tdt-0.6b-v3-q4_k.gguf" \
  --parakeet-device cpu
```

Parakeet binaries and models are downloaded only during explicit setup, never at runtime.

## Transcribe from the CLI

Publish TXT, Markdown, JSON, SRT, and WebVTT beside the source:

```console
uv run stt-mcp transcribe "/path/to/media.mp4"
```

Choose formats and an output directory:

```console
uv run stt-mcp transcribe "/path/to/media.mp4" --output "/path/to/transcripts" --format json --format srt
```

JSON output includes the effective `backend`. Subtitle timing is coarse 30-second source-window
timing for consistent behavior across both backends.

## Use as an MCP server

Register a supported client, then restart it:

```console
uv run stt-mcp register opencode
uv run stt-mcp register claude-desktop
```

Other stdio clients can launch the installed Python environment with:

```text
-m stt_mcp.server
```

The server exposes:

```text
transcribe(
  source_path: string,
  output_directory?: string,
  formats?: ("txt" | "md" | "json" | "srt" | "vtt")[]
)
```

Omitting `output_directory` writes beside the source. Omitting `formats` publishes every format.

## Operational behavior

- One backend is selected and latched for the MCP process.
- One transcription runs at a time; concurrent requests fail immediately as busy.
- Explicit Granite remains fail-closed under its CUDA/NVML safety policy.
- Cancellation closes the active Parakeet process or invalidates the Granite worker.
- Artifacts are atomically published, so failed requests do not leave partial output.
- Runtime code never downloads Parakeet assets.
- Granite loads only when selected; Parakeet-only imports do not load Torch, Transformers, or NVML.

## Contributor and backend verification

The authoritative verification and real-acceptance sequence is in
[`docs/SETUP.md`](docs/SETUP.md#6-run-static-and-default-test-gates). The default project gates are:

```console
uv sync --python 3.13 --locked --extra granite
uv run ruff check .
uv run basedpyright
uv run ty check src tests
uv run mypy
uv run pytest
uv build
```

Run real acceptance with the configured backend:

```powershell
$env:STT_MCP_TEST_MEDIA = (Resolve-Path "D:\path\to\speech.wav").Path
uv run pytest tests/test_mcp_acceptance.py -q -s
Remove-Item Env:STT_MCP_TEST_MEDIA
```

## Uninstall

```console
uv run stt-mcp unregister opencode
uv run stt-mcp unregister claude-desktop
```

The repository workflow installs STT-MCP into `.venv` through `uv sync`; it does not create a uv
tool installation. Delete `.venv` only if you also want to remove that project environment.
Transcripts, configuration, model caches, and downloaded Parakeet assets remain until you remove
them explicitly.
