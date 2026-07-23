# STT-MCP

A fast, local speech-to-text MCP server that lets AI agents understand audio and video; requires Python 3.13, `uv`, FFmpeg, and exactly one NVIDIA CUDA GPU, and currently enforces an 8 GiB globally free VRAM startup safety floor.

STT-MCP uses the pinned
[`ibm-granite/granite-speech-4.1-2b-nar`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar)
model and publishes TXT, Markdown, JSON, SRT, and WebVTT locally.

## Setup

Follow [`docs/SETUP.md`](docs/SETUP.md) for CUDA, FFmpeg, model download and checksum
verification, Windows/Linux/WSL2 instructions, and real MCP acceptance testing.

- Windows x86-64 is verified with CUDA SDPA.
- Linux x86-64 and WSL2 use FlashAttention 2 but remain runtime-unverified.
- macOS, CPU/MPS fallback, containers, multi-GPU setups, and remapped
  `CUDA_VISIBLE_DEVICES` are unsupported.
- The observed Windows inference peak was about 4.61 GiB. The enforced 8 GiB free-VRAM floor is
  conservative startup headroom, not intrinsic model consumption.

## Install

```console
uv tool install --python 3.13 .
stt-mcp --help
```

For development:

```console
uv sync --python 3.13 --locked
uv run stt-mcp --help
```

## Transcribe from the CLI

Publish all five formats beside the source:

```console
stt-mcp transcribe "/path/to/media.mp4"
```

Choose the output directory and formats:

```console
stt-mcp transcribe "/path/to/media.mp4" --output "/path/to/transcripts" --format json --format srt
```

## Use as an MCP server

Register a supported client, then restart it:

```console
stt-mcp register opencode
stt-mcp register claude-desktop
```

Other stdio clients can launch the installed environment's Python with:

```text
-m stt_mcp.server
```

The server exposes one tool:

```text
transcribe(
  source_path: string,
  output_directory?: string,
  formats?: ("txt" | "md" | "json" | "srt" | "vtt")[]
)
```

Omitting `output_directory` writes beside the source. Omitting `formats` publishes every format.

## Operational behavior

- One transcription runs at a time; concurrent requests fail immediately as busy.
- Cancellation terminates and invalidates the worker.
- Artifacts are atomically published, so failed requests do not leave partial output.
- Subtitle timing is coarse 30-second source-window timing because the model does not provide
  sentence or word timestamps.
- The first transcription downloads the exact model revision pinned in `src/stt_mcp/model.py`.

## Verify

```console
uv run ruff check .
uv run basedpyright
uv run ty check src tests
uv run mypy
uv run pytest
uv build
```

Run the real CUDA/MCP acceptance test with a local speech file:

```powershell
$env:STT_MCP_TEST_MEDIA = (Resolve-Path "D:\path\to\speech.wav").Path
uv run pytest tests/test_mcp_acceptance.py -q -s
Remove-Item Env:STT_MCP_TEST_MEDIA
```

## Uninstall

```console
stt-mcp unregister opencode
stt-mcp unregister claude-desktop
uv tool uninstall stt-mcp
```

Transcripts and the Hugging Face model cache remain until explicitly removed.
