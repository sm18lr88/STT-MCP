# Agent Instructions

## Mandatory setup and acceptance guide

Before installing dependencies, changing CUDA/model configuration, registering an MCP client, or
claiming this project works on a machine, read and follow [`docs/SETUP.md`](docs/SETUP.md) from top
to bottom.

That runbook is the authoritative agent procedure for:

- native Windows and Linux bare-metal setup;
- WSL2 setup and the explicitly unsupported macOS/container paths;
- NVIDIA driver, CUDA 12.8, VRAM, FFmpeg, Python 3.13, and `uv` checks;
- downloading the exact pinned Hugging Face model revision, verifying cached checksums, and
  reviewing the remote Python code boundary;
- running static gates, CLI transcription, and the opt-in real FastMCP acceptance test;
- registering, unregistering, and cleaning up MCP clients safely.

Do not replace CUDA with CPU or Apple MPS, remap `CUDA_VISIBLE_DEVICES`, relax the exact model
revision, or claim Linux/WSL2 runtime validation without evidence. STT-MCP currently fails closed
unless exactly one NVIDIA GPU is visible and its conservative 8 GiB globally free startup safety
floor is met. This floor is not a claim that inference intrinsically consumes 8 GiB.

Use `uv` exclusively for Python environments, dependencies, commands, and builds. Keep stdout
clean when running the stdio server because stdout is the MCP protocol wire.

## Completion criteria

An environment is not verified until all applicable commands in `docs/SETUP.md` pass, including:

1. CUDA and FFmpeg probes.
2. Exact Hugging Face revision download and checksum verification.
3. Ruff, BasedPyright, `ty`, Mypy, pytest, and package build.
4. A real local speech file transcribed through
   `tests/test_mcp_acceptance.py` with both SRT and VTT artifacts published.
5. Zero surviving `stt_mcp.worker` processes after the acceptance test exits.
