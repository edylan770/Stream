# ClipForge

Stream → clip → video pipeline. Built to the specification in
[`spec/CLIPFORGE-SPEC.md`](spec/CLIPFORGE-SPEC.md).

**Status: Phase 1 (spec §15) in progress.** Nothing from Phase 2+ is implemented.

See [`HANDOFF.md`](HANDOFF.md) for what is built, what is next, and every place the
implementation deliberately departs from the spec — several of those fix silent-failure
bugs and should not be reverted without reading why.

## What Phase 1 does

Register a recording, probe it, build a scrubbing proxy, split audio, extract mic
RMS, parse markers, score naively (markers + RMS), generate candidate windows,
export FCPXML, and review the candidates in a keyboard-driven UI.

## Setup

This repo is developed on one machine and run on another. Nothing machine-specific
is committed.

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
clipforge doctor
```

`doctor` will tell you what is missing. ffmpeg and ffprobe are required by three
stages; on Windows: `winget install --id Gyan.FFmpeg`.

Machine-local settings — `data_root`, ffmpeg paths, review port — go in
`clipforge/config/local.yaml`, which is gitignored. Copy `local.yaml.example` to
start. Everything else in `clipforge/config/` is the shared tuning surface (§17)
and *is* tracked.

## Layout

See spec §2.3. Directories for Phases 4–7 (`digest/`, `ideate/`, `trends/`) exist
as a visible roadmap and contain no code.
