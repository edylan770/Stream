# ClipForge — project state

Read this, then [`spec/CLIPFORGE-SPEC.md`](spec/CLIPFORGE-SPEC.md). The spec is the
requirement; this file records where the build is and which of its instructions have
been deliberately departed from.

**`git log` is the real archive.** Every commit message explains what was built and
why, including the reasoning behind each deviation. `git log --format='%s%n%n%b'` is
worth skimming before changing anything in `score/` or `pipeline/`.

---

## Status

**Phase 1 of §15 only.** Nothing from Phase 2+ is implemented, deliberately.

| # | Commit | What |
|---|---|---|
| 1 | `0bb502a` | skeleton, packaging, `doctor` |
| 2 | `9c55eb3` `dcfacd6` | layered config; §3.2 schema + migrations |
| 3 | `9a892b1` | synthetic fixture with ground truth by construction |
| 4 | `240a72f` | pipeline stage runner (DAG, staleness, atomic writes) |
| 5–6 | `f8454d9` `42cb994` | `register` command; `probe` stage |
| 7 | `b5e0aa8` | `proxy` stage |
| 8–9 | `b51ff82` `922fd9b` | `audio_split`; proxy encoder auto-detection |
| 10 | `4d872f1` | `audio_features` (RMS at 10 Hz) |
| 11 | `dc9b4c7` | `marker_events` + `synth-markers` |
| 12 | `89ca0d4` | scoring engine (§6.2–§6.7) |
| 13 | `2ca5046` | review screen (§7.3 subset) |

**Remaining in Phase 1:**

- **14 — app shell.** Stream list, add-a-recording via a *server-side file browser*
  (browsers cannot expose local paths, so drag-and-drop would mean uploading a 50 GB
  master), running the pipeline from the browser in a background thread with live
  progress. Approved but not started.
- **15 — FCPXML export.** §10.5. Must read operator ratings **across generations**,
  not `is_current`: a re-score that drops a moment you already approved must not drop
  it from the export. The hard part is that FCPXML times must be exact integer
  multiples of a rational `frameDuration` (29.97 is `1001/30000`, not `1/29.97`), or
  Resolve rejects the file or silently rounds.

547 tests pass. `.venv\Scripts\python.exe -m pytest -q`.

---

## Deviations from the spec, and why

These were each agreed explicitly. Do not revert one without reading its reason —
several fix silent-failure bugs where the spec's literal text produces code that
looks correct and is not.

**Data model**

- **Candidates are append-only generations.** §6.1 promises re-scoring is free and
  infinitely repeatable; §3.2 gives candidates no stable identity and cascades ratings
  off them, so a re-score would delete the operator's judgment calls — the one thing
  §13.2 calls irreplaceable. A new generation is created only when `config_version`
  differs; identical config replaces in place. Operator ratings carry forward by time
  overlap, tagged `rating_source='inherited'` so §14's tuning never counts one twice.
- **`config_version` is `<profile>@<sha256[:8]>`** of the canonicalized scoring config,
  not a bare profile name — two different weight sets under one name would otherwise be
  indistinguishable.
- **Artifact paths are stored relative to `data_root`**, in POSIX form, so the database
  survives moving to the streaming PC. `master_path` is absolute; `clipforge db relink`
  repairs it.
- **`record_start_epoch_ms` is nullable**, with `marker_time_base`, so footage with no
  OBS anchor can be registered.
- **`exports` gained `export_items`** — §3.2's single `candidate_id` cannot describe an
  FCPXML timeline referencing forty clips.

**Extraction**

- **Absolute dB is stored, not baselined.** §5.4.1's "delta vs. rolling baseline"
  describes how the signal is *consumed* (§6.2 step 3). Baselining at extract time
  would freeze `rolling_baseline_window_s` — a §17 tunable — into every stream ever
  processed.
- **`events.t` is the raw marker press time**, not `t − 20`. Same reason: the retro
  offset is tunable, so the row records the observation and scoring decides what it
  implies.
- **VFR detection fits PTS against a constant rate.** Measured: a CFR 30 fps MKV on a
  1 ms timebase stores alternating 33/34 ms frames, so comparing PTS deltas flags
  *every OBS recording* as variable-rate; `duration_time` reports a nominal value and
  cannot detect real VFR at all.
- **Standard frame rates are never rewritten.** Snapping picks the *nearest* rate;
  "first within tolerance" silently turned exact 60 fps into 59.94.
- **Track roles: metadata tags → §4.2 position → config override.** Two or three audio
  tracks are **refused**, not guessed: mistaking game audio for the mic poisons every
  signal and §4.2 calls it unrecoverable.

**Scoring**

- **Event kernels are unit-*peak*, not unit-area.** A unit-area Gaussian at σ=10 s
  peaks near 0.004, so §6.5's `marker_definite: 3.0` would contribute ~0.01 against
  z-scores swinging ±3 — the highest-weighted signal, silently ignored.
- **The marker kernel is an asymmetric plateau**, not a point estimate at `t−20`. §4.3
  says the reaction delay is a *range* (5–15 s). Overlapping markers combine with
  `max`, not sum — two presses eight seconds apart describe one moment.
- **Window thresholds are measured from each peak's base**, not from zero. §6.3's
  `exit = 0.35 × v_peak` assumes the composite returns to zero; on a marker plateau the
  pedestal is 3.0 and the threshold lands below it, so expansion never terminates.
  Measured: every window came out clamped at 54 s until this was fixed.
- **Peaks require a minimum value (default 0).** A local maximum in negative territory
  is a local maximum of *quietness*, and `0.35 × v_peak` is above `v_peak` when
  negative.
- **`hysteresis_enter` finally does something** — §6.3 defines it and never reads it.
  It decides whether two nearby peaks are two moments or one bump with a wobble.
- **Rolling variance is computed on centered data**, and σ has a floor
  (`score.zscore_std_floor`). Without the floor, a digitally silent stretch has σ=0 and
  every dither becomes z=50.
- **§6.6's spacing penalty is single-pass** (`accepted_only`): as written it penalises a
  third candidate twice and freezes its iteration order against scores it then mutates.
- **Auto-calibration counts candidates, not raw peaks**, and floors its target — §6.7's
  per-hour rate asks a 60 s fixture for 0.45 candidates.
- **§6.2 step 8 is a documented no-op** in Phase 1: one profile, nothing to merge across.

**Media**

- **`-ss` before `-i` is enforced in code**, not by comment — `ffmpeg.run` refuses a
  command where `-ss` has no `-i` after it (A1).
- **Proxy forces CFR** and adds `-pix_fmt yuv420p` (§5.2 omits it; a 10-bit master
  yields a proxy no browser can decode). Target height is computed in Python so small
  sources are never upscaled.
- **The proxy verifies itself after encoding** — duration, frame count, and *keyframe
  spacing*. A2's fixed GOP is the entire basis for reviewing by seeking the proxy; if a
  future ffmpeg breaks `-sc_threshold`, scrubbing degrades silently without this check.
- **Encoder detection is a test-encode**, not an `-encoders` listing: `h264_nvenc` is
  compiled into most builds and fails at open without an NVIDIA driver. Measured on the
  build machine: libx264 333 s vs h264_qsv 139.5 s for the same file.

**Review UI**

- **No autoplay.** §7.3 assumes §7.2's pre-rendered 2 s clips, which are Phase 3.
  Seek-and-hold on focus; `space` plays the window.
- **Range serving is written out**, not delegated — it is the one behaviour the whole
  review experience rests on.
- **`metrics` reports the median**, not the mean: one candidate left on screen over
  lunch would swamp an average and make §7.1's 4 s target unmeasurable.

---

## Running it

```bash
.venv\Scripts\clipforge.exe doctor                       # environment + chosen encoder
.venv\Scripts\clipforge.exe register --master <file>     # add a recording
.venv\Scripts\clipforge.exe run <stream_id>              # the pipeline
.venv\Scripts\clipforge.exe score <stream_id> --list     # re-score, free and repeatable
.venv\Scripts\clipforge.exe review                       # the UI
.venv\Scripts\clipforge.exe metrics <stream_id>          # is review fast enough?
```

`status`, `signals`, `db`, `config`, `synth-markers` are the rest. Everything takes
`--set key.path=value` to override config for one invocation.

**The fixtures are synthetic test data, not footage.** `fixture_short` and
`fixture_long` are ffmpeg `testsrc2` video with numerically authored audio —
colour bars and static, deliberately. They exist because real footage cannot validate a
detector: nobody can say what the correct mic RMS at t=412 of a real recording is.
Their `manifest.json` carries the ground truth every numeric test asserts against.
Regenerate with `python tests\fixtures\make_fixture.py [--duration 600 --name long]`.

---

## How this build has been run

Worth continuing, because it has caught real bugs:

1. **Plan before code, and disagree first.** Each session started by reading the
   relevant spec sections and reporting what was ambiguous, contradictory, or wrong
   before writing anything. Roughly a dozen genuine spec bugs were found this way.
2. **Measure, don't assume.** The VFR classifier, the encoder detection, the fixture's
   clipping and the pedestal bug were all found by checking rather than reasoning.
3. **Never assert against hardcoded numbers.** Every numeric test reads
   `manifest.json`. A tolerance wide enough to pass a broken fixture is not a tolerance
   — 0.5 dB was hiding a 0.45 dB clipping error until it was tightened to 0.25.
4. **Small commits, each with a full explanation**, and stop after each for the
   operator to test.

---

## Starting a fresh session

Open with something like:

> Read `HANDOFF.md` then `spec/CLIPFORGE-SPEC.md`. We're building Phase 1 only,
> currently through commit 13. Continue with commit 14 (the app shell). Same as
> before: tell me anything you'd do differently first, test against the fixture
> manifest rather than hardcoded numbers, small commits, stop after each.

Nothing else needs carrying over — the deviations above and the commit messages hold
the reasoning.
