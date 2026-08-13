# Working rules for ClipForge

Read this, then [`HANDOFF.md`](HANDOFF.md) for state and deviations. The spec is
[`spec/CLIPFORGE-SPEC.md`](spec/CLIPFORGE-SPEC.md); §ref means a section of it.

## How to work here

1. **Plan first, and disagree first.** Read the relevant spec sections and report what is
   ambiguous, contradictory, or wrong *before* writing code. Roughly a dozen genuine spec
   bugs have been found this way; every one was cheaper to find in the plan.
2. **Measure, don't assume.** The VFR classifier, the encoder choice, the fixture's
   clipping, the pedestal bug, the frame-boundary epsilon and the Whisper prompt overflow
   were all found by checking rather than reasoning. If a claim is checkable, check it.
3. **Never assert against hardcoded numbers.** Every numeric test reads `manifest.json`,
   the config object, or the stage registry. A tolerance wide enough to pass a broken
   fixture is not a tolerance.
4. **Small commits, each with a full explanation, then stop** so the operator can test.
   Commit messages carry the reasoning — `git log` is the real archive.
5. **Never write to `data/clipforge.db`.** §13.2 calls it the irreplaceable tier and
   there are no backups. Copy it to a scratch directory and point `--set paths.db=` at
   the copy. This rule exists because it was broken once.
6. **Report honestly.** If a measurement comes out negative, say so rather than tuning
   until it passes. If something is untested, name it.

## Constraints that settle design arguments (§1.2)

- **C1 Deterministic over probabilistic.** Models only where determinism is impossible.
- **C2 Recall over precision.** A false positive costs 3 seconds of review; a false
  negative costs a clip. When rounding or clamping, expand rather than contract.
- **C3 Extraction and scoring are separate.** Extraction is expensive and runs once;
  scoring is pure, cheap and re-runnable over the whole back catalogue.
- **C4 The review UI is the critical path.** 120 candidates in under 8 minutes. If review
  is slow, fix it before adding anything anywhere.
- **C5 Never build ahead of the data.** Zero streams exist. Don't build for a case the
  operator's setup cannot produce.
- **C6 Log everything from day one**, including for features not yet built. Data not
  captured is gone.
- **C7 Idempotent and resumable.** A crash mid-pipeline must never destroy completed work.
- **C8 ≤ ~35 min hands-on per stream.** A feature that increases it must displace one.

## Gotchas that are easy to get silently wrong (Appendix A)

- **A1** `-ss` goes **before** `-i`. `ffmpeg.run` enforces this.
- **A2** Proxies need a fixed GOP (`-g 30 -keyint_min 30 -sc_threshold 0`), verified after
  encoding — it is the whole basis for reviewing by seeking the proxy.
- **A3** ASS colour is **BGR** (`&HBBGGRR&`). Config takes `#RRGGBB` and converts.
- **A4** WhisperX VAD must stay on, or Whisper hallucinates over silence.
- **A5** SQLite in WAL mode; the review UI reads while the pipeline writes.
- **A6** Signal arrays are BLOBs, never row-per-sample.
- **A7** Record MKV, never plain MP4.
- **A8** Epoch **milliseconds** everywhere in capture; convert at ingest only.
- **A9** Feature vectors are logged **in full**, including unweighted signals.
- **A10** Stages write to a temp path and atomically rename.

## Project-specific invariants

- **Config, never constants.** §17 requires every tunable in a config file. See below.
- **dB are logarithms.** Convert to linear power before averaging or ratioing them.
  This has caused two real bugs.
- **Stages defer, they do not fail.** A stage that cannot run here (missing optional
  extra, disabled in config) returns a reason from `available()` so the rest of the
  pipeline still reaches `score`.
- **Ratings are read across scoring generations, never `is_current`.** A re-score must
  never drop a moment the operator approved.
- **The capture layer imports nothing from the rest of `clipforge`** (§2.1).

---

## GUESSWORK DISCIPLINE

Most parameters in this project are unvalidated — no real stream data exists yet. When
you introduce or change one:

1. **Put it in config, never inline.**
2. **Record it in [`spec/GUESSES.md`](spec/GUESSES.md)** with: value, rationale,
   confidence (grounded / plausible / arbitrary), and the observation that would show
   it's wrong.
3. **Prefer deriving the value from the stream** over fixing a constant, where that's
   cheap. Auto-calibration beats a good guess.
4. **If a metric would reveal the guess is wrong and that metric isn't being logged yet,
   add it to `tool_metrics` now.**
5. **When a choice is genuinely arbitrary, say so in the commit message** rather than
   presenting it as reasoned.

Flag anything where you'd want real data before committing to an approach — better said
than buried.
