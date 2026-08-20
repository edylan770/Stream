# Start here — context for a fresh session

Phases 0–4 are done and Phase 3 is done bar one stage. This file exists so the
next session can pick up without re-reading seven commit messages. Delete it
once you have.

**Read in this order:** [`CLAUDE.md`](CLAUDE.md) (the standing rules),
[`HANDOFF.md`](HANDOFF.md) (state and every deviation, with reasons),
[`spec/GUESSES.md`](spec/GUESSES.md) (every unvalidated number and what would
show it wrong). Then the spec sections you are touching.

---

## Where the build actually is

**Phases 0, 1, 2, 3 and 4 of §15 are complete, and Phase 5 is in progress.
1721 tests pass** (`.venv\Scripts\python.exe -m pytest -q`, 25-30 minutes),
plus 3 that need `--asr`.

One caveat: `scene_events` is built but its OBS log parser has never seen a real
log, and it fails silently when wrong. `clipforge scene-events --check "<log>"`
settles it in one command — see HANDOFF.

A recording goes all the way today:

```
register → run → review → render → hook
```

and comes out a 1080×1920 MP4 with burned-in per-speaker captions, audio
normalised to −14 LUFS, and a hook line you chose.

**Phase 3 was skipped deliberately at first** — it adds signals that change
*which* moments surface, and §8's renderer was worth more — then built out in
commits 31–39: pitch, laughter, derived signals, §4.4's input log, §6.4's gated
negatives, §6.5's two profiles with the combined ranking, §7.2's preview assets
and `scene_events`. **Phase 5 is under way**: commit 40 built §11.6's pull
search, 41 §9.3's chapter segmentation, and 42 the shared §12 validation and
transports in `clipforge/llm/` — the three items testable with no footage.
§9.4's digest and §10's three passes are next. **Nothing has ever called a
model**: the paste round trip is still the transport, and the API source
reports itself unavailable without a key. Phases 6 and 7 are unbuilt.

`git log --format='%s%n%n%b'` is the real archive; every deviation is argued
there. Three worth knowing before touching anything:

- **§6.5's two profiles order the fixture's moments identically** (rho +1.000).
  `gameplay` is 80% inert before Phase 7's vision, and 71% of its live weight is
  markers — so §7.4's combined section is currently a restatement of the
  entertainment ranking. `combined_rank_agreement` in `tool_metrics` is what
  says when that stops being true.
- **§7.2's own preview command costs 13.6 min per 120 candidates.** Measured,
  and replaced with a preset 9.5× faster for 0.007 of SSIM. The synthetic
  fixture understates that cost by 3.4× and would have hidden it.
- **§6.4's negative penalties have never removed a candidate.** Measured, and
  recorded rather than tuned until they did.

---

## THE NEXT THING TO DO IS STILL NOT CODE

This has been true since Phase 1 and is more true now.

**Zero streams exist.** Every weight, threshold, window length, crop
coordinate, filler word and duration cap is a guess. `spec/GUESSES.md` lists
all of them with a falsifier each. §17's tuning procedure needs
`signal_firing_rate_by_rating` from `tool_metrics`, which needs ratings, which
needs footage. §15 names the failure mode directly:

> month three, half the system built, zero videos published, and having become
> a person who builds video tooling rather than a person who makes videos.

**Record one stream. Run it. Review it. Render it. Post something.** Ten of
those will say more about what to build next than any amount of planning.

### The five things a first real stream settles immediately

1. **The crop coordinates.** Every number in `crop_templates.yaml` is
   §8.4's example, and §8.4's example was never measured — its own gameplay
   region distorts 23%. `clipforge render <id> --stills` writes one PNG per
   clip through the identical filter graph; a still costs a second, an encode a
   minute. That is the loop for fitting a template to a real OBS layout.
2. **Whether `obs_anchor`'s WebSocket path works at all.** The hotkey fallback
   is verified end to end; nothing has ever received a real
   `RecordStateChanged`.
3. **Whether `score.peak.target_candidates_per_hour` gives a reviewable
   number.**
4. **Whether §7.1's 4 s per candidate holds.** `clipforge metrics` reports it,
   and the review screen now grades against the configured target rather than a
   hardcoded one.
5. **Whether −14 LUFS sounds right** next to whatever is already in your feed.

### Still untested, and only a real machine can

- `marker_daemon` and `input_logger` against real hotkeys — both tested through
  fakes, neither has seen `pynput`.
- WhisperX on CUDA with `large-v3`. Only `tiny` on CPU has run.
- The FCPXML export actually importing into Resolve.
- Every clip the renderer makes, watched by a person. The tests assert
  geometry, duration, loudness and caption structure; none of them can tell you
  it looks good.

### Missing, and asked for by no phase

**§13.2's backup is built** (commits 29/29a). `clipforge backup` takes one,
`--schedule` prints the nightly `schtasks` line, `--verify` is §13.3's restore
test. **Set the schedule up before your first review session** — the database
already holds 3 ratings and 112 `tool_metrics` rows, so there is already
something in there that cannot be reconstructed.

**§7.3's nudge keys are built** (commit 30), which closes GUESSES gap 1. Press
`[`/`]` to move a window's start and `{`/`}` its end; the trim is what gets
exported and rendered, and every nudge is recorded for §17. **The
instrumentation now exists and wants ~10 real streams** — which is, again, the
thing at the top of this file.

---

## How this build has been run, and why it kept working

Six habits, all of which caught real bugs. Worth continuing.

1. **Plan first, and disagree first.** Every session started by reading the
   relevant spec sections and reporting what was ambiguous, contradictory or
   wrong *before* writing code. Phase 4 alone found: §8.3's caption pseudocode
   leaves the screen blank between every word; §8.3's `both → alternate per
   word by source track` is not implementable from the field it names; §8.4's
   own example distorts by 23%; §8 never says which of four audio tracks
   becomes the clip's audio.
2. **Measure, don't assume.** In Phase 4 this found that no escaping of a
   Windows path survives ffmpeg's filter parser, that `loudnorm` hands the
   encoder 192 kHz, that `linear=true` is never actually granted on this
   material, and that `loudnorm`'s own `input_i` cannot detect a silent window.
   None of those are in any documentation.
3. **When a measurement contradicts the plan, the plan is wrong.** Commit 26's
   premise survived, but only after two measurements that said the opposite —
   both of which turned out to be measuring the wrong thing. Measure the
   artifact the operator gets, not something adjacent to it.
4. **Never assert against hardcoded numbers.** Every numeric test reads
   `manifest.json`, the config object, or the stage registry.
5. **Small commits, each with a full explanation, then stop** so the operator
   can test. The commit message carries the reasoning.
6. **Report honestly.** If a measurement comes out negative, say so. If
   something is untested, name it. Phase 4's commits say plainly which numbers
   are arbitrary.

**Never write to `data/clipforge.db`.** Copy it to a scratch directory and
point `--set paths.db=` at the copy. This rule exists because it was broken
once.

---

## What Phase 4 built, in one paragraph each

- **Captions (§8.3)** — `render/ass.py`, `render/words.py`, `render/timeline.py`.
  Word groups with the active word highlighted, coloured **by `segments.track`
  rather than `segments.speaker`**, which is the only way §8.3's three rules
  work. Two people talking at once stack on two rows.
- **Crop and render (§8.4)** — `render/crop.py`, `render/clips.py`,
  `render/cmd_render.py`. Static templates, refusal on aspect mismatch,
  overlapping regions refused and gaps reported. `--stills` and `--dry-run`
  exist because the coordinates are placeholders.
- **The UI** — a design system under the existing palette, one `#topbar`, the
  §7.3 transcript column (conditional on the stream having segments), Render as
  a job kind, and greyed-and-labelled space for everything §7 asks for that no
  phase has built.
- **Loudness and presets (§8.2/§8.3)** — two-pass, measured; an `ebur128`
  silence gate expressed as a gain limit; the output verified and warned about.
- **Muting and filler (§8.6/§8.2)** — `render/edits.py`, **both toggles off by
  default**, and a default render is byte-identical to the one before them.
- **Hook text (§8.5)** — a paste round trip to a frontier model with §12
  enforced on the way back: real ids so a fabrication is detectable, verbatim
  quotes checked against the transcript, and `llm_invalid_id_rate` recorded.

---

## Opening a fresh session

> Read CLAUDE.md, HANDOFF.md and spec/CLIPFORGE-SPEC.md §&lt;the sections you are
> touching&gt;. Phases 0, 1, 2 and 4 are done; NEXT-SESSION.md has the state.
> &lt;What you want built.&gt; Same as before: tell me anything you'd do differently
> first, test against the fixture manifest rather than hardcoded numbers, small
> commits, stop after each.

### If you are building rather than streaming

In rough order of value:

| What | Why now | Cost |
|---|---|---|
| **`signal_firing_rate_by_rating`** | §14 calls it "the primary weight-tuning input" and §17's whole procedure needs it. The data is being logged per A9; nothing aggregates it. Needs ~10 streams to mean anything. | Half a day |
| ~~**Phase 3 proper**~~ | Done, commits 31–39. Four of its signals still need `input_logger` to have run on a real stream before they mean anything. | — |
| **Phase 5** (digests) | Search (40), chapters (41) and §12's validation and transports (42) are in. §9.4's digest is next; it needs an API key or the paste round trip `hook` already uses — `clipforge/llm/` now serves both. | Days |
| **Phase 6** (trends) | The n-gram half works sooner; clustering needs 60+ streams. | Days |
| **Phase 7** (vision) | §5.9 says build last and cut if hard. | Days |

But the honest answer is at the top of this file: **record a stream.**
