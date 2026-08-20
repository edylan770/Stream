# Guesses

Every unvalidated parameter in the codebase, with what would show it is wrong.

**Why this file exists.** C5: *"Signal weights, thresholds, and window lengths in this
document are educated guesses, not validated values. They must be tuned against real
footage."* Zero streams exist. Without a written record of which numbers are measured and
which were invented, they all start to look equally authoritative — and the invented ones
are exactly the ones that need revisiting after ten streams.

**Confidence levels**

| | Meaning |
|---|---|
| **grounded** | Measured on real data, or forced by an external constraint (a codec, a model, a format). Changing it needs a reason. |
| **plausible** | Reasoned from domain knowledge, consistent with the spec, but never checked against this operator's footage. |
| **arbitrary** | Something was needed and this was picked. No evidence either way. |

Kept current as part of the change that introduces a parameter — see `CLAUDE.md`.

---

## Scoring — the numbers that decide what you review

These matter most: they determine which moments surface at all. §17 lists most of them
as requiring empirical tuning, and the tuning input is
`signal_firing_rate_by_rating` from `tool_metrics` (§14).

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `score.profiles` | `[entertainment, gameplay]` | see "Profiles and the combined score" below | §6.5's two real profiles as of commit 37. `naive.yaml` — the Phase 1 placeholder with three weights — stays reachable as `--profile naive` and is what every pre-Phase-3 generation was scored with | — |
| `score.rolling_baseline_window_s` | `300` | **plausible** | §17's default. Long enough to span a game's loudness, short enough to follow energy drift over three hours | Long-term drift leaking into scores: candidates clustering in the loudest half-hour of a stream |
| `score.zscore_std_floor` | `0.5` dB | **plausible** | NOT IN §17. A near-silent five minutes has σ≈0, so dither becomes z=50 and the quietest moment tops the stream. Coupled to `extract.rms.db_floor` | Quiet stretches producing candidates; or, too high, real quiet-room moments never scoring |
| `score.smoothing_sigma_s` | `2.0` | **plausible** | §6.2 step 6's value | Peaks landing beside the moment rather than on it, or two adjacent moments merging into one |
| `score.min_peak_value` | `0.0` | **grounded** | NOT IN §17. A local maximum in negative territory is a local maximum of *quietness*, and §6.3's `exit = 0.35 × v_peak` is *above* `v_peak` when negative, so expansion terminates immediately | Nothing — this is arithmetic, not tuning |
| `score.window.min_window_s` / `max_window_s` | `8` / `60` | **plausible** | §17's defaults | How often the operator nudges boundaries during review — **now being collected** (commit 30). `clipforge metrics` reports the direction of each nudge and, crucially, how many nudged windows had come out sitting exactly on one of these two clamps. Repeatedly extending windows that arrived at exactly 60 s means `max_window_s` is too low; repeatedly trimming ones that arrived at exactly 8 s means `min_window_s` is too high. Needs ~10 streams |
| `score.window.hysteresis_enter` | `0.6` | **plausible** | §17 lists it; §6.3 never reads it. Repurposed as the peak-merge threshold: a later peak opens a new window only if the composite dipped below `enter × v_peak` since the last one | Two distinct moments merged into one window, or one moment split in two |
| `score.window.hysteresis_exit` | `0.35` | **plausible** | §17's default | The window-length distribution: everything clamped at `max_window_s` means it is too low |
| `score.window.clamp_mode` | `expand_around_peak` | **grounded** | §6.3 says "clamp" without saying how. This is the only variant that cannot move the peak outside its own window, which the schema forbids | Nothing — the alternatives are invalid |
| `score.window.snap_max_distance_s` | `0.5` | **plausible** | NOT IN §6.3. Without a leash, a window ending in silence snaps to a word twenty seconds away | Windows visibly starting or ending in the wrong place; or clipped syllables surviving, meaning it is too small |
| `score.spacing.window_s` / `factor` | `30` / `0.5` | **plausible** | §17's defaults | Complaints about clustered candidates (too small), or a genuinely dense stretch losing everything after the first (too large) |
| `score.spacing.mode` | `accepted_only` | **grounded** | §6.6 as written penalises a third candidate twice and freezes iteration order against scores it then mutates | Nothing — `spec_literal` exists for comparison and is worse |
| `score.peak.target_candidates_per_hour` | `[27, 50]` | **plausible** | §6.7's "80–150 per 3 h", divided | Review taking longer than §7.1's 8 minutes per 120 (too many), or obviously good moments missing (too few) |
| `score.peak.min_candidates` | `3` | **arbitrary** | A 60 s fixture targets 0.45 candidates, which is useless. Picked so short streams still produce something reviewable | Nothing real — it only binds on test fixtures |
| `score.peak.max_iterations` | `24` | **arbitrary** | Bisection depth. Picked as "enough" | Calibration reporting it could not reach the target range |
| `score.markers.retro_offset_s` | `20.0` | **plausible** | §17 and §4.3's default | §17's own test: the fraction of marker-anchored windows where the moment is actually inside |
| `score.markers.pre_s` / `post_s` / `shoulder_s` | `25` / `5` / `3` | **plausible** | NOT IN §17. §4.3 says the reaction delay is a *range* (5–15 s), so a point estimate at `t−20` misses by ~14 s when the delay was 6 s. The plateau covers the whole span | Marker-anchored windows systematically starting too early or too late |
| `score.markers.combine` | `max` | **plausible** | Two presses eight seconds apart describe *one* moment; summing would give `marker_definite` a contribution of 6.0 and let a flurry dominate the stream | A deliberate double-press meaning "this is really good" failing to score higher — in which case markers want `sum` after all |
| `score.events.default_sigma_s` | `3.0` | **plausible** | Non-marker events (kills, laughs, phrases) are point-in-time and their kernels should be narrower than a marker plateau | Composite peaks landing beside their event |
| `score.events.combine` | `sum` | **plausible** | Three kills in eight seconds genuinely *is* stronger evidence than one — the opposite of markers | A single loud moment with many overlapping events outscoring a genuinely better one |
| `score.rating_inherit_min_overlap` | `0.5` | **arbitrary** | Half a window felt like "the same moment". No evidence | Ratings failing to carry across a re-score (too high), or carrying onto a moment that is not the same one (too low). `render/selection.py` reads across generations precisely because this is unreliable — and since commit 43 the operator's ORIGINAL row always survives on the superseded generation, so a bad threshold now costs a missing *copy* rather than a lost rating |
| `score.combined.alpha` | `0.5` | **plausible** | §17's default. See "Profiles and the combined score" below — it is live as of commit 37 | Whether combined-score winners are actually the best clips (§6.5) |
| `score.score_grid_hz` | `10.0` | **grounded** | DEVIATION from §6.2's 1 Hz. Rolling stats are cumulative-sum cheap, and 1 Hz quantises window edges to a whole second against §7.3's 0.5 s nudges | Nothing — strictly more information at negligible cost |

## Transcript and speech (Phase 2)

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `extract.speaker.dominance_ratio` | `1.5` | **plausible** | §5.8's value, but applied to **linear power** rather than the stored dB — see HANDOFF | Two-person crosstalk attributed to one person (too high), or solo speech labelled `both` (too low). Visible as caption colours in Phase 4 |
| `extract.whisperx.model` | `large-v3` | **grounded** | §5.7 names it | Nothing on a GPU. On CPU it is unusable, which is what `local.yaml` is for |
| `extract.whisperx.language` | `en` | **plausible** | NOT IN §5.7, which sets none and so detects per chunk — and a chunk of near-silence is where Whisper decides the language is Welsh | A stream with genuine non-English speech being mistranscribed |
| `extract.whisperx.vad_method` | `silero` | **plausible** | §5.7's `vad_onset`/`vad_offset` are pyannote's, but whisperx's pyannote path needs a HuggingFace account and licence, contradicting §5.8's own reason for avoiding pyannote | Phantom segments over silence, or speech onsets being clipped — both would argue for pyannote |
| `extract.whisperx.vad_options.vad_onset` / `vad_offset` | `0.5` / `0.363` | **grounded** | §5.7's values, verified to apply to the silero path too | Speech missed at the start of utterances (onset too high) or noise transcribed (too low) |
| `extract.whisperx.vocabulary_mode` | `hotwords` | **grounded** | MEASURED: `both` overflows Whisper's 448-token prompt limit and raises. Also measured better — 2.9% WER against 5.8% for `both` | Nothing; measured on the fixture. Worth re-measuring on real footage with `large-v3` |
| `extract.whisperx.max_vocabulary_terms` | `160` | **arbitrary** | An upper bound before the token budget trims further. Picked as "a plausible amount of jargon" | Terms being trimmed that the transcript then gets wrong — visible in the stage's own log line |
| `extract.whisperx.batch_size` | `16` | **arbitrary** | whisperx's own default | Out-of-memory on the streaming PC's GPU |
| `extract.phrases.speech_rate_window_s` | `3.0` | **grounded** | §5.4.1 states it | Rate too twitchy to peak-find, or so smooth it flattens the pause before a punchline |
| `extract.phrases.swear_window_s` | `10.0` | **grounded** | §5.6 states it | — |
| `extract.phrases.repeat.window_s` / `min_occurrences` | `90.0` / `3` | **grounded** | §5.4.2 states both | A running gag spread over four minutes never firing (window too short) |
| `extract.phrases.repeat.min_words` / `max_words` | `2` / `6` | **plausible** | §11.2 uses 2–6 grams for the cross-stream version of the same job | One-word catchphrases missed (min too high), or noise from long coincidental matches |
| `phrases.yaml: excitement` | §5.6's list verbatim | **plausible** | The spec's own list | The fraction of `phrase_excitement` events landing on rating-2 candidates. A phrase that fires constantly and never on a good moment should go |
| `phrases.yaml: profanity` | ~18 words | **arbitrary** | A general English list. The operator's actual vocabulary is unknown | Swear density flat on a stream that was full of swearing |
| `phrases.yaml: repeat.stopwords` / `tics` | hand-written | **arbitrary** | **This is a stand-in for §11.2's `is_baseline_tic`, which is "computed from the first ~10 streams" and cannot exist yet.** A written list cannot know which phrases *this* operator says constantly | `phrase_repeat` firing on filler. **Phase 6 should replace this with the measurement** |
| `extract.embeddings.model` | `nomic-embed-text` | **grounded** | §5.10's *second* choice. Its first, `bge-small-en-v1.5`, is not in Ollama's library under that name. 768-dim, so ~600 MB per 100 streams against §5.10's 300 MB estimate | Retrieval quality once §11.6's search exists. **Changing it invalidates every stored vector** — two models' spaces are unrelated |
| `extract.embeddings.document_prefix` | `search_document: ` | **grounded** | NOT IN §5.10, which names a model that does not use prefixes. Nomic models are trained with paired task prefixes; the query side is `QUERY_PREFIX` in `extract/embeddings.py` | Degraded retrieval with no error — the failure is silent, which is why the pair is written down in two places |
| `extract.embeddings.normalise` | `true` | **grounded** | §5.10's "~50 ms over 200k vectors" is one matmul, which holds only over unit vectors. MEASURED: Ollama's `/api/embed` already returns unit vectors for this model, so it is usually a no-op — but the legacy endpoint and other models promise nothing | Nothing; it makes an invariant true that a search would otherwise have to assume |
| `extract.embeddings.batch_size` | `32` | **arbitrary** | Enough to amortise the HTTP round trip, small enough not to hold a large request in memory. No evidence for the exact number | Embedding taking a visible share of a run (too small), or Ollama timing out on a batch (too large) |
| `extract.embeddings.timeout_s` | `120.0` | **arbitrary** | Picked as "longer than a batch should ever take" | Timeouts on a slow machine, or a hung Ollama taking two minutes to report |

## Pitch (Phase 3, §5.4.1)

Unusually for this file, most of these are **grounded** — and not because pitch is
better understood than anything else here, but because §5.4.1's own instruction
(`librosa.pyin, fmin=65, fmax=400`) does not survive contact with the 10 Hz signal
grid, so every value below had to be measured to find one that runs at all. The
measurements are on the speech fixture (95 s, both tracks, best of three after a
numba warm-up), extrapolated to 4 hours.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `extract.f0.enabled` | `true` | **grounded** | §5.4.1 defines `mic_f0`/`party_f0`, and §6.5 weights `mic_f0_variance` at 1.8 — the highest continuous weight in the entertainment profile after the markers | — |
| `extract.f0.roles` | `[mic, party]` | **plausible** | §5.4.1 declares a pitch signal for these two and none for `game`, which is right: game audio has no voice. **NO §6.5 PROFILE WEIGHTS `party_f0` or `mic_f0` directly** — both exist to feed `mic_f0_variance` — so dropping `party` halves the stage's cost and loses nothing any current weight reads. On by default per C6 | Wanting the ~8 min/stream back. The counter-argument is that recomputing needs the master, and §13.1 does not promise to keep those forever |
| `extract.f0.fmin_hz` / `fmax_hz` | `65` / `400` | **grounded** | §5.4.1 states both | `f0_rail_fraction` in `tool_metrics` (an INVENTED name). MEASURED that content outside the range is **pinned to the rail, not reported as absent**: a 32 Hz tone comes back as exactly 65.0 Hz on every frame, confidently voiced. So an excited shout above 400 Hz reads as a *flat* 400 Hz — excitement clipped into calm. The stage warns past 25% |
| `extract.f0.analysis_rate_hz` | `8000` | **grounded** | fmax is 400 Hz so 8 kHz is generous. MEASURED it is not only faster: decimating took the mic track's false-positive rate (pitch found over authored silence) from **5.0% to 0.2%**, by removing the high-frequency content that was producing it | Pitch missed on a genuinely high voice, which would argue the decimation filter is too aggressive |
| `extract.f0.analysis_hop_s` | `0.02` | **grounded** | MEASURED across four hops. Not monotonic, which is the point: 10 ms was both slower *and* worse (6.3% false, 25.9% octave-low) than 20 ms, and 50 ms collapsed — the party track's median f0 fell from 222 Hz to **81 Hz** with **41%** of authored-silence frames given a pitch. 20 ms was the best on every accuracy measure and must divide the 100 ms storage period, which it does exactly | A different machine or librosa release moving the optimum. The sweep is cheap to re-run |
| `extract.f0.analysis_frame_s` | `0.128` | **grounded** | librosa's default, and YIN's floor is two periods of fmin (31 ms). MEASURED that 64 ms is not simply coarser: combined with a long hop it produced octave errors, median f0 falling to 83 Hz, at no time saving | — |
| `extract.f0.resolution` | `0.1` | **grounded** | librosa's default. MEASURED that coarsening it is a false economy — it drives the Viterbi's cost directly, but 0.2 took the mic hit rate from 79.1% to 44.5% and 0.5 took it to 12.9% | — |
| `extract.f0.chunk_s` / `overlap_s` | `300` / `2.0` | **plausible** | Forced, not chosen: pyin allocates `2 x 315 x n_frames` float64, which is **3.6 GB** over a 4-hour stream at a 20 ms hop. 300 s holds it to ~75 MB and is what makes the stage heartbeat. MEASURED that chunking at 17 s, 10 s and 7.3 s reproduced the single-pass values **exactly** (0.0000 Hz) wherever both were voiced; the overlap is what buys that | Memory pressure on the streaming PC (lower it), or a boundary artefact appearing in a real pitch track (raise the overlap) |

**Three things measured that are not parameters**, recorded so nobody re-derives
them the hard way:

- **pyin CANNOT be run at §5.4's own 10 Hz hop.** Its HMM transition window is
  `round(35.92 · 12 · hop/sr) · bins_per_semitone + 1` states and must fit inside
  the pitch grid; at a 100 ms hop that is 431 against 315 and librosa raises. Even
  where it does not raise, a window that wide spans the whole 65–400 Hz range, so
  the Viterbi smoothing that distinguishes pyin from plain yin does nothing.
  Pitch is therefore tracked at 20 ms and the median of each five frames stored.
- **Aggregating to 10 Hz improves the signal, it does not just shrink it.** On the
  speech fixture the frame-level hit rate was 79.1% (mic) / 77.7% (party); after
  taking the median of each bucket it was **88.8% / 88.3%**, with false positives
  still 0.4% / 0.0%. A bucket is voiced if any frame in it was, which is C2.
- **~79% of authored-speech frames carrying pitch is the right answer, not a
  shortfall.** Every s, f, t and k is genuinely unvoiced. A test demanding 100%
  would be demanding a wrong number, so the assertions compare speech against
  silence rather than against 1.0.

**The cost, stated plainly:** roughly **16 minutes for a 4-hour stream across both
tracks** on the build machine, recorded per role as `f0_seconds_per_audio_hour`
(INVENTED — §14 has no extraction-cost row that can attribute time *within* a
stage). §1.3 budgets 20–40 minutes for all unattended processing, so this is the
largest single consumer after WhisperX and the proxy. `extract.f0.roles: [mic]`
roughly halves it.

**And the thing that was not a parameter at all.** `mic_f0` is the first signal
in this project with genuine *gaps* — every earlier one was defined at every
instant, because RMS has a value even over silence. Three places assumed that
without saying so, and all three were silent failures rather than errors:

- **A9's weight-0 loading became a poison pill.** `build_tracks` loads every
  stored signal at weight 0 so the feature vector is complete; `0.0 * NaN` is
  NaN, so an unweighted, unscored, archive-only pitch track turned the whole
  composite into NaN and every stream produced **zero candidates**. The symptom
  was 75 unrelated tests reporting "nothing rated 2 or above". `composite_of`
  now skips unweighted tracks and counts a missing sample as contributing zero
  — which is what adds nothing to a sum of weighted z-scores, and is explicitly
  not a claim that the value was average.
- **`json.dumps` writes a bare `NaN`**, which is not valid JSON. The review UI's
  `JSON.parse` rejects the entire payload, so one unvoiced sample would have
  taken the whole review screen down rather than blanking one number. Feature
  vectors and the `?` panel's context now write **null**, which is what
  `feature_schema.yaml` already means by "not computed".
- **A cumulative sum never recovers from a NaN**, so `rolling_zscore` would not
  have degraded *near* a gap — every sample after the first unvoiced consonant,
  for the rest of the stream, would have been NaN. Both it and `resample` take a
  masked path when the input has gaps and the byte-identical original path when
  it does not, so no existing signal's numbers move.

`resample` also **never bridges a gap**: interpolating across four unvoiced
seconds would draw a smooth pitch glide between two words and hand it to
`mic_f0_variance` as prosody.

## Input signals (Phase 3, §4.4 / §5.4.1)

Almost nothing here is a guess, which is unusual for this file: §4.4 fixes the
record shape and the 10 Hz rate, §4.1 fixes the conversion, and §5.4.1 defines
`input_rate` as "keys+clicks per second". What there is instead is a **structural
decision** that matters more than any number.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `contract.INPUT_HZ` | `10` | **grounded** | §4.4 states it, and it is the logger's own aggregation rate | — |
| `input_rate` = keys + clicks | — | **grounded** | §5.4.1 names ONE signal, "keys+clicks per second", not two | Wanting to weight a click differently from a keypress, which would need two signals and a §6.5 row for each |
| bucket reduction | `mean` | **grounded** | Two 10 Hz records inside one 100 ms bucket describe the same tenth of a second twice, not twice as much input. Summing would make the value depend on the logger's rate rather than on the operator | — |

**The structural decision: a gap is not a zero.** The logger can start late,
crash, or be restarted mid-session. Reading an unlogged stretch as "no input
activity" is wrong in the most damaging direction available, because §6.4's AFK
penalty fires on *no input AND no speech* — so a quiet passage of a stream whose
logger was down, or of a stream with no log at all, would be penalised for
something nobody observed. The rates are therefore **NaN** where nothing was
recorded and `input_coverage` is **0**, and `score/derived.py`'s
`input_stillness` will not claim stillness without coverage.

`input_coverage` is itself never NaN: "the logger was not running here" is an
observation about the logger, and it is always known.

**Two things worth knowing that are not parameters:**

- **The log is DAILY, not per-recording.** `input_logger.input_path` writes
  `input-YYYY-MM-DD.jsonl` into a capture directory because §4.5 forbids the
  capture daemons from depending on OBS to learn where it is writing. A day with
  three streams has one file covering all three, so the stage slices it by the
  anchor and drops what falls outside — the rule `marker_events` already follows,
  and for the same reason: clamping would invent activity at second zero of every
  stream that shares a log.
- **`score` does not depend on this stage, deliberately.** A DEFERRED stage never
  enters the runner's `satisfied` set, so anything requiring it is BLOCKED. No
  stream in existence has an input log, so making `score` depend on it would take
  candidate production to zero everywhere. It is the same reason `score.requires`
  has never included `whisperx`.

**Still true, and the reason this stage cannot be validated yet:** §4.4's logger
has never run on a real stream (HANDOFF's "untested, and only a real machine can
test it"). Everything above is exercised against logs written through
`capture/contract.py`'s own record builder, which proves the parsing and the
arithmetic and says nothing about whether `mouse_velocity` discriminates a flick.
§6.5 weights it 1.5 in the gameplay profile on the strength of §4.4's claim that
this is "arguably the second-strongest gameplay signal after markers" — a claim
ten streams would settle.

## Laughter (Phase 3, §5.5)

§5.5 offers this as "cheap, no model, works surprisingly well" and every number
below was measured against a fixture that authors amplitude modulation in and out
of the band. **What that can show is the mechanism; what it cannot show is
laughter.** The fixture proves the detector responds to envelope periodicity
inside 4–7 Hz and not outside it, and not to loudness. Whether real laughter has
that signature is unmeasured, and §5.5's own fallback — "a small pretrained audio
event classifier (YAMNet or similar)" — is the path if it does not.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `extract.laughter.band_hz` | `[4.0, 7.0]` | **grounded** | §5.5 and §17 both state it. Moved out of `deferred:` now that something reads it | Laughter detection precision/recall on real footage — §17's own tuning target for this row |
| `extract.laughter.envelope_hz` | `100` | **grounded** | Forced, not chosen: the stored 10 Hz grid has a Nyquist of 5 Hz, *below the band's own top edge*, so content at 5–7 Hz has already aliased by the time it is sampled and `scipy.signal.butter` refuses to design the filter at all. Must also be a whole multiple of `extract.signal_hz` | — |
| `extract.laughter.frame_s` | `0.02` | **plausible** | 2× the hop, the same overlap rule `extract.rms.frame_s` uses | — |
| `extract.laughter.score_window_s` | `2.0` | **grounded** | MEASURED as the gap between the worst in-band region and the best non-in-band one: 1.0 s/order 4 gave 0.247, 2.0 s/order 6 gave **0.369**. Longer windows mostly help the denominator — unmodulated noise wanders into the band by chance, and averaging pulls that ceiling down | A laugh shorter than the window being smeared into its surroundings |
| `extract.laughter.filter_order` | `6` | **grounded** | MEASURED better than 4 at every window length, and the reason generalises: a Butterworth's cutoffs are its **−3 dB points**, not the edges of a flat passband, so modulation at exactly 7 Hz is half-power attenuated by a filter "at 4–7 Hz". At order 4 the band-edge region scored 0.70 against mid-band's 1.00 | — |
| `extract.laughter.min_depth` | `0.1` | **grounded** | MEASURED, and the fix for a real failure: the share is scale-free, so a noise floor fluctuating in-band scored as high as a real laugh, and the SPEECH fixture produced **ten laugh events** with no laughter in it. Depth is the band component's swing relative to the local level — authored depths 0.3 and 0.8 read 0.21 and 0.55, while noise, out-of-band regions and the floor all read 0.013–0.017 | Shallow real laughter being missed |
| `extract.laughter.floor_db` | `-55.0` | **arbitrary** | Below this the score is a ratio of two noise floors, so it is NaN. An absolute threshold, which `sudden_silence` deliberately avoids — defensible here only because the question is "is there enough signal to form a ratio", not "is this loud for this stream" | A quiet recording scoring NaN throughout |
| `score.derived.laughter.threshold` | `0.60` | **grounded** | The midpoint of the measured gap: worst in-band 0.791 (6.8 Hz, at the band edge), best non-in-band 0.422 (unmodulated noise) | C2 argues for lowering it once real footage exists — a missed laugh costs a clip, a false one three seconds of review |
| `score.derived.laughter.min_duration_s` | `3.0` | **grounded**, and **the cost is real** | THE gate that separates laughter from speech. MEASURED: on the speech fixture the score crosses the threshold 13 times, every run between 0.1 s and 2.2 s and each about 1.7 s after an utterance ENDS — an offset transient, which looks periodic for about one smoothing window. The laughter fixture's 6 s regions produce runs of 4.8–7.5 s. 3.0 sits above every false run and below every true one | **A laugh shorter than three seconds is not detected**, which C2 would normally refuse. Real laughter is what prices that trade |

**The measured result, both directions**: on the laughter fixture all four
in-band regions produce an event and none of the six out-of-band or unmodulated
regions does; on the speech fixture, which authors no laughter, **zero** events
on either track.

**Two things measured that are not parameters:**

- **Speech's syllable rate is inside the band and that turns out not to matter.**
  4–6 Hz syllables sit squarely in 4–7 Hz, but speech spreads its envelope
  fluctuation across the spectrum rather than concentrating it, and the score is
  a *share*: TTS speech measured 0.46 mean against 0.98+ for sustained
  modulation. The scores that do cross are transients, which is what
  `min_duration_s` removes.
- **The score must be detrended LOCALLY.** Subtracting the file's global mean
  made the value at one instant depend on unrelated material elsewhere in the
  recording — identical 5.5 Hz modulation scored 0.77 in a file with other
  content and 0.99 in one without. A rolling mean fixed it. A signal whose value
  depends on what else is in the stream is not a local measurement, and §6.2's
  rolling z-score would then be normalising something already normalised by the
  wrong thing.

## Derived signals (Phase 3, §5.4.1 / §5.4.3)

Computed at score time, not stored — each is a pure function of a stored
primitive **plus a tunable**, and storing it would freeze that tunable into every
stream ever processed (C3). Retuning anything below costs a re-score, which §6.1
promises is free, rather than a re-extraction of the library.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `score.derived.f0_variance_window_s` | `5.0` | **grounded** | §5.4.1 states it | A window too short to span a phrase, or so long it averages a whole paragraph's prosody into one number |
| `score.derived.f0_variance_min_observations` | `8` | **arbitrary** | Below this many voiced samples the answer is NaN rather than zero. A "prosodic range" over two voiced frames is noise wearing the name of a signal, and zero would claim the speaker was monotone | `mic_f0_variance` null on lines that are plainly expressive (too high), or spiking on single words (too low) |
| `score.derived.silence.lookback_s` | `2.0` | **grounded** | §5.4.1's own "within 2 s of high activity" | — |
| `score.derived.silence.activity_z` / `drop_z` / `quiet_z` | `1.0` / `1.5` / `0.5` | **plausible** | See below — `drop_z` is the one that detects | Firing through ordinary pauses between sentences (thresholds too loose), or missing a real stop-dead (too tight). MEASURED on the speech fixture: 9 of 9 authored mic lines produce a firing, 0 of the party-only lines do |
| `score.derived.silence.ramp_s` | `0.5` | **arbitrary** | §6.2 smooths the composite afterwards, so a square wave would be smoothed into something arbitrary anyway; ramping makes the shape intentional | — |
| `score.derived.vad.margin_db` | `6.0` | **plausible** | Above the track's own rolling baseline, so it encodes no microphone's gain. MEASURED against the fixture's authored utterances: recall 100% on both tracks, precision 84.5% (mic) and 89.1% (party) | Speech missed on a quiet speaker (too high), or room tone counted as speech (too low) |
| `score.derived.vad.min_speech_s` | `0.3` | **arbitrary** | A shorter run is a click, a keyboard, a cough | Short interjections ("what?") being dropped |
| `score.derived.vad.hangover_s` | `0.4` | **plausible** | The gaps between words in one sentence are below any energy threshold; without this, one sentence becomes nine "utterances" none of which survives `min_speech_s`. MEASURED: the predicted overlap runs exactly this far past the authored one and no further | Two separate remarks merged into one utterance (too long) |
| `score.derived.stillness.*` | `3.0` / `3.0` / `0.5` / `0.5` | **arbitrary** | Inert until the `input_signals` stage exists; nothing has ever produced an input rate to calibrate against | Everything. These are the least evidenced numbers in the file |
| `score.derived.reaction.window_s` | `2.0` | **grounded** | §5.4.3's own "within 2 s" | — |
| `score.derived.reaction.silence_level` / `party_spike_z` | `0.5` / `1.0` | **arbitrary** | How far `sudden_silence` must have ramped, and how loud the party must be | Nothing yet — **neither fixture authors §5.4.3's shape**, so this composite is tested synthetically only |
| `score.zscore_std_floor_by_signal` | `0.05` × 3, `0.25` | **arbitrary** | `score.zscore_std_floor` is 0.5 and in DECIBELS. For a 0..1 gate firing under 25% of the time sigma is below 0.5, so the dB floor binds and z lands at exactly ~2.0 however rare the firing — the dB number rescuing a signal it knows nothing about. 0.05 lets a gate firing 1% of the time reach z ≈ 4 and one firing 20% reach z ≈ 2, which is the ordering C2 wants | A derived signal dominating the composite (floor too low), or its firings all scoring alike (too high) |

**Four things measured that are not parameters:**

- **`sudden_silence` cannot be defined as "below a threshold".** The first cut
  tested `z <= quiet_z` with a negative threshold and fired **zero times on every
  input**, because on any real stream *silence is the baseline*: the operator is
  not talking most of the time, so the rolling mean sits near the quiet level and
  z during silence is about zero, never meaningfully negative. The **drop** from
  the recent peak is what detects; `quiet_z` is only a confirmation.
- **`reaction_onset` requires the overlap to BEGIN after the silence.**
  `sudden_silence` (the mic went quiet) and `overlap_speech` (both mics live) are
  nearly contradictory, so the only way both hold at one instant is inside
  `vad.hangover_s`. MEASURED: a concurrent test fired on the speech fixture at
  54.6 s and on the noise fixture at 27.3 s — real moments, found through a
  constant chosen to stop one sentence being chopped into nine. §5.4.3's word is
  "followed", and a rising edge is what that means.
- **Neither fixture authors §5.4.3's shape.** The noise fixture's overlap starts
  *before* its silence; the speech fixture's overlap starts 0.6 s after a silence
  that has already run nineteen seconds, by which time `sudden_silence` has
  decayed. So `reaction_onset` has synthetic tests only, and the real check is a
  hand-labelled clip.
- **`overlap_speech` needs no model.** Recall was 100% against the manifest's own
  `overlap_windows`, and the entire imprecision is the hangover overrunning the
  authored end by 0.34 s. Zero firings across the 19.4 s of authored silence.

## Profiles and the combined score (Phase 3, §6.5 / §7.4)

Both weight sets are §6.5's own numbers, transcribed verbatim. That is the only
defensible choice today — changing one before a single stream has been reviewed
would replace the spec's guess with a worse-evidenced one — but it means every
weight in the detector is **plausible** at best and none has been tuned.

**How much of each profile has a producer**, which is what decides whether the
combined score can be what §6.5 says it is:

| Profile | total | live today | Phase 7 vision | needs §4.4's log | needs WhisperX |
|---|---|---|---|---|---|
| `entertainment` | 20.7 | **16.8 (81%)** | 0.4 (2%) | 0.2 (1%) | 3.3 (16%) |
| `gameplay` | 21.0 | **4.2 (20%)** | 14.1 (67%) | 2.7 (13%) | 0 |

`gameplay`'s four live signals are `marker_definite` 3.0, `mic_rms` 0.8 and the
two laughs at 0.2 — **71% of its live weight is markers**, and all four are also
in `entertainment`. So before Phase 7 it is a marker detector, and §6.5's
"intersection of mechanics and personality" is an intersection with markers.

**MEASURED on `fixture_long`, and it is exactly what that predicts:**

    combined vs entertainment: rho +1.000, top-20 overlap 100%
    moments: 1 by entertainment, 5 by entertainment+gameplay, 2 by gameplay

The two profiles order the eight moments **identically**. They do not agree
about which moments exist — three of the eight were found by one profile alone,
which is §6.2 step 8's merge earning its place — and they do not produce the
same scores (9.79 against 3.61 at the top, 3.58 against 3.84 at the bottom, so
`gameplay` ranks the last three *above* where `entertainment` puts them). But
the ordering is the same, so the section §7.4 puts first is currently a
restatement of the section under it.

Both numbers are written to `tool_metrics` on every run as
`combined_rank_agreement` (an **INVENTED** name — §14 has no such row), with the
`found_by` counts in its `meta`. Rho falling below 1.0 as vision lands is the
observation that says the combined section has started earning its place.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `score.profiles` | `[entertainment, gameplay]` | **grounded** | §6.5: "Marvel Rivals streams: run **both** profiles plus combined." Its other two cases are one line each — `[entertainment]` for casual/comedy, `--profile naive` for a Phase 1 comparison. The removed singular `score.profile` is refused **by name** rather than by the `config_schema` bump, which cannot see it: that guard compares the merged value, and a `local.yaml` inherits it from the packaged defaults | Streaming something other than Marvel Rivals, which §6.5 already answers |
| `entertainment` weights | §6.5 verbatim | **plausible** | The spec's own numbers for its own primary profile. A test compares the YAML against a table transcribed from §6.5, so a typo in either is a failure rather than a quietly different detector | `signal_firing_rate_by_rating` from `tool_metrics` (§14, §17): a signal firing equally on rating-0 and rating-2 candidates is not discriminating and its weight is wrong |
| `gameplay` weights | §6.5 verbatim | **plausible**, and 80% inert | As above. Deliberately NOT rebalanced for the signals that happen to exist today: compensating now would bake this phase's gaps into the profile and have to be undone when vision lands | Same, plus `combined_rank_agreement` — while rho is 1.0 this profile is not adding an ordering |
| `score.combined.alpha` | `0.5` | **plausible** | §17's default, and even is the neutral reading of §6.5's "both must be non-trivial". Above 0.5 leans the geometric term on the primary profile | §17's own test: "whether combined winners are actually the best clips". Needs footage |
| `normalise` = share of the per-stream max | — | **grounded** | NOT IN §6.5, which leaves `normalize` undefined. Min-max is the obvious alternative and is wrong in a way that matters: it sends whichever candidate is a profile's *weakest* to exactly 0.0, and §6.5's product then returns 0 for it whatever the other profile said — so on a stream where `gameplay` is a marker detector, the funniest unmarked moment would rank last in the section §7.4 puts first | Nothing measurable yet; it is a definition. **Its consequence is that combined ranks WITHIN a stream and is not comparable across streams** |
| `review.sections.combined_top_n` | `20` | **arbitrary** | §7.4 says "combined-score winners (both profiles high)" and gives no size. §7.1 reviews ~120 per session and §6.7 targets 80–150 per 3 h, so 20 is about a sixth — a first screen the operator can finish before forming an opinion of the section | Reaching the end of the section and wanting more of it, or losing interest before the end of it. `combined_rank_agreement` says whether the section is distinct from the list below it at all |
| §7.4 section 4 = every marker-anchored candidate outside section 1 | — | **plausible** | §7.4's "marker-anchored candidates that did not rank highly (safety net)" does not define "highly". Once the combined cut is the only thing above it, missing that cut is what the phrase can mean. A key the operator pressed is a different kind of evidence from a weighted sum and should not be buried in the tail of a list sorted by weights | Marker-anchored candidates the operator would rather have seen in the entertainment ranking, i.e. the safety net becoming the place good moments go to be ignored. **Watch for sections 2 and 3 coming out empty**, which is what happens when nearly everything is marker-anchored — see below |

**MEASURED, and the thing to watch on the first real stream:** on `fixture_long`
every one of the 8 candidates is marker-anchored — 7 have a press inside their
own window and all 8 have a marker contribution — so §7.4's sections 2 and 3 come
out **empty** and the rail reads "combined winners, then everything else". That
is the fixture rather than the rule: it presses a marker roughly every 20
seconds (30 over 600 s), where a real stream is 10–30 presses over three hours
against 80–150 candidates. If sections 2 and 3 are still empty on real footage,
`marker_anchored` is too loose for §7.4's purpose — it is true for any window
overlapping §4.3's ±25 s marker plateau, not only for a moment the operator
actually marked — and tightening it is the fix. The definition is deliberately
unchanged here: it also drives §7.3's `m` filter, and changing both on the
strength of a saturated fixture would be tuning against the wrong thing.

**Three things that are not parameters:**

- **A merged window is re-clamped to `max_window_s`.** Each profile clamps its
  own windows per §6.3, but two that overlap by a second and extend in opposite
  directions union to almost twice that, and §6.2 step 8 does not exempt the
  result from §6.3's range. Clamped around the winning peak, which is the only
  variant §3.2's `CHECK (t_peak BETWEEN t_start AND t_end)` permits. **It never
  fired on the fixture** — the longest merged window came out at 54.8 s against
  a 60 s bound — so it has synthetic tests only.
- **§6.6's spacing runs AFTER the merge**, where §6.2 step 6 puts it inside the
  per-profile loop. Spacing exists to stop "ten candidates from one 90-second
  stretch" in the list the operator reviews, and after step 8 that is one merged
  list; penalising per profile first would let two profiles each contribute
  their own suppressed-but-kept candidate into the same thirty seconds. The
  factor multiplies all three scores, so §7.4's three rankings cannot disagree
  about which moments were suppressed.
- **`contributing_signals` explains the PRIMARY profile only.** It is weight ×
  value, so a two-profile row has two breakdowns; storing both would change a
  payload shape `review/queries.py`, `clipforge score --list` and the review
  UI's `?` panel all parse. All three scores go into its context instead, so the
  panel explains the number in the score box. A per-profile breakdown is what to
  build when the gameplay ranking becomes worth explaining, which is Phase 7.

## Negative signals (Phase 3, §6.4)

§6.4 is the most prescriptive section in the spec — "this is a specific
requirement and must be implemented exactly as described" — and it supplies two
of the seven numbers below. The other five had to be invented, because §6.4
describes a *shape* (gate, wait, ramp, remove) and never a magnitude.

**The honest headline is that neither penalty has ever removed a candidate.**
MEASURED on `fixture_long` with a synthesised §4.4 input log, idle across every
gap in the fixture's own speech:

| `afk_threshold_s` | share of stream held down | calibrated prominence | candidates | without §6.4 | peaks suppressed |
|---|---|---|---|---|---|
| 60 (default) | 0% — never fires | 1.918 | 7 | — | — |
| 15 | 13.0% | 2.651 | 8 | 7 | **0** |
| 10 | 21.3% | 3.631 | 8 | 7 | **0** |

Two things follow, and both are recorded rather than tidied away. The penalty
lands where nothing was scoring anyway — a stretch with no input and no speech
has a composite at or below zero already, and `score.min_peak_value` discards a
local maximum in negative territory before any penalty is applied. And digging a
trough *raises* the calibrated prominence by 38% and 89%, because
`scipy.signal.find_peaks` measures a peak above the higher of its two flanking
cols: a hole between two peaks makes both more prominent, and §6.7's
auto-calibration then bisects the threshold upward to compensate. Both candidate
counts sit inside §6.7's own target band, so the 7-vs-8 difference is where
bisection landed and not a moment found or lost.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `score.negatives.menu_grace_period_s` | `8` | **plausible** | §17 and §6.4 both state it. Moved out of `deferred:` — `score/gates.py` reads it now | §17's own test: "whether lobby jokes ever get suppressed". Inert until Phase 7 produces `menu_screen` |
| `score.negatives.afk_threshold_s` | `60` | **plausible** | §17 and §6.4 both state it. Moved out of `deferred:` | §17's own test: "false AFK penalties". `negative_penalty_share` in `tool_metrics` is the number — a share climbing above a few percent on a stream the operator was present for means it is too low |
| `score.negatives.ramp_s` | `4.0` | **arbitrary** | NOT IN §6.4, which says "ramping in gradually (not a step function)" and names no duration; §17 has no row for it. One thing about it is not arbitrary: §6.2 smooths the composite with `smoothing_sigma_s` (2.0) *after* the penalty is applied, so a ramp much shorter than that sigma is smoothed into something indistinguishable from a step and the requirement becomes decorative. A test asserts `ramp_s > smoothing_sigma_s` | A penalty edge visibly truncating a window that should have run through it (too long), or the ramp being invisible in the smoothed composite (too short) |
| `score.negatives.menu_penalty` | `2.0` | **arbitrary** | §6.4 says "apply penalty" and never says how much. The unit is the one every §6.5 weight uses — roughly one standard deviation of notability per 1.0 — so this is "about as much as two ordinary signals firing at once". **Subtracted, never multiplied**: the composite is signed, and scaling a negative composite *raises* it | A moment the operator wanted being suppressed (too high), or a menu stretch still producing candidates (too low). Nothing can price it until Phase 7 exists |
| `score.negatives.afk_penalty` | `2.0` | **arbitrary** | As `menu_penalty`, and deliberately a separate key so the two can diverge | Same, and MEASURED so far to remove nothing at all — see the table above |
| `score.negatives.afk_max_input_rate` | `0.0` | **plausible** | §6.4 says "no input activity" and never defines it; this is the literal reading. Configurable because a single stray keypress in sixty seconds currently restarts the timer, and ten real streams may well show that somebody sitting still still twitches | The AFK penalty never firing on a stream where the operator plainly was away |
| `score.negatives.afk_max_mouse_velocity_px_s` | `0.0` | **plausible** | NOT IN §6.4, which names one condition where §4.4's logger writes two fields. A mouse moving with no clicks is input activity, so both must be idle — which makes the penalty *harder* to fire, the direction C2 argues for: a false AFK penalty costs a clip, a missed one costs three seconds of review | Same as above. Raising it is what to try first if AFK never fires |

**Four things that are not parameters:**

- **`menu_screen` is a STATE and `feature_schema.yaml` declares an EVENT.** The
  reconciliation is `events.t_end`, which has said "NULL for instantaneous"
  since `0001_init.sql` — so §5.9's vision writes one row per menu span and
  nothing has to move. **It must set `t_end`**: a row without one covers a
  single sample and can never satisfy an 8-second grace period. Until then the
  gate is built, tested, and inert, and the run log says so on every score
  rather than showing a column of zeros that reads as "never active".
- **The penalties are not z-scored, unlike every other continuous track.** A
  rolling z-score of a mostly-zero array does two wrong things at once: it makes
  the rare firing enormous, and it makes every *unpenalised* sample slightly
  positive — a negative signal quietly rewarding the rest of the stream. They
  enter on the unit-peak scale the event kernels use.
  `score.zscore_std_floor_by_signal` therefore does not apply to them.
- **§6.4's `audio_energy < rolling_baseline` is implemented as `<=`.** A stretch
  pinned at `extract.rms.db_floor` equals its own rolling mean whenever the
  whole baseline window is pinned too, so the strict reading fails on digital
  silence — the most unambiguous "nothing is happening" audio there is. The same
  degenerate case `score.zscore_std_floor` exists for, one condition over.
- **"ANY speech" is `derived.speech_gate`, verbatim.** §6.4's reset condition and
  §5.4.1's "VAD on both tracks" are the same question asked twice; a second
  energy test here could have disagreed with the first with nothing to report
  it. A test asserts the two agree sample for sample.

## Extraction and media

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `extract.signal_hz` | `10.0` | **grounded** | §5.4 states it | — |
| `extract.rms.frame_s` | `0.2` | **plausible** | 2× the hop, so each sample overlaps its neighbours by half. §5.4.1 implies librosa's 2048-sample default (128 ms) | Transients smeared across frames (too long) or RMS too noisy to z-score (too short) |
| `extract.rms.db_floor` | `-80.0` | **grounded** | Digital silence is `20·log10(0) = -inf`, which poisons every downstream mean. Coupled to `score.zscore_std_floor` | — |
| `ingest.audio.sample_rate_hz` | `16000` | **grounded** | §5.3 states it, and it is what WhisperX wants | — |
| `ingest.proxy.height` / `video_bitrate` | `720` / `2M` | **grounded** | §5.2 states both | Scrubbing feeling slow, or proxies too large for §13.1's budget |
| `ingest.proxy.gop` / `keyint_min` / `sc_threshold` | `30` / `30` / `0` | **grounded** | A2 and §5.2. Verified after every encode | — |
| `ingest.proxy.force_cfr` | `true` | **grounded** | NOT IN §5.2, which claims the proxy shares the master's timebase — false for a VFR master. Forcing CFR makes the claim true | — |
| `ingest.proxy.pix_fmt` | `yuv420p` | **grounded** | NOT IN §5.2. A 10-bit or 4:2:2 master yields a proxy no browser can decode, and the review UI shows a black player with no error | — |
| `ingest.proxy.verify.max_duration_delta_s` | `0.5` | **plausible** | Every timestamp assumes proxy and master are interchangeable | A legitimate encode failing verification |
| `ingest.proxy.verify.max_keyframe_interval_ratio` | `1.5` | **arbitrary** | Allows some slack around the GOP without accepting scene-cut keyframes | An encoder that legitimately varies spacing failing |
| `ingest.probe.rate_sample_seconds` | `4.0` | **plausible** | Enough PTS to fit a constant rate and check the residual; two windows are sampled | A recording that starts CFR and degrades later being classified constant |
| `ingest.audio.verify.silence_peak_dbfs` | `-80.0` | **plausible** | Below this counts as digital silence. Warns, never fails — a silent `party.wav` just means nobody was in Discord | A quiet-but-real track being flagged |

## Chapter segmentation (Phase 5, §9.3)

§9.3 lists four boundary inputs. **Three of them produce nothing on a stream
processed with shipped defaults**, so this is a long-silence splitter until at
least transcription is turned on — and `clipforge chapters` says which input
produced each boundary and why the others did not, rather than letting four
names in the spec imply four sources agreed.

| §9.3 input | Producer | State today |
|---|---|---|
| Long silence > 60 s | `mic_rms`/`party_rms` through §6.4's gate | **live on every stream** |
| Transcript embedding shift | `segment_embeddings` | needs `extract.whisperx.enabled`, off by default |
| Scene changes | `scene_events` (commit 39) | producer exists, **parser unvalidated** — yields nothing yet |
| Game changes | — | **no producer anywhere.** `streams.games` is untimed; per-moment game ID is §5.9, Phase 7 |

**§9.3's own embedding formulation is the weaker of two, MEASURED.** It says
"cosine distance between consecutive rolling-window embeddings". Against a
deliberately maximal topic change — five Marvel Rivals lines followed by five
baking lines, real model:

| formulation | window 2 | window 3 | window 4 | peak / mean |
|---|---|---|---|---|
| §9.3's consecutive windows | **misses the seam** | hits | — | 1.2–1.3× |
| before/after centroids at each gap | hits | hits | hits | 1.2–1.3× |

The before/after form localises the seam at every window size. **But the ratio is
the more important number**: even for two topics that unrelated, the boundary is
only 1.2–1.3× the mean distance. `nomic-embed-text` compresses similarity so hard
that everything is roughly half-similar to everything — the same property that
kept a `min_similarity` knob out of `search:`. On the speech fixture, thirteen
lines of one conversation with **no topic change at all**, consecutive distances
still run 0.000–0.466 (mean 0.292).

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `digest.chapters.min_silence_s` | `60` | **grounded** | §9.3 states it | Chapters breaking at ordinary pauses (too low), or a three-hour stream coming out as one chapter (too high) |
| `digest.chapters.merge_within_s` | `120` | **grounded** | §9.3 states it | Two genuinely different topics 90 s apart being merged into one chapter |
| `digest.chapters.target_min_s` / `target_max_s` | `600` / `1800` | **grounded** | §9.3's "10–30 minutes". **Guidance, not a constraint** — see below | Chapters routinely outside the range on real streams, which would mean the inputs are too sparse rather than the targets wrong |
| `digest.chapters.embedding.window` | `3` | **arbitrary** | Segments compared either side of a gap. MEASURED that 2, 3 and 4 all give the same 1.2–1.3× ratio on the seam probe, so nothing in that range is clearly better | A window shorter than a topic (too small), or one spanning two topics and smearing the boundary (too large) |
| `digest.chapters.embedding.prominence_sd` | `1.0` | **arbitrary** | Peak prominence in the distance array's **own** standard deviations. RELATIVE on purpose — see below | Boundaries on every conversational wobble (too low) or none at all (too high) |
| `digest.chapters.scene_changes` | `true` | **plausible** | On costs nothing: scene changes never propose a boundary alone, they only corroborate | — |

**Four things that are not parameters:**

- **Prominence is relative because an absolute threshold cannot mean one thing.**
  MEASURED: the distance scale moves with the window — max 0.466 at window 1
  against 0.138 at window 2 — so `min_distance: 0.05` would be strict at one
  setting and meaningless at another. Prominence is therefore a multiplier of the
  array's own standard deviation, and the absolute value handed to
  `score/windows.py`'s peak finder is computed per stream.
- **The merge defers to the most trustworthy source, not the earliest — and the
  fixture is what forced that.** The first cut took the earliest boundary in a
  cluster, on the reasoning that starting a chapter at the first evidence loses
  no content. On the speech fixture that let a **1.21-sd embedding bump at 29.4 s
  displace a nineteen-second silence at 52.0 s**, putting the boundary 23 seconds
  early, mid-sentence. Priority is `silence > embedding > scene`: silence is the
  only input validated on real data and its timestamp means something exact
  (speech demonstrably resumed), the embedding peak is a weak statistic localised
  only to within a window of segments, and §9.3 itself calls scene changes a
  tie-breaker.
- **A silence boundary is the END of the gap, not its middle.** The dead air
  belongs to the chapter that just finished. Splitting it would put half of a
  two-minute pause at the head of the next chapter, where §9.4 hands it to a
  model as context.
- **Chapters tile the stream, and it is asserted rather than assumed.** No gaps,
  no overlaps, first starts at 0, last ends at `duration_s`. §9.2's structure is a
  partition and §9.4 chunks over it, so a hole would silently drop that transcript
  from the digest — no error, just a stream the model was never shown part of.
  §9.3's target range is guidance around that: an over-long chapter splits only at
  a boundary that was genuinely found inside it, never at an invented midpoint,
  and when there is none the shortfall is reported.

**What no test here can show:** that these are good chapters. The speech fixture
is one continuous conversation and `fixture_long` is band-limited noise. The
mechanism is tested; the judgement needs a real transcript.

## Semantic search (Phase 5, §11.6)

Unusually little here is a guess, because the two decisions that mattered were
both settled by measurement rather than reasoning — and both came out against
the obvious choice.

**§5.10's performance claim is verified, and its implied implementation is
wrong.** MEASURED at 768 dims over 200k vectors:

| | time | peak memory |
|---|---|---|
| the matmul alone — §5.10's "~50 ms" | **47 ms** | — |
| load every row, then one matmul | 1004 ms | 614 MB |
| **stream in 4096-row chunks, running top-K** | **716 ms** | **12.6 MB** |

So "brute-force cosine over 200k vectors in numpy is ~50 ms; no vector database
is required" is true about the *matmul* and silent about the *read*, which
dominates it 20:1. Streaming is faster **and** 49× lighter, which is not a
trade-off. None of it matters at today's scale; it is built this way because the
shape is cheaper to get right now than to retrofit.

**Retrieval works, on thirteen lines.** Hand-written queries sharing no content
words with their targets: 4 of 5 put the target at rank 1, the fifth at rank 3.
That says the *mechanism* does what §11.6 claims — "the operator remembers the
vibe, not the words" — and says nothing about whether retrieval is good on a real
transcript, which needs footage.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `search.limit` | `20` | **arbitrary** | §11.6 gives no number. Low deliberately: this is "I remember a moment, find it", not a browse list — a wanted result at rank 30 means the query was wrong, not the limit | Routinely paging past the end looking for something that is there |
| `search.chunk_size` | `4096` | **grounded** | The table above. Also beat 16384 and 65536 on both time and memory | Memory pressure on a much larger corpus (lower it); or a machine where fewer, larger reads win (raise it) |
| `search.snippet_chars` | `240` | **arbitrary** | Enough to recognise a line without wrapping the result list | Results routinely elided mid-sentence so you cannot tell them apart |
| `_RANK_DECIMALS` | `5` | **grounded**, and NOT in config | See below — it exists so `chunk_size` cannot change what a query returns | Two segments genuinely 1e-5 apart in meaning, which the 0.345–0.610 band makes impossible |
| **no** `search.min_similarity` | — | **grounded** | MEASURED over 65 query–document pairs: scores span **0.345–0.610** (mean 0.477, sd 0.060). A threshold anywhere in that band keeps roughly half of *everything* while reading like a relevance filter. These scores support rank and nothing else | A model whose scores actually separate related from unrelated. Re-measure before adding one |

**Three things that are not parameters:**

- **Chunk size was silently changing the results, and that was a real bug.**
  MEASURED: the three identical `"Let's go, Hawkeye."` lines in the speech
  fixture scored `0.462986` at chunk=1 and `0.462985` at chunk=2. Not a
  tie-break problem — `matrix @ vector` dispatches to a different BLAS path
  depending on how many rows it is given, and float32 accumulation over 768
  terms is not associative. So a config value with no business affecting output
  was reordering results. Ranking is therefore computed on scores **quantised to
  5 decimals**, with `segment_id` breaking what remains: 1e-5 sits far above the
  observed ~1e-6 of FP noise and far below anything meaningful. The *unrounded*
  score is still what gets returned and displayed.
- **A query must use `QUERY_PREFIX`, and a search must filter by `model`.** Both
  fail silently otherwise — the first degrades retrieval with no error, the
  second returns a finite, ordered, meaningless ranking across two unrelated
  geometries. A test asserts `search.py` imports the prefix rather than
  restating it.
- **Search returns SEGMENTS, and `candidate_id` is nullable by design.** A
  memorable line is frequently nowhere near a detected peak, so the UI says "no
  candidate covers this" rather than focusing an unrelated moment and presenting
  it as the match.

## LLM transport and §12 validation (Phase 5, §12)

Everything about the API path here is unvalidated in the way that matters most:
**no request has ever been sent.** MEASURED on this machine — the `anthropic`
package is not installed and `ANTHROPIC_API_KEY` is unset — so
`llm.source: anthropic` reports itself unavailable and names which of the two
is missing. What the suite proves is that §12's checks refuse bad replies and
that the request is assembled from config; it proves nothing about whether the
API accepts that request.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `llm.source` | `manual` | **grounded** | The paste round trip needs a person and a browser, not a credential, so it is always available. It is the default because it works, not as a fallback | — |
| `llm.prompts.file` | `prompts.yaml` | **grounded** | Same arrangement as `phrases.yaml` and `crop_templates.yaml`: named from config, resolved against the config directory | — |
| `llm.anthropic.model` | `claude-opus-5` | **plausible** | §12.4 budgets a whole stream at cents and a digest is one call, so the capable model is affordable. Never run | A real digest costing more than §12.4's estimate, or a smaller model reading as well |
| `llm.anthropic.effort` | `high` | **arbitrary** | Depth control in place of a token budget. There is no basis for choosing between `medium` and `high` without a reply to judge | A `medium` digest reading as well for fewer tokens |
| `llm.anthropic.max_tokens` | `16000` | **arbitrary** | Enough for a chapter map on a stream that does not exist | A reply that stops with `stop_reason: max_tokens` |
| `llm.anthropic.fallbacks` | `default` | **plausible** | A safety classifier can decline a request and return nothing at all; a stopped digest is worse than one answered by a slightly smaller model | The beta being rejected — one config edit (`fallbacks: ""`) to disable |
| `anthropic>=0.116` in `pyproject.toml` | floor | **arbitrary** | The release the request shape was written against. Nothing here has installed it | A 400 on the first real call |

**Deliberately NOT config: the key itself.** `llm.anthropic.api_key_env` names
the environment variable to read and never holds a value. This repository is a
git repository and a `local.yaml` is one `git add -A` away from being in it.

**Not a guess, and the one thing about the request that could be checked
blind:** it carries no `temperature`, `top_p`, `top_k` or `budget_tokens`.
Those are a documented rejection on the configured model rather than a matter
of taste, so a test asserts their absence rather than trusting the author.

**Worth watching once real replies exist:** `llm_invalid_id_rate` in
`tool_metrics` — see the hook-text section, which already writes it. It is now
computed by shared code, so a digest and a hook reply produce a comparable
number rather than two metrics that happen to share a name.

## Scene events (Phase 3, §5.1 stage 11 / §4.2)

**Everything in this section is unvalidated in a way nothing else in this file
is.** Every other guess here is a *number* that might be badly chosen. These are
*regexes that may not match at all*, and a regex that does not match produces
zero events silently — indistinguishable from a session in which nobody switched
scenes. No OBS log has ever been seen by this code; there is no OBS install on
the build machine.

**The correction path, in full:**

```bash
clipforge scene-events --check "<path to any OBS log>"
```

It prints which patterns fired, the share of lines carrying a parseable
timestamp, the recording spans found, the scene timeline, and every line
mentioning a scene or a recording that no pattern claimed. Run it on the
streaming PC and send back the report rather than the log — OBS logs carry
machine paths, hardware details and sometimes stream URLs. Then fix the patterns
in `extract.scene_events.patterns` (the only place in the codebase with a regex
for OBS's text), drop the log into `tests/fixtures/obs_logs/` — that directory is
parametrized, so it is covered with no test changes — and move these rows to
**grounded**.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `extract.scene_events.enabled` | `true` | **plausible** | C6. Costs nothing when no log is attached: the stage defers | — |
| `patterns.elapsed` | `^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}\.\d+):\s` | **arbitrary** | Believed to be OBS's `HH:MM:SS.mmm: ` prefix — elapsed since OBS launched, **not** a wall clock. THE load-bearing pattern: if it is wrong nothing else can be trusted, which is why `--check` reports it first and separately | The share of timestamped lines in `--check`. Under 50% and this is the only thing worth looking at |
| `patterns.scene_switch` | `Switched to scene '(?P<scene>[^']*)'` | **arbitrary** | The documented wording. Studio-mode preview switches may log differently | Zero hits in `--check`; or scene changes appearing that never happened, which would be studio mode |
| `patterns.recording_start` / `recording_stop` | `={2,}\s*Recording (Start|Stop)\s*={2,}` | **arbitrary** | Deliberately loose about the `=` run, which has changed between OBS versions | Zero hits in `--check` |
| `patterns.recording_path` | `""` (unset) | **grounded** as a *choice*, arbitrary as a *value* | Empty on purpose. If OBS names its output file, a log holding several recordings can be matched to this master by filename — exact, and needing no clock. **Guessing the pattern would silently select the wrong recording**, which is worse than the refusal it would replace | Nothing, until a real log shows how OBS writes it. Setting it correctly is what turns the multi-recording refusal into a match |

**Four things that are not parameters:**

- **No wall clock is reachable from the parser, and that is enforced
  structurally.** `t = elapsed(scene line) − elapsed(recording start)`, both read
  from inside the same file. Reading the log's *filename* — the obvious
  alternative, since it carries a local timestamp — would drag in a timezone and
  a DST discontinuity and be wrong by an hour twice a year with nothing
  downstream able to notice. That is §4.1's unrecoverable-offset class and A8's
  reason for existing. A test parses `obs_log.py`'s AST and asserts it imports
  no `datetime`, `time`, `calendar` or `zoneinfo`, so the guarantee is a
  property of the module rather than a habit.
- **A log with several recordings and nothing to tell them apart is REFUSED.**
  One OBS session can start and stop recording repeatedly. Rules, in order: the
  span whose logged output path ends with the master's filename; the only span;
  otherwise refuse. Picking one at random offsets every event by a constant.
- **There is no auto-discovery from `%APPDATA%\obs-studio\logs`.** Selecting
  the right log out of a directory needs the wall clock above, and selecting
  wrong is silent. `register --obs-log <path>` is explicit, and the
  beside-the-master fallback is `obs-log.txt` exactly — never `*.txt`, which
  would claim any stray notes file.
- **Scenes are stored as SPANS** (`t` .. `t_end`), the shape commit 36 settled
  for `menu_screen`. §9.3's consumer wants how long a scene was up, and a
  boundary is derivable from a span where a span is not derivable from an edge.
  The scene already up when recording began runs from t=0: OBS logs a switch when
  it happens, not when recording starts, so dropping it would leave the opening
  of every stream with no scene at all.

**Noted and deliberately not built:** a scene named "BRB" or "Starting Soon" is
exactly §6.4's `menu_screen` case, available with no Phase 7 vision — `gates.py`
already reads `menu_screen` events and is inert only because nothing writes them.
Wiring scene names to that gate would be **scoring on a parser that has never
seen a real log**, which is what C5 and §16's rejection of scene changes as a
scorer both forbid. Revisit once `--check` has run against real text.

## Preview assets (Phase 3, §7.2)

Unusually for this file, the two numbers that matter here are **grounded** — not
because previews are better understood than anything else, but because §7.2's
own command does not fit §1.3's time budget and the only way to find that out
was to run it.

**MEASURED**, five 2 s clips from real 720p footage (`Testvid.mp4`), SSIM against
a lossless reference of the same scaled window, extrapolated to §7.1's 120
candidates:

| variant | SSIM | bytes | s/clip | 120 candidates |
|---|---|---|---|---|
| §7.2 verbatim (libvpx defaults) | 0.9638 | 249 KB | 6.78 | **13.6 min** |
| `-cpu-used 5 -deadline good` | 0.9572 | 248 KB | 3.01 | 6.0 min |
| `-cpu-used 8 -deadline realtime` | 0.9564 | 320 KB | 0.71 | **1.4 min** |
| h264 `crf 28` | 0.9193 | 132 KB | 0.56 | 1.1 min |
| h264 `crf 23` | 0.9416 | 250 KB | 0.60 | 1.2 min |

Two conclusions, and they point in opposite directions:

- **VP9 is right and §7.2 is right to name it.** At a matched ~250 KB, h264
  scores 0.9416 against every VP9 variant's 0.9564+. This was worth checking
  because the first comparison, at h264's default `crf 28`, looked 4× faster
  *and* smaller — which is what comparing two encoders across different quality
  scales always looks like.
- **§7.2's speed preset is wrong for this budget.** §1.3 allows 20–40 minutes
  for all unattended processing and `extract.f0` already spends ~16. Spending
  another 13.6 on an asset whose purpose is saving seek latency is the wrong
  trade at 9.5× the cost of the realtime setting, for 0.007 of SSIM.

**And the synthetic fixture says the opposite**, which is why the numbers above
are from real footage: on `fixture_long` the same command runs at 2.04 s/clip
because that source is 640×360. The fixture understates the real cost by **3.4×**
and would have made §7.2's defaults look affordable.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `previews.clip.codec` / `container` | `libvpx-vp9` / `webm` | **grounded** | §7.2 names both, and the SSIM-at-matched-size comparison above agrees | A browser that cannot decode VP9, which none of the review UI's targets are |
| `previews.clip.duration_s` / `width` | `2.0` / `480` | **grounded** | §7.2 states the duration. The width is from §7.2's own `scale=480:-2` — note its prose says "480p", which conventionally means *height*; the command wins because §7.2 says to hardcode it | A 480×270 loop being too small to judge a moment by |
| `previews.clip.crf` | `40` | **grounded** | §7.2 states it | Previews too blocky to judge a moment, which is the only thing they are for |
| `previews.clip.cpu_used` / `deadline` | `8` / `realtime` | **grounded**, and the one deviation | The table above. Set `0` / `good` for §7.2 verbatim | The preview quality being visibly worse in a way that changes a rating. Watch `preview_bytes_total` too: this trades 28% more bytes for 9.5× the speed |
| `previews.thumbstrip.frames` / `width` | `5` / `160` | **grounded** | §7.2 states both | — |
| `previews.thumbstrip.quality` | `6` | **arbitrary** | ffmpeg's `-q:v`, 2 (best) to 31 (worst). Measured at ~16 KB per strip against §7.2's ~30 KB estimate, so there is room to lower it | Strips too coarse to tell two moments apart |
| `previews.enabled` | `true` | **plausible** | On by default (C6). It is a dependency of nothing, so turning it off costs the operator seek latency and nothing else | The stage's cost mattering on a real candidate count — `preview_bytes_total` and `stage_duration_s` are the numbers |

**Three things that are not parameters:**

- **Assets are named by their WINDOW, never by `candidate_id`.** §7.2's own
  command writes `previews/{candidate_id}.webm`, and candidates here are
  append-only generations — a re-score with different weights mints entirely new
  ids. Under §7.2's naming, §6.1's promise that re-scoring is "free and
  infinitely repeatable" would quietly come to include a full re-encode of every
  preview in the library. The clip is keyed on `t_peak` alone and the strip on
  the window, so a boundary nudge reuses the expensive asset and an identical
  re-score regenerates **nothing**. VERIFIED: a forced re-run and a forced
  re-score both left every file's mtime untouched.
- **§7.2's waveform PNG is deliberately not written.** `review/queries.py`
  already ships a downsampled envelope in the candidate payload and `review.js`
  draws it as inline SVG — vector, theme-aware, no files, no stage cost. A PNG
  would be a second copy of the same numbers, 120 more files per stream, and
  orphaned by every re-score. What was genuinely missing was §7.2's **second
  track**, and that is now in the payload: every available role is drawn over
  **one shared dB range**, because per-track normalisation would put a silent
  party mic at the same height as a shouting operator and lose the only
  comparison the picture exists to support.
- **The 2 s clip SHIFTS at the ends of a recording rather than shortening.**
  §7.2's `-ss {t_peak-1}` is negative for a peak inside the first second. C2
  says expand rather than contract, so the window slides and shortens only when
  the recording itself is shorter than two seconds.

## Render — captions (Phase 4, §8.3)

None of these has been seen on a real clip. They are the numbers that decide
whether captions *read*, and the only way to falsify them is to watch a finished
short — which is one more reason the next thing to do is record a stream.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `render.captions.group_size` | `4` | **plausible** | §8.3 says "3–5 words on screen at a time"; 4 is the middle. Not in §17 | Captions reading as too dense (too high) or the highlight jumping between screens mid-phrase (too low) |
| `render.captions.min_group_size` | `3` | **plausible** | §8.3's lower bound, used to stop a run ending with one word alone on screen | A last group of 5+ words looking crowded, meaning the rebalance should merge less eagerly |
| `render.captions.max_gap_s` | `1.2` | **arbitrary** | NOT IN §8.3. Something has to end a group at a silence, or the word before a four-second pause shares the screen with the word after it and the highlight sits on a word nobody is saying. 1.2 s was picked as "longer than a breath, shorter than a beat" | Groups splitting mid-sentence at ordinary pauses (too low), or a caption sitting on screen through a silence (too high) |
| `render.captions.tail_hold_s` | `0.35` | **arbitrary** | NOT IN §8.3. Without a hold the last word of a sentence vanishes on its own final syllable. Picked as "about a frame count you notice" | The last words of sentences feeling clipped, or captions lingering after the speaker has moved on |
| `render.captions.min_line_s` | `0.08` | **arbitrary** | Below this a highlight state is a flash rather than a read. Only fires on words that share a start, which is what interpolation produces | Words visibly never highlighting; or a flicker, meaning it is too low |
| `render.captions.font_size` | `64` | **arbitrary** | In PlayRes units against a 1080-wide output, so roughly 6% of frame width. A guess at what reads on a phone | Captions wrapping to three lines, or being unreadable at phone size |
| `render.captions.outline` | `3.0` | **plausible** | §8.3 wants "white with dark outline"; an outline has to survive gameplay of any brightness | Text disappearing against a bright explosion (too thin) or looking like a sticker (too thick) |
| `render.captions.margin_v` (`mic` 220, `party` 330) | | **arbitrary** | The two differ *only* so simultaneous speakers stack instead of colliding. The absolute values are a guess at "above the platform's own UI chrome" | Captions sitting under TikTok's caption bar or Shorts' title, or the two rows overlapping when both people talk |
| `render.captions.styles.mic.colour` | `#FFFFFF` | **plausible** | §8.3's "color A (e.g. white with dark outline)", verbatim | — |
| `render.captions.styles.party.colour` | `#7FE7FF` | **plausible** | §8.3's "color B (e.g. light cyan)" | Two speakers being hard to tell apart on a phone, or the party colour reading as a UI element |
| `render.captions.highlight` | `#FFD400` | **arbitrary** | NOT IN §8.3, which never names the highlight colour — the one thing that makes word-level highlighting visible at all. Yellow because it is distinct from both speaker colours | The highlight being invisible against gameplay, or too loud to read past |
| `render.captions.uppercase` | `false` | **plausible** | Short-form convention is uppercase, but the transcript's own casing is information and this is a taste decision | The operator preferring the convention after seeing five clips |
| `render.captions.wrap_style` | `0` | **grounded** | libass smart wrapping. 2 (no wrap) puts an unusually long group off the side of the frame with no warning | — |
| `render.handles_s` | `0.25` | **arbitrary** | Distinct from `export.handles_s` because they are different decisions: an editor conforming an FCPXML wants room, a short's first frame is the hook (§8.5) and dead air in front of it is worse than a tight cut | Clips starting mid-word (too small) or opening on silence (too large) |

## Render — crop and encode (Phase 4, §8.4)

**Every coordinate in `crop_templates.yaml` is arbitrary.** They are §8.4's
example numbers, and §8.4's example was not checked — its gameplay region is
`src` 960×800 into `dst` 1080×1110, a 23% vertical stretch. Nothing can validate
them except looking at a frame of a real stream. `clipforge render <id> --stills`
exists for exactly that loop: a still costs a second, an encode costs a minute.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `crop_templates.yaml: marvel_rivals_facecam` | §8.4's numbers | **arbitrary** | Copied verbatim from a spec example measured on no real canvas | One look at a frame of the operator's own OBS layout |
| `crop_templates.yaml: gameplay_only` | a guess at "the middle" | **arbitrary** | Something had to be there for a camera-off scene | Same |
| `crop_templates.yaml: full_frame` | whole frame, centre-cropped | **grounded** | Not a guess at all: it is the identity crop for any 16:9 source, which is why it is the default. It throws away the sides and is never *wrong* | Nothing — it is arithmetic |
| `render.crop.fit` | `fill` | **plausible** | §8.4's own example distorts by 23% under `stretch`. `fill` keeps geometry true at the cost of a sliver of edge | Faces or HUD elements being clipped at region edges, which would argue for `contain` |
| `render.crop.upscale_warn_factor` | `2.0` | **arbitrary** | A 16:9 master reframed to 9:16 always enlarges ~1.78×, so warning on *any* enlargement fires on every clip and means nothing. 2.0 is above that floor and below the 3× a small facecam needs | The warning still firing on every render (too low), or a visibly mushy facecam passing silently (too high) |
| `ASPECT_TOLERANCE` | `0.01` | **grounded** | Covers 1920×1080 against 1280×720 rounding and nothing else; 16:9 against 16:10 is 11% and must not pass | Nothing — it is arithmetic |
| `render.video.crf` / `preset` | `18` / `slow` | **plausible** | §8.3 states both. Deliberately libx264 rather than `ingest.proxy`'s auto-detected hardware encoder: that detector optimises a three-hour proxy for throughput, and a 45-second deliverable optimises for quality | File sizes too large for a platform, or visible blocking in fast motion |
| `render.video.audio_bitrate` | `192k` | **plausible** | §8.3 states it | — |
| `render.verify.max_duration_delta_s` | `0.5` | **plausible** | Mirrors `ingest.proxy.verify.max_duration_delta_s`; a clip that came out the wrong length is worth catching before upload | A legitimate encode failing verification |
| `render.audio.source` | `auto` | **grounded** | §8's burn-in has a bare `-af`, so ffmpeg takes stream 0 — the mix only by luck of the OBS layout. `auto` reuses `proxy.audio_map_index`, which §5.2 already needed for the same reason | A clip carrying game audio and no voice |

## Render — loudness and presets (Phase 4, §8.2/§8.3)

Unusually for this file, most of these are **grounded** — the whole point of
loudness normalisation is that the result is measurable, so it was measured
rather than reasoned about. Every number below comes from encoding a window of
the speech fixture and reading the resulting file on its own.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `render.loudness.target_lufs` | `-14.0` | **grounded** | §8.3 states it, and it is where the platforms normalise to — arriving there means they leave the audio alone | A platform changing its target, or clips sounding quiet against others in a feed |
| `render.loudness.true_peak_db` / `lra` | `-1.5` / `11` | **grounded** | §8.3 states both | Inter-sample clipping on a lossy re-encode (TP too high) |
| `render.loudness.two_pass` | `true` | **grounded** | MEASURED over six windows, encoded and measured standalone: one-pass 1.45 LU mean error, two-pass 0.42. Over nine jittered windows: worst case 3.00 vs 1.70. Better on both statistics | One-pass coming out closer on real footage — `test_two_pass_beats_one_pass_on_real_windows` reruns the comparison and fails if so |
| `render.loudness.max_gain_db` | `20.0` | **plausible** | MEASURED: the fixture's authored silence reads −36.2 LUFS and would need +22.2 dB, where its quiet speech needs 4–10. 20 sits between them. The *shape* of the knob is grounded (gain, not an absolute floor, so it tracks `target_lufs`); the number is not | A quiet-but-real clip being skipped, or a silent one still being lifted |
| `render.loudness.verify_tolerance_lu` | `2.0` | **grounded** | Set outside measured variance rather than inside it: two-pass came out 0.63 LU off on average and 1.70 at worst, and a 0.2 s window shift moved one result 1.9 LU. A tighter tolerance would fire on ordinary clips and stop being read | Warnings never firing (too loose), or firing on clips that sound fine (too tight) |
| `render.loudness.sample_rate_hz` | `48000` | **grounded** | MEASURED: `loudnorm` hands the encoder 192 kHz unless something pins the rate, and AAC-LC tops out at 96 | — |
| `render.presets.*.max_duration_s` | `180` / `600` / `90` | **arbitrary** | Best guess at each platform's current limit. **These change, and were not verified from here.** Warned about, never enforced by truncation | An upload rejected for length, or a clip warned about that the platform accepted |
| `render.presets.default` | `shorts` | **arbitrary** | Something had to be first. The three are otherwise identical today | Nothing — pick whichever you post to most |

**Two things measured that are not parameters**, recorded so nobody re-derives
them the hard way:

- **`loudnorm`'s own `input_i` cannot detect a silent window.** Over the
  fixture's authored silence it reports −17.75 LUFS where a standalone
  `ebur128` reports −36.2, because its gate sits above the noise floor. The
  silence guard therefore reads `ebur128`, in its own pass.
- **`linear=true` was never granted on this material** — every run came back
  `normalization_type: dynamic`, because the gain needed would push true peak
  past the ceiling. The flag is still set for quieter sources, but nothing here
  may claim the normalisation is a transparent linear gain.

## Render — muting and filler removal (Phase 4, §8.6/§8.2)

**Both features are off by default**, so none of these affects a normal render.
They are here because turning one on makes every number below live.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `render.mute.enabled` | `false` | **grounded** | §16 lists "profanity muting by default" among the explicitly rejected features | — |
| `render.mute.pad_s` | `0.08` | **plausible** | MEASURED that the boundary leaks: muting exactly 2.0–4.0 s left the edges audible, and a mute that starts late leaves the consonant. C2 — expand rather than contract. The *need* for padding is grounded; 80 ms is not | A swear word still audible at its start or end (too small), or a syllable of the surrounding words lost (too large) |
| `render.mute.words` | `null` → `phrases.yaml` | **grounded** | The profanity list already exists for §5.6's `swear_density`. One list, one place to edit | — |
| `render.filler.enabled` | `false` | **grounded** | Your instruction, and §15's "ship, then stream ten times" | — |
| `render.filler.words` | 9 general-English fillers | **arbitrary** | Exactly the status of `phrases.yaml`'s profanity list: assembled from general English, not from this operator's speech. **Ten streams of transcript would settle it in minutes** | The list cutting words you meant to say, or missing the ones you actually fill with |
| `render.filler.roles` | `[mic]` | **grounded** | Cutting the other person's speech because the operator said "um" is not a thing anyone wants | — |
| `render.filler.min_duration_s` | `0.12` | **arbitrary** | Below this a cut is a glitch rather than an edit. No evidence for the exact number | Audible clicks at cut points (too small), or real fillers left in (too large) |
| `render.filler.merge_gap_s` | `0.35` | **arbitrary** | Two removals closer than this leave an island of audio that reads as a stutter. Picked as "shorter than a beat" | Stuttery cuts surviving, or unrelated speech between two fillers being swallowed |
| `render.filler.max_share` | `0.25` | **arbitrary** | A plan removing more than a quarter of a clip means the word list is matching something it should not, rather than the clip being full of filler. Refusing the whole plan is the honest response to that | The refusal firing on a clip that genuinely is a quarter filler |

**The thing no parameter can settle:** whether cutting filler improves a clip at
all. Cutting a word cuts the video, and on gameplay that is a visible jump.
The mechanism is deterministic and tested; the taste question needs footage.

## Render — hook text (Phase 4, §8.5)

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `render.hooks.source` | `manual` | **grounded** | The one that can run. Since commit 42 this key also accepts `anthropic`, which reports itself unavailable without a key; §12.4 prices one at roughly $0.10–0.30 per stream. See "LLM transport" below | — |
| `render.hooks.options` | `5` | **grounded** | §8.5 says "propose 5 hook variants" | Five being more than you ever read, or fewer than you need to find a good one |

**Worth watching once real replies exist:** `llm_invalid_id_rate` in
`tool_metrics`. §14 calls it hallucination monitoring, and it is now being
written on every `--apply`. A rate that climbs means the prompt has stopped
being clear, not that the model got worse.

**Not a guess, but recorded because it looks like one:** the ASS file is
referenced from the filter graph by bare filename with ffmpeg run from its
directory. MEASURED — no escaping of an absolute Windows path survives ffmpeg's
filter parser once a parent directory contains an apostrophe. See
`ffmpeg.filter_file`.

## Backup (§13.2)

Unusually, almost nothing here is a guess: §13.2 states the retention numbers and
the compression step, and the rest was measured on this database. The one
invented value is the mirror.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `backup.keep_daily` / `keep_monthly` | `30` / `12` | **grounded** | §13.2 states both. The *rule* for applying them is not in the spec and is written down in `db/backup.py` | Wanting a backup from a day that has been pruned. At ~88 KiB each, 42 files is under 4 MB — raising these costs nothing |
| `backup.compress` | `true` | **grounded** | §13.2's gzip step. MEASURED on this database: 408 KiB → 88 KiB (78% smaller), whole backup ~30 ms | Nothing at this size. A multi-GB database where the CPU time started to matter |
| `backup.dir` | `./data/backups` | **plausible** | One directory holds everything the app owns. But it is therefore on the same disk as the database, which is exactly why `mirror_dir` exists | Nothing — the placement is a convenience; the protection comes from the mirror |
| `backup.mirror_dir` | `null` | **arbitrary** | NOT IN §13.2, which says "upload to B2/S3". No account exists (C5), and this is what is available with no credentials. Off by default because there is nowhere to point it yet | Losing the disk. That is the whole falsifier, and it only fires once |
| `backup.schedule_time` | `04:00` | **arbitrary** | Late enough that a stream is over, early enough to precede the next one. Only fills in the printed `schtasks` line | Streaming past 04:00, in which case the backup runs mid-session — which is safe (MEASURED against an open WAL connection) but pointlessly timed |
| `doctor` stale-backup threshold | `7` days | **arbitrary** | In `doctor._check_backups`, not config: it is a display heuristic, not a tunable. "Long enough that a nightly has definitely stopped" | Warning while the schedule is healthy, or staying quiet for a fortnight after it broke |

**Three things measured that are not parameters**, recorded so nobody re-derives
them the hard way:

- **`VACUUM INTO` works over a `file:...?mode=ro` connection**, and the source's
  SHA-256 is unchanged across a copy. The read-only open is load-bearing:
  `db.connect`'s `PRAGMA journal_mode=WAL` *does* rewrite the header of a
  database not already in WAL.
- **Committed but un-checkpointed WAL frames are captured.** With 626 KB of
  outstanding WAL and a live writer holding the connection, every row was in the
  copy. This is what makes a 04:00 backup trustworthy while the app is open.
- **`VACUUM INTO` refuses a non-empty target and silently accepts a zero-byte
  one.** So the existence check cannot be delegated to SQLite; every write goes
  through `atomic_output`, which unlinks first.

**Worth watching once nightlies are running:** `backup_duration_s` in
`tool_metrics` — an INVENTED name, since §14's table has no backup row. Its
`meta` carries `source_bytes` and `stored_bytes`, so it doubles as the growth
curve of §13.1's metadata tier, which is currently an estimate ("~5 MB/stream")
that nothing has checked.

## Weight tuning (§14, §17)

§14 names `signal_firing_rate_by_rating` **the primary weight-tuning input** and
§17 builds its whole procedure on it. Nothing computed it until commit 44b, and
nothing had ever read `candidates.feature_vector` at all — A9 has been filling
that column since Phase 1 for exactly this.

**The ranking column is threshold-free, and that is the load-bearing decision.**
`feature_vector` holds three different kinds of number under one roof: a rolling
z-score for `continuous`, a decayed kernel level 0..1 for `events` and
`composite`, and §6.4's gate ramp 0..1 for `afk`/`menu_screen`. A single firing
threshold across those cannot mean one thing. So signals are ranked on
**separation** — the probability that a randomly chosen *clip it* moment
outscores a randomly chosen *skip* one, i.e. the normalised Mann-Whitney U. It
is a rank statistic, so it reads the same over all three scales, and **below 0.5
means a signal discriminates the wrong way**, which no firing rate shows as such.

§14's literal firing rate is reported beside it because §14 names it. Events,
composites and gates fire at `> 0`, which is grounded rather than chosen: that is
what a kernel level of zero means. Only `continuous` needs a cut, which is why
there is exactly one arbitrary number below rather than three.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `tuning.firing_threshold_z` | `1.0` | **arbitrary** | One standard deviation above the rolling baseline is a conventional reading of "notable" and nothing more; no real distribution of these values has been looked at. It moves the two rate columns and **cannot** move the ranking — a test asserts that | The rate columns reading as all-or-nothing (too low or too high). Nothing about the ranking, by construction |
| `tuning.min_rated_moments` | `40` | **arbitrary** | Below this no ranking prints at all. The honest number is unknowable in advance — it is however many ratings make a per-signal comparison mean something, which depends on how often each signal fires. §17 tunes "after every ~5 streams" and §6.7 targets 80–150 candidates per 3 h, so five streams is a few dozen ratings. **A deliberately low bar**: it exists to stop a table built from three ratings reading as evidence, not to certify that 40 is enough | Ranking that still looks like noise above the threshold (raise it), or a real signal being withheld well past five streams (lower it) |
| `tuning.min_moments_per_class` | `10` | **arbitrary** | Per signal, per class. Two observations each give a separation of 1.0 or 0.0 almost by chance, and printed beside signals with two hundred it would sort to the top. 10 is "a perfect separation is not a coin flip" (2⁻¹⁰ ≈ 0.1%) and nothing more principled | Signals with a suspiciously perfect separation and a small `n` reaching the top of the table |
| `tuning.approved_rating` | `2` | **grounded** | §7.3's "clip it", and what `export --min-rating` already defaults to for the same reason | — |

**Three things that are not parameters:**

- **Ratings are read through `clipforge/moments.py`, never `is_current`.** The
  obvious query is `review_metrics`' own, and it returns NOTHING on a re-scored
  stream: after commit 43 the operator's row stays on the superseded generation
  while the current one carries an `'inherited'` copy the `'operator'` filter
  excludes. The primary tuning input would have read zero on exactly the corpus
  it exists to measure. One opinion per moment, across generations, latest wins.
- **Only the schema's DECLARED keys are scored.** MEASURED on the real database:
  every candidate in it predates `feature_schema` version 2 and its vectors carry
  context keys the current writer no longer emits — `mic_rms_db` is an absolute
  dB level, and iterating the stored JSON rather than `feature_schema.keys` would
  have ranked it as a signal.
- **`marker_precision` uses `press_inside` only.** The looser §7.4 reading also
  counts a non-zero marker *contribution*, which comes from
  `contributing_signals`, which `features.breakdown` builds from **weighted**
  tracks — so it moves when a marker weight moves, and a weight-tuning input that
  does that cannot tune anything. A test zeroes the weight and asserts the
  precision is unmoved.

**Gaps 2 and 3 below are closed by this.** What is still missing is footage: on
the real database today the corpus is 3 ratings, and the command correctly
refuses to rank.

## The rubric (the learning layer — no § reference)

The spec never gave the learning layer a section, so unlike everything else in
this file there is no spec default to compare against. There is also almost
nothing to tune: the rubric is prose, and prose has no parameters.

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `rubric.warn_chars` | `6000` | **arbitrary** | Every downstream prompt carries the rubric's full text, so its size is worth a warning — but it is **never truncated**, because silently dropping the end of the operator's own judgement is the exact failure the rubric exists to prevent. §12.4 budgets a theme call at ~4k tokens and an assembly at ~5k, so 6000 characters is roughly 1.5k — about a third of the smaller budget, which felt like where guidance starts crowding out the material it is guidance about. **No rubric has ever been written** | The warning firing on a rubric that reads as concise (raise it), or a rubric visibly dominating a prompt without ever firing (lower it) |

**Three things that are not parameters:**

- **The rubric is a table, not a config file.** `prompts.yaml`, `phrases.yaml`
  and `crop_templates.yaml` are git-tracked, hand-edited and read-only at
  runtime. This is operator-authored through the review UI, versioned, and
  appended per review batch — and a server writing into its own installed
  package directory would be wrong.
- **Append-only, and the version integer is the identity.** There is
  deliberately no `rubric_of()` hash beside `prompts.digest_of()`: that one
  exists because a prompt template lives in a file that can be edited in place,
  so its identity is its content. A rubric row is immutable, so a hash would be
  a second name for the same thing.
- **One free-text column, not named sections.** `what_worked` / `what_didnt` /
  `watch_for` would decide the shape of the operator's thinking before a single
  rubric had been written, and markdown headings inside `text` cost nothing if a
  shape turns out to want one. C5, applied to a table.

## Review and export

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `review.target_ms_per_candidate` | `4000` | **grounded** | §7.1's hard target, arithmetic from "120 in under 8 minutes" | — |
| `review.nudge_step_s` | `0.5` | **grounded** | §7.3 states 0.5 s for `[`/`]`/`{`/`}` | Every adjustment taking a dozen presses (too small), or never landing where you want it (too large). The keypress count is in each `window_nudge_s` row, so this one falsifies itself |
| `export.fcpxml_version` | `1.10` | **plausible** | Current at time of writing | Resolve rejecting the file — try `1.8` |
| `export.source` | `master` | **grounded** | The master is the quality source; §10.5 exports reference it | — |
| `export.handles_s` | `0.0` | **arbitrary** | No handles by default. Editors often want a second either side | Finding every clip needs manual extension in Resolve |
| `--min-rating` default | `2` | **plausible** | §7.3's "clip it" | Exports routinely missing moments rated `maybe` |

## Deferred — declared but wired to nothing

Present so §17 is fully represented in config and nobody hardcodes them later. All
**plausible** at best; each is the spec's own default and none has been exercised.

`deferred.trends.hdbscan_min_cluster_size` (5),
`deferred.trends.ngram_recency_halflife_days` (30).

Moved out as the phase consuming each was built: `laughter.band_hz` to
`extract.laughter.band_hz` (commit 33), and `negatives.menu_grace_period_s` /
`negatives.afk_threshold_s` to `score.negatives` (commit 36).

## Not listed

Pure plumbing that cannot change what the system detects or produces: `review.host`,
`review.port`, `review.browse.max_entries`, `ingest.stream_id_slug_max_len`,
`ingest.proxy.codec_preference` (auto-detected by test-encoding), file paths, and codec
argument blocks. Wrong values there fail loudly rather than silently.

---

## Gaps — guesses with no way to check them yet

These are the ones to be uneasy about: the parameter is unvalidated **and** the
observation that would falsify it is not being recorded.

1. ~~**Window length.**~~ **CLOSED in commit 30.** §7.3's `[`/`]`/`{`/`}` keys are built
   and every nudge writes a `window_nudge_s` row carrying direction, size, keypresses,
   whether the detector's window had been clamped to `min_window_s`/`max_window_s`, and
   whether the operator's window still contains the peak. `clipforge metrics` reports it.
   **Still needs ~10 real streams before the numbers mean anything** — the gap now is
   footage, not instrumentation.

   Note what the nudge deliberately does *not* do: it is **not clamped to
   `min_window_s`/`max_window_s`**. Those are the values being measured, so refusing a
   window outside their range would make the measurement circular — the operator could
   never record "I wanted this shorter than 8 seconds". VERIFIED in a browser: a window
   nudges down to 0.6 s against a `min_window_s` of 8.
2. ~~**Marker precision and recall.**~~ **CLOSED in commit 44b.** Both are computed by
   `clipforge metrics --tuning`, over `events` and the moments `clipforge/moments.py`
   assembles, using the strict "press inside the window" reading. **Still needs ~10 real
   streams before the numbers mean anything** — the gap now is footage, not
   instrumentation. They are the direct test of `score.markers.retro_offset_s`.
3. ~~**`signal_firing_rate_by_rating`.**~~ **CLOSED in commit 44b.**
   `clipforge metrics --tuning` aggregates the A9 vectors and ranks signals by how well
   they separate *clip it* from *skip*; `--record` writes §14's own metric name into
   `tool_metrics` where §17 says to pull it from. **It refuses to rank below
   `tuning.min_rated_moments`**, which on today's 3 ratings is what it does. The gap is
   footage, not instrumentation.
4. **Profanity list.** Assembled from general English, not from the operator's speech.
   The first ten streams' transcripts would settle it in minutes.
