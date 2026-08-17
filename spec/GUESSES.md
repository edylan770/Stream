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
| `score.profile` weights | `marker_definite 3.0`, `marker_maybe 1.5`, `mic_rms 1.0` | **plausible** | §6.5's `entertainment` values for exactly the signals that exist, unchanged so the scale does not shift under existing ratings when Phase 3 adds the rest | `signal_firing_rate_by_rating` showing a signal fires equally on rating-0 and rating-2 candidates — it is not discriminating and its weight is wrong |
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
| `score.rating_inherit_min_overlap` | `0.5` | **arbitrary** | Half a window felt like "the same moment". No evidence | Ratings failing to carry across a re-score (too high), or carrying onto a moment that is not the same one (too low). `render/selection.py` reads across generations precisely because this is unreliable |
| `score.combined.alpha` | `0.5` | **plausible** | §17's default. Unused in Phase 1 — one profile exists | Whether combined-score winners are actually the best clips (§6.5) |
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
| `render.hooks.source` | `manual` | **grounded** | The only one built. An API-backed source needs a key the operator does not want yet; §12.4 prices one at roughly $0.10–0.30 per stream | — |
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

`deferred.negatives.menu_grace_period_s` (8), `deferred.negatives.afk_threshold_s` (60),
`deferred.laughter.band_hz` ([4.0, 7.0]), `deferred.trends.hdbscan_min_cluster_size` (5),
`deferred.trends.ngram_recency_halflife_days` (30).

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
2. **Marker precision and recall.** §14 defines `marker_precision` and
   `marker_recall_proxy` and they are the direct test of `retro_offset_s`. Neither is
   computed yet; both are pure SQL over `ratings` and `events` once ten streams exist.
3. **`signal_firing_rate_by_rating`.** §14 calls this "the primary weight-tuning input"
   and §17's whole procedure depends on it. The data is being logged — full feature
   vectors per candidate, per A9 — but nothing aggregates it. Needed before any weight
   is changed on evidence.
4. **Profanity list.** Assembled from general English, not from the operator's speech.
   The first ten streams' transcripts would settle it in minutes.
