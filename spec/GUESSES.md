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
| `score.window.min_window_s` / `max_window_s` | `8` / `60` | **plausible** | §17's defaults | How often the operator nudges boundaries during review. §7.3's `[`/`]` keys are unbuilt, so this is currently unobservable — **a gap** |
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

## Review and export

| Parameter | Value | Confidence | Rationale | Falsified by |
|---|---|---|---|---|
| `review.target_ms_per_candidate` | `4000` | **grounded** | §7.1's hard target, arithmetic from "120 in under 8 minutes" | — |
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

1. **Window length.** §17 tunes `min_window_s`/`max_window_s` against "how often the
   operator nudges boundaries during review" — and §7.3's `[`/`]`/`{`/`}` nudge keys are
   not built, so nudges cannot be counted. Either build the keys or accept the defaults
   untested.
2. **Marker precision and recall.** §14 defines `marker_precision` and
   `marker_recall_proxy` and they are the direct test of `retro_offset_s`. Neither is
   computed yet; both are pure SQL over `ratings` and `events` once ten streams exist.
3. **`signal_firing_rate_by_rating`.** §14 calls this "the primary weight-tuning input"
   and §17's whole procedure depends on it. The data is being logged — full feature
   vectors per candidate, per A9 — but nothing aggregates it. Needed before any weight
   is changed on evidence.
4. **Profanity list.** Assembled from general English, not from the operator's speech.
   The first ten streams' transcripts would settle it in minutes.
