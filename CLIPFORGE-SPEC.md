# Stream → Clip → Video Pipeline: Complete Build Specification

**Document version:** 1.0
**Purpose:** Full technical specification for an application that ingests stream recordings, detects clip-worthy moments through deterministic signal analysis, assists in short-form clip production, and generates YouTube video concepts from single-stream and cross-stream data.

---

# TABLE OF CONTENTS

1. [Strategic Context and Non-Negotiable Constraints](#1-strategic-context-and-non-negotiable-constraints)
2. [System Architecture](#2-system-architecture)
3. [Data Model](#3-data-model)
4. [Layer 1: Capture](#4-layer-1-capture)
5. [Layer 2: Extraction](#5-layer-2-extraction)
6. [Layer 3: Scoring](#6-layer-3-scoring)
7. [Layer 4: Review UI](#7-layer-4-review-ui)
8. [Layer 5: Auto-Finish Renderer](#8-layer-5-auto-finish-renderer)
9. [Layer 6: Digest Generation](#9-layer-6-digest-generation)
10. [Layer 7: YouTube Pipelines](#10-layer-7-youtube-pipelines)
11. [Layer 8: Trend Detection and Ideas Database](#11-layer-8-trend-detection-and-ideas-database)
12. [LLM Usage Rules](#12-llm-usage-rules)
13. [Storage, Backup, and Retention](#13-storage-backup-and-retention)
14. [Instrumentation](#14-instrumentation)
15. [Build Order](#15-build-order)
16. [Explicitly Rejected Features](#16-explicitly-rejected-features)
17. [Open Parameters Requiring Empirical Tuning](#17-open-parameters-requiring-empirical-tuning)

---

# 1. STRATEGIC CONTEXT AND NON-NEGOTIABLE CONSTRAINTS

## 1.1 What this tool is for

The operator is a beginning gaming content creator. Primary content identity is **comedy and entertainment**, typically co-streaming with his girlfriend, across Marvel Rivals (ranked, Hawkeye main, Celestial rank), Valorant (casual), Roblox, and assorted co-op games. Secondary content is **gameplay highlights** — flashy Hawkeye plays — treated as supporting material rather than the channel's foundation.

The tool exists to reduce time-to-published-content. It does not exist to replace judgment.

## 1.2 Design constraints that govern every decision below

These are ordered by importance. When a design question arises that this document does not answer, resolve it against these principles.

**C1. Deterministic over probabilistic.** Wherever a deterministic method exists, use it. Models are used only where determinism is impossible (natural language understanding, semantic similarity, idea generation). Every model output that references time or content must be validated against a deterministic index.

**C2. Recall over precision.** A false positive costs ~3 seconds of review time. A false negative costs a clip permanently. Tune thresholds low. Generate 80–150 candidates per 3-hour stream. Make review cheap rather than making detection precise.

**C3. Extraction and scoring are separate, with a hard boundary.** Extraction is expensive and runs once per stream. Scoring is cheap and must be re-runnable over the entire back catalog at any time. This means early bad weight choices cost nothing.

**C4. The review UI is the critical path.** Every other subsystem feeds one screen. If review is slow, the tool goes unused and the entire system is dead weight. Target: **120 candidates reviewed in under 8 minutes.** If review exceeds this, fix the UI before adding any new feature.

**C5. Never build ahead of the data.** Signal weights, thresholds, and window lengths in this document are educated guesses, not validated values. They must be tuned against real footage. Do not build subsystems whose value depends on a corpus that does not yet exist.

**C6. Log everything from day one, even for features not yet built.** Feature vectors, ratings, and digests cannot be reconstructed retroactively. Data captured and unused costs nothing. Data not captured is gone.

**C7. Idempotent and resumable.** Every extraction stage checks whether its output already exists and skips if so. A crash mid-pipeline must never destroy completed work.

**C8. Total hands-on time per stream must stay at or below ~35 minutes.** If a feature increases this, cut a different feature or cut the new one.

## 1.3 Realistic time budget

| Task | Target |
|---|---|
| Unattended processing (overnight) | 20–40 min for a 4-hour stream |
| Candidate review | 8 min |
| Finishing 3–5 shorts (post-auto-finish) | 10–15 min |
| Long-form assembly review | 20 min (only on days a long-form is assembled) |

---

# 2. SYSTEM ARCHITECTURE

## 2.1 Three physical layers

```
┌─────────────────────────────────────────────────────────┐
│ LAYER A: CAPTURE (during stream, on streaming PC)       │
│ Independent, dumb, crash-proof. Writes files only.      │
│ - OBS recording (multi-track audio)                     │
│ - Marker hotkey daemon → JSONL                          │
│ - Input activity logger → JSONL                         │
│ - Record-start anchor → JSON                            │
└─────────────────────────────────────────────────────────┘
                          ↓ files
┌─────────────────────────────────────────────────────────┐
│ LAYER B: PROCESSING (after stream, unattended)          │
│ The application. Reads files, writes DB.                │
│ - Proxy generation                                      │
│ - Signal extraction (audio, transcript, vision)         │
│ - Embedding generation                                  │
│ - Scoring → candidates                                  │
│ - Preview asset generation                              │
│ - Digest generation                                     │
└─────────────────────────────────────────────────────────┘
                          ↓ SQLite
┌─────────────────────────────────────────────────────────┐
│ LAYER C: INTERACTION (whenever)                         │
│ - Review UI                                             │
│ - Auto-finish renderer                                  │
│ - Semantic search                                       │
│ - Idea dashboard                                        │
│ - YouTube assembly                                      │
└─────────────────────────────────────────────────────────┘
```

**Critical:** Layer A must be independently runnable and must not depend on the application being installed or functional. If the input logger crashes mid-stream, one signal is lost — the stream is not.

## 2.2 Technology stack

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Ecosystem for audio/ML is unmatched; ffmpeg bindings mature |
| Database | SQLite (WAL mode) | Single-file, zero-admin, trivially backed up, fast enough at this scale |
| Media | ffmpeg / ffprobe (CLI subprocess) | Universal, reliable, no binding fragility |
| Transcription | WhisperX (local, GPU) | Word-level timestamps via forced alignment; free/BSD-4 |
| Audio analysis | librosa, numpy, soundfile | Standard |
| Embeddings | Ollama serving `bge-small-en-v1.5` (384-dim) or `nomic-embed-text` (768-dim) | Local, free, fast |
| Clustering | HDBSCAN (`hdbscan` package) | Handles variable-density clusters, no k required |
| Vision/OCR | OpenCV template matching; Tesseract only if needed | Template match is faster and more reliable than OCR for fixed UI |
| Reasoning LLM | External API (frontier model) | Quality difference on ideation is large; cost is negligible (see §12.4) |
| UI | Local web app (FastAPI + vanilla JS, or Tauri) | Must be fast; avoid heavy frameworks |

**On local vs. external models:** Transcription, embeddings, and audio analysis run locally because they are GPU-bound and the operator owns a GPU. Reasoning steps (digest synthesis, theme generation, video ideation) run through an external API because quality matters disproportionately there and cost is trivial. Do not be ideological about local-only.

## 2.3 Directory layout

```
clipforge/
├── clipforge/
│   ├── capture/          # Layer A scripts (deployed to streaming PC)
│   │   ├── marker_daemon.py
│   │   ├── input_logger.py
│   │   └── obs_anchor.py
│   ├── ingest/           # Stream registration, proxy gen, file discovery
│   ├── extract/          # One module per signal family
│   │   ├── audio.py
│   │   ├── transcript.py
│   │   ├── vision.py
│   │   └── embeddings.py
│   ├── score/            # Profiles, weighting, window generation
│   ├── review/           # UI backend + frontend
│   ├── render/           # Captions, reframe, normalize, export
│   ├── digest/           # Digest generation
│   ├── ideate/           # Theme, assembly, cross-stream
│   ├── trends/           # Clustering, n-grams, scheduler
│   ├── db/               # Schema, migrations, queries
│   └── config/           # Weight profiles, crop templates, vocab
├── data/
│   ├── clipforge.db
│   ├── streams/<stream_id>/
│   │   ├── raw/          # Symlinks or paths to masters
│   │   ├── proxy/
│   │   ├── audio/        # Extracted per-track WAV
│   │   ├── previews/     # 2s webm previews, thumb strips, waveforms
│   │   └── exports/      # Finished clips
│   └── backups/
└── spec/
    └── CLIPFORGE-SPEC.md  # This document
```

---

# 3. DATA MODEL

## 3.1 Core principle

**Video files are immutable blobs. Everything else is metadata pointing into them.**

A clip is never a file until export. It is `(stream_id, t_start, t_end)`. All retrieval is SQL returning tuples, resolved to file paths at render time. This is what allows the system to scale to hundreds of hours without changing.

## 3.2 Schema

```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ============================================================
-- STREAMS
-- ============================================================
CREATE TABLE streams (
    id                    TEXT PRIMARY KEY,      -- e.g. '2026-08-14_marvel'
    date                  TEXT NOT NULL,         -- ISO date
    title                 TEXT,
    games                 TEXT,                  -- JSON array
    profile_used          TEXT,                  -- weight profile name
    master_path           TEXT NOT NULL,
    proxy_path            TEXT,
    record_start_epoch_ms INTEGER NOT NULL,      -- THE sync anchor
    duration_s            REAL,
    fps                   REAL,
    resolution            TEXT,
    audio_track_map       TEXT,                  -- JSON: {"mic":1,"game":2,"party":3}
    obs_scene_log_path    TEXT,
    notes                 TEXT,
    created_at            TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- PIPELINE STATE (idempotency / resumability)
-- ============================================================
CREATE TABLE pipeline_stages (
    stream_id     TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    stage         TEXT NOT NULL,     -- 'proxy','audio_split','whisperx',...
    status        TEXT NOT NULL,     -- 'pending','running','done','failed'
    started_at    TEXT,
    finished_at   TEXT,
    output_hash   TEXT,
    error         TEXT,
    attempt_count INTEGER DEFAULT 0,
    PRIMARY KEY (stream_id, stage)
);

-- ============================================================
-- CONTINUOUS SIGNALS  (stored as BLOB arrays, NOT rows)
-- ============================================================
CREATE TABLE signal_series (
    stream_id     TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,     -- 'mic_rms','mic_f0','speech_rate',...
    sample_rate_hz REAL NOT NULL,    -- typically 1.0 or 10.0
    t0            REAL DEFAULT 0.0,  -- offset of first sample, seconds
    n_samples     INTEGER NOT NULL,
    dtype         TEXT DEFAULT 'float32',
    data          BLOB NOT NULL,     -- raw numpy array bytes
    PRIMARY KEY (stream_id, kind)
);
-- RATIONALE: row-per-second-per-signal would produce ~8.6M rows over
-- 100 streams. As float32 arrays this is a few hundred MB and loads
-- in milliseconds.

-- ============================================================
-- DISCRETE EVENTS  (everything point-in-time gets a row)
-- ============================================================
CREATE TABLE events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id  TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    t          REAL NOT NULL,        -- seconds into VOD
    t_end      REAL,                 -- NULL for instantaneous
    source     TEXT NOT NULL,        -- 'marker','ocr','scene','phrase','laugh'
    kind       TEXT NOT NULL,        -- 'marker_maybe','kill','multikill',...
    value      REAL,                 -- magnitude/confidence, nullable
    meta       TEXT,                 -- JSON
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_events_stream_t ON events(stream_id, t);
CREATE INDEX idx_events_kind     ON events(kind);
-- RATIONALE: uniform shape means adding a new discrete signal requires
-- ZERO schema migration. New sensor = new `source` value.

-- ============================================================
-- TRANSCRIPT
-- ============================================================
CREATE TABLE segments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id  TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,     -- sequential within stream; LLM-facing ID
    t_start    REAL NOT NULL,
    t_end      REAL NOT NULL,
    text       TEXT NOT NULL,
    speaker    TEXT,                 -- 'operator','party','unknown'
    track      INTEGER,              -- which audio track it came from
    words      TEXT                  -- JSON: [{w,start,end,score},...]
);
CREATE INDEX idx_segments_stream_t ON segments(stream_id, t_start);
CREATE UNIQUE INDEX idx_segments_seq ON segments(stream_id, seq);

CREATE TABLE segment_embeddings (
    segment_id INTEGER PRIMARY KEY REFERENCES segments(id) ON DELETE CASCADE,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vec        BLOB NOT NULL         -- float32
);

-- ============================================================
-- CANDIDATES
-- ============================================================
CREATE TABLE candidates (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id            TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    t_start              REAL NOT NULL,
    t_end                REAL NOT NULL,
    t_peak               REAL NOT NULL,
    score_entertainment  REAL NOT NULL,
    score_gameplay       REAL NOT NULL,
    score_combined       REAL NOT NULL,   -- see §6.5 — NOT a simple sum
    contributing_signals TEXT,            -- JSON {signal: contribution}
    feature_vector       TEXT NOT NULL,   -- JSON, FULL vector, always
    config_version       TEXT NOT NULL,   -- which weight profile produced this
    preview_path         TEXT,
    thumbstrip_path      TEXT,
    waveform_path        TEXT,
    created_at           TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_candidates_stream ON candidates(stream_id);
CREATE INDEX idx_candidates_scores ON candidates(score_combined DESC);

-- ============================================================
-- RATINGS  (the primary training signal)
-- ============================================================
CREATE TABLE ratings (
    candidate_id INTEGER PRIMARY KEY REFERENCES candidates(id) ON DELETE CASCADE,
    rating       INTEGER NOT NULL,   -- 0=skip, 1=maybe, 2=clip it
    tags         TEXT,               -- JSON array, free-form
    note         TEXT,
    adjusted_start REAL,             -- if operator trimmed
    adjusted_end   REAL,
    rated_at     TEXT DEFAULT (datetime('now')),
    review_ms    INTEGER             -- time spent, for instrumentation
);

-- ============================================================
-- EXPORTS / PUBLISHED
-- ============================================================
CREATE TABLE exports (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER REFERENCES candidates(id),
    path         TEXT NOT NULL,
    preset       TEXT,               -- 'shorts','tiktok','reels'
    hook_text    TEXT,
    exported_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE performance (
    export_id       INTEGER PRIMARY KEY REFERENCES exports(id),
    platform        TEXT,
    views           INTEGER,
    retention_pct   REAL,
    shares          INTEGER,
    normalized_score REAL,           -- vs rolling median of last 20
    measured_at     TEXT
);

-- ============================================================
-- DIGESTS
-- ============================================================
CREATE TABLE digests (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id  TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    version    INTEGER NOT NULL,
    content    TEXT NOT NULL,        -- JSON, structured (see §9.2)
    markdown   TEXT,                 -- human-readable mirror
    model_used TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX idx_digest_ver ON digests(stream_id, version);

-- ============================================================
-- IDEAS  (living records, accumulate evidence over months)
-- ============================================================
CREATE TABLE ideas (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,       -- 'per_stream','cross_stream','bit','compilation'
    title       TEXT NOT NULL,
    premise     TEXT NOT NULL,
    beat_structure TEXT,             -- JSON array of beats
    status      TEXT DEFAULT 'open', -- 'open','ready','shot','published','dead'
    coverage_pct REAL,
    gaps        TEXT,                -- JSON: which beats lack footage
    origin      TEXT,                -- 'llm_ideation','cluster','ngram','manual'
    created_at  TEXT DEFAULT (datetime('now')),
    last_scored TEXT
);

CREATE TABLE idea_evidence (
    idea_id      INTEGER NOT NULL REFERENCES ideas(id) ON DELETE CASCADE,
    candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    beat_index   INTEGER,            -- which beat this fills
    role         TEXT,               -- 'cold_open','setup','payoff','filler'
    strength     REAL,
    added_at     TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (idea_id, candidate_id)
);

-- ============================================================
-- TRENDS
-- ============================================================
CREATE TABLE clusters (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    label          TEXT,             -- LLM-generated
    n_members      INTEGER,
    n_streams      INTEGER,          -- streams spanned — key metric
    first_seen     TEXT,
    last_seen      TEXT,
    centroid       BLOB,
    created_at     TEXT DEFAULT (datetime('now'))
);

CREATE TABLE cluster_members (
    cluster_id   INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    distance     REAL,
    PRIMARY KEY (cluster_id, candidate_id)
);

CREATE TABLE ngrams (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    phrase        TEXT NOT NULL,
    n             INTEGER NOT NULL,
    total_count   INTEGER,
    stream_count  INTEGER,
    first_seen    TEXT,
    last_seen     TEXT,
    recency_score REAL,              -- see §11.2
    is_baseline_tic INTEGER DEFAULT 0,
    UNIQUE(phrase)
);

CREATE TABLE open_loops (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id   TEXT REFERENCES streams(id),
    t           REAL,
    text        TEXT NOT NULL,
    kind        TEXT,                -- 'promise','question','unsolved'
    status      TEXT DEFAULT 'open', -- 'open','resolved','dead'
    resolved_stream_id TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- INSTRUMENTATION
-- ============================================================
CREATE TABLE tool_metrics (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id  TEXT,
    metric     TEXT NOT NULL,
    value      REAL,
    meta       TEXT,
    recorded_at TEXT DEFAULT (datetime('now'))
);
```

---

# 4. LAYER 1: CAPTURE

## 4.1 The synchronization problem — and its solution

**Do not attempt post-hoc alignment of separately recorded signals.** Clock drift and manual offset hunting are a recurring failure mode.

There are exactly two categories of signal:

**Derived signals** — extracted *from the OBS recording file itself* (mic RMS, pitch, speech rate, silence, laughter, OCR). These are inherently frame-accurate. **No sync work required, ever.**

**Live signals** — captured during the stream by external processes (markers, input activity). These need one anchor.

**The anchor mechanism:**

1. All live-signal loggers write **Unix epoch milliseconds**.
2. At record start, capture the wall-clock epoch ms when OBS began recording. Obtain via OBS WebSocket (`RecordStateChanged` event) or, as a fallback, a script that writes `time.time()*1000` at the moment it triggers the record hotkey.
3. Store as `streams.record_start_epoch_ms`.
4. Convert: `vod_time_s = (event_epoch_ms - record_start_epoch_ms) / 1000.0`

One number per stream. No drift. No manual alignment. This is the entire sync solution.

## 4.2 OBS configuration (mandatory)

**Audio tracks — separate, non-negotiable:**

| Track | Source |
|---|---|
| 1 | Mixed (for the stream/VOD itself) |
| 2 | Mic only |
| 3 | Game audio only |
| 4 | Discord / party audio only |

**Rationale:** mic RMS analysis is garbage if game audio is mixed in — an explosion registers as a mic spike. Party audio as its own track is what enables (a) independent laughter detection from other people and (b) deterministic speaker identification for colored captions. This is unrecoverable after the fact.

Recording format: **MKV** (crash-safe; remux to MP4 after) or fragmented MP4. Never plain MP4 for long recordings.

**Also configure:**
- Scene switch logging (OBS log file already contains this; parse it)
- Hotkeys bound to the marker daemon (see below)

## 4.3 Marker daemon

A small always-running script. Two hotkeys:

| Key | Meaning | Weight |
|---|---|---|
| F1 | "maybe" — something might have happened | Moderate |
| F2 | "definitely" — that was good | High |

Writes JSONL, one line per press:
```json
{"epoch_ms": 1755123456789, "kind": "marker_maybe"}
{"epoch_ms": 1755123499123, "kind": "marker_definite"}
```

**CRITICAL — retroactive offset.** A marker is pressed *after* the operator registers that something happened. Realization plus reaction is 5–15 seconds. **Default the marker's window anchor to `t − 20s`** and expand outward from there. Make this offset a configurable parameter (`marker_retro_offset_s`, default 20.0).

## 4.4 Input activity logger

Low-level keyboard/mouse hook (`pynput` on Windows). Writes JSONL at 10 Hz aggregate rather than per-event to keep files small:

```json
{"epoch_ms": 1755123456700, "keys_per_s": 4.2, "mouse_vel_px_s": 1840.5, "clicks_per_s": 2.0}
```

Derived downstream: **mouse velocity spikes** (flicks), **input rate spikes** (panic), **sudden stillness** (shock/death).

This signal is cheap to collect, almost nobody uses it, and it is arguably the second-strongest gameplay signal after markers.

## 4.5 Failure isolation

Each capture process runs independently. If the input logger crashes, the stream continues and one signal is lost. Capture processes must **never** be able to interrupt OBS.

---

# 5. LAYER 2: EXTRACTION

## 5.1 Pipeline stages (all idempotent)

Each stage checks `pipeline_stages` for `status='done'` and skips. Order:

```
1.  register_stream      → streams row, anchor, file paths
2.  probe                → ffprobe: duration, fps, resolution, track map
3.  proxy                → generate proxy
4.  audio_split          → extract per-track WAV
5.  audio_features       → RMS, F0, silence, laughter (per track)
6.  whisperx             → transcript + word timestamps
7.  speaker_assign       → deterministic, via track energy
8.  phrase_detect        → passive voice triggers, repeated phrases
9.  input_signals        → parse input JSONL → signal_series
10. marker_events        → parse marker JSONL → events
11. scene_events         → parse OBS log → events
12. vision               → OCR/template match (DEFERRED — see §15)
13. embeddings           → per-segment vectors
14. score                → candidates (cheap, rerunnable)
15. previews             → 2s webm, thumbstrip, waveform per candidate
16. digest               → LLM digest generation
```

Stages 1–13 are **extraction** (expensive, once). Stage 14 is **scoring** (cheap, infinitely rerunnable). This boundary is the single most important architectural decision in the system.

## 5.2 Proxy generation

```bash
ffmpeg -i MASTER.mkv \
  -map 0:v:0 -map 0:a:0 \
  -c:v libx264 -preset veryfast -b:v 2M \
  -vf scale=-2:720 \
  -g 30 -keyint_min 30 -sc_threshold 0 \
  -c:a aac -b:a 128k \
  -movflags +faststart \
  PROXY.mp4
```

**`-g 30` with `-sc_threshold 0` forces a keyframe every 30 frames** (0.5s at 60fps, 1s at 30fps). This is what makes scrubbing instantaneous. Fixed GOP is essential — without `-sc_threshold 0`, x264 inserts keyframes on scene changes and spacing becomes unpredictable.

The proxy has an **identical timebase** to the master, so all timestamps are interchangeable between them.

Size: ~4 GB per 4-hour stream.

## 5.3 Audio track extraction

```bash
ffmpeg -i MASTER.mkv \
  -map 0:a:1 -ac 1 -ar 16000 -c:a pcm_s16le audio/mic.wav \
  -map 0:a:2 -ac 1 -ar 16000 -c:a pcm_s16le audio/game.wav \
  -map 0:a:3 -ac 1 -ar 16000 -c:a pcm_s16le audio/party.wav
```

16 kHz mono is correct for both WhisperX and all audio feature analysis. Do not extract at higher rates.

## 5.4 Signal catalog — complete

All continuous signals are computed at **10 Hz** (100 ms hop) then optionally downsampled to 1 Hz for scoring. Store at 10 Hz; downsample at score time.

### 5.4.1 Approved continuous signals

| `kind` | Source track | Method | Notes |
|---|---|---|---|
| `mic_rms` | mic | `librosa.feature.rms`, hop 1600 | Convert to dB. Use **delta vs. rolling baseline**, not absolute. |
| `mic_f0` | mic | `librosa.pyin`, fmin=65, fmax=400 | Pitch. A pitch spike outperforms a volume spike as an excitement marker. |
| `mic_f0_variance` | derived | rolling std of `mic_f0` over 5 s | High prosodic range = comedic delivery. Distinct from peak pitch. |
| `speech_rate` | transcript | words/sec in 3 s sliding window | Detects both excitement and dead air. |
| `sudden_silence` | mic | RMS falls below floor within 2 s of high activity | Death, shock, concentration. Inverse of intuition — a strong signal. |
| `game_rms` | game | as `mic_rms` | Context only. Never a highlight signal alone. |
| `party_rms` | party | as `mic_rms` | Girlfriend/party reaction. |
| `party_f0` | party | as `mic_f0` | |
| `overlap_speech` | mic+party | VAD on both tracks, boolean AND | Everyone talking at once = something happened. |
| `input_rate` | input log | keys+clicks per second | |
| `mouse_velocity` | input log | px/s | Flick detection input. |
| `input_stillness` | derived | inverse of `input_rate`, gated | Sudden stillness after activity. |

### 5.4.2 Approved discrete events

| `kind` | Detection | Notes |
|---|---|---|
| `marker_maybe` | Hotkey F1 | Apply `-20s` retro offset |
| `marker_definite` | Hotkey F2 | Apply `-20s` retro offset. Highest weight in system. |
| `laugh_operator` | Acoustic classifier on mic | See §5.5 |
| `laugh_party` | Acoustic classifier on party | **Independent validation** — someone else laughing means the joke landed |
| `phrase_excitement` | Transcript pattern match | See §5.6 |
| `phrase_repeat` | Same phrase ≥3× in 90 s window | Bit formation in real time |
| `kill` | OCR/template (deferred) | |
| `multikill` | Temporal cluster: ≥3 kills in 8 s | Team wipe |
| `clutch` | Low HP + survived + kills | Strongest single gameplay signal |
| `mvp_screen` | Template match end-of-match MVP/SVP sequence | Marvel Rivals has MVP/SVP, **not** POTG (that is Overwatch) |
| `ult_used` | Template match | **Not an independent scorer.** Correlation component only. |
| `headshot` | Template match hit indicator | **Not an independent scorer.** Flick signature component only. |
| `scene_change` | OBS log | Structural boundary only, weight ~0 |
| `menu_screen` | Template match | Negative signal input, gated (§6.4) |
| `afk` | Derived: no input AND no speech >60 s | Negative signal, gated |

### 5.4.3 Composite / correlation signals

These are computed at score time from the above. They are the highest-value detections in the system.

**`flick_signature`** — the defining clip signature for a projectile character like Hawkeye:
```
mouse_velocity spike (> p95 of rolling baseline)
  AND headshot event within 500 ms
  AND kill event within 500 ms
→ emit composite event, high gameplay weight
```

**`multikill_with_ult`** — `ult_used` within 3 s preceding a `multikill` scores higher than a bare multikill.

**`reaction_onset`** — `sudden_silence` followed within 2 s by `overlap_speech` + `party_rms` spike. Classic "oh my god" moment.

## 5.5 Laughter detection

Whisper transcribes laughter poorly, but the acoustic signature is distinctive: rapid periodic bursts, 4–7 Hz envelope modulation, characteristic spectral shape.

**Approach (in order of preference):**
1. Envelope-periodicity heuristic: band-pass the RMS envelope at 4–7 Hz, threshold the energy. Cheap, no model, works surprisingly well.
2. If insufficient, a small pretrained audio event classifier (YAMNet or similar) has a laughter class.

Run **independently on the mic track and the party track.** Party laughter is the more valuable of the two because it is external validation.

## 5.6 Passive voice triggers

**Note:** the operator explicitly rejected an *active* voice trigger (deliberately saying a codeword). This is the passive version — pattern matching on natural speech that already occurs. Zero behavior change.

Pattern list (configurable, in `config/phrases.yaml`):
```
"oh my god", "no way", "what the fuck", "did you see that",
"are you kidding", "holy shit", "let's go", "oh my days",
"what just happened", "i can't believe"
```

Plus **swearing density**: profanity count per 10 s window as a continuous signal.

Match against the transcript, emit `phrase_excitement` events with the matched phrase in `meta`.

## 5.7 WhisperX configuration

```python
model = whisperx.load_model(
    "large-v3",
    device="cuda",
    compute_type="float16",
    vad_options={"vad_onset": 0.5, "vad_offset": 0.363},
    asr_options={
        "initial_prompt": VOCABULARY_PROMPT,   # see below
        "hotwords": VOCABULARY_LIST,
    },
)
```

**VAD filtering must be enabled.** Without it, Whisper hallucinates phantom text over silence and music — a well-documented failure mode that would poison both the transcript and every downstream signal.

**Vocabulary seeding (important, low effort, high impact).** Maintain `config/vocabulary.txt` containing:
- Marvel Rivals hero names (Hawkeye, Iron Fist, Jeff, Luna Snow, Mantis, Namor, ...)
- Game-specific slang (ult, ulted, dive, peel, flick, one-shot, headshot, wallbang)
- Valorant agent names and callouts
- The girlfriend's name and any recurring nicknames
- Recurring bit terminology (append as bits emerge)

Without this the transcript says "Hawk I" and "Iron First" permanently, and bad captions read as amateur immediately.

**Alignment:** run `whisperx.align()` for word-level timestamps. These drive captions, cut-point snapping, and speech-rate calculation.

**Do NOT use pyannote diarization.** See next section.

## 5.8 Speaker assignment — deterministic, no model

Because mic and party are separate tracks, speaker identity is simply *which track has energy*:

```python
def assign_speaker(seg_start, seg_end, mic_rms, party_rms):
    mic_e   = mean_energy(mic_rms,   seg_start, seg_end)
    party_e = mean_energy(party_rms, seg_start, seg_end)
    if mic_e > party_e * 1.5:   return "operator"
    if party_e > mic_e * 1.5:   return "party"
    return "both"               # overlap
```

This is more reliable than pyannote diarization and requires no Hugging Face token, no license acceptance, and no model. It also directly enables speaker-colored captions (§8.3).

Run WhisperX **separately on the mic track and the party track**, then merge segments by timestamp. This gives clean per-speaker transcripts with no cross-talk contamination.

## 5.9 Vision / OCR (DEFERRED — build last)

**Honest assessment:** this is the piece most likely to defeat the build. Template-matching a kill feed against a moving background, at varying resolutions, with UI that changes every game patch, is real computer-vision work with ongoing maintenance cost.

It is also **the signal most affordable to lose** — audio signals already fire during multikills because the operator reacts to them.

**Build this last. If it proves difficult, cut it.**

When built:
- Sample at **2–4 fps**, not 60. OCR dominates processing time otherwise.
- Use OpenCV `matchTemplate` against cropped UI regions, not full-frame OCR.
- Store templates in `config/templates/<game>/` with a defined ROI per template.
- Version templates by game patch; expect to re-capture after major updates.

## 5.10 Embeddings

Per transcript segment, generate an embedding via Ollama:

```python
# bge-small-en-v1.5 → 384 dims  (recommended: smaller, faster, sufficient)
# nomic-embed-text  → 768 dims  (higher quality, 2x storage)
```

Storage math at 384 dims: 100 streams × ~2,000 segments × 384 × 4 bytes ≈ **300 MB**.

Brute-force cosine similarity over 200k vectors in numpy is **~50 ms**. **No vector database is required.** `sqlite-vec` is optional convenience only.

---

# 6. LAYER 3: SCORING

## 6.1 Core principle

Scoring is a **pure function** over stored signals. It must be re-runnable across the entire back catalog in seconds. Every candidate row records the `config_version` that produced it.

**Consequence:** the first 20 streams are not wasted on bad weights. Bad weights + good extraction = re-score later at zero cost.

## 6.2 Algorithm

```
FOR each stream:
  1. Load all signal_series and events
  2. Resample everything to a common 1 Hz grid
  3. For each signal: rolling z-score against a 300-second (5 min) window
     centered on the sample
        z[t] = (x[t] - mean(x[t-150 : t+150])) / std(x[t-150 : t+150])
  4. Convert discrete events into impulse signals, then apply a decay kernel
     (Gaussian, sigma configurable per event kind)
  5. Compute composite/correlation signals (§5.4.3)
  6. FOR each profile in {entertainment, gameplay}:
       composite[t] = Σ (weight_i * z_i[t])
       apply gated negative penalties (§6.4)
       smooth with Gaussian, sigma = 2 s
       peak-find (scipy.signal.find_peaks, prominence threshold)
       expand each peak into a window with hysteresis (§6.3)
       apply spacing penalty (§6.6)
  7. Compute score_combined (§6.5)
  8. Merge overlapping windows across profiles; write candidates
```

### Why rolling z-score is mandatory

Mic gain drifts. Games have different loudness. Energy changes over three hours. A global normalization constant means the detector finds "the loud game" rather than "the loud moment." The 5-minute rolling baseline is what makes signals comparable within a session.

## 6.3 Window generation and hysteresis

Rank **windows**, not seconds. Without hysteresis you get hundreds of one-second fragments.

```
peak found at t_peak with value v_peak
enter_threshold = 0.6 * v_peak
exit_threshold  = 0.35 * v_peak

expand left  from t_peak while composite > exit_threshold
expand right from t_peak while composite > exit_threshold
clamp to [min_window_s=8, max_window_s=60]
snap t_start / t_end to nearest word boundary from WhisperX
```

**Word-boundary snapping** prevents clipped syllables and is free given word-level timestamps.

Default: `min_window_s = 8`, `max_window_s = 60`.

## 6.4 Negative signals — GATED, not additive

**This is a specific requirement and must be implemented exactly as described.**

Negative signals must never fire when audio indicates something is happening. Audio is ground truth for "is anything going on."

**Menu/lobby penalty:**
```
IF menu_screen_active
   AND no_speech_detected_for >= 8 seconds
   AND audio_energy < rolling_baseline
THEN apply penalty, ramping in gradually (not a step function)

ANY speech detected → reset the 8-second timer to zero, remove penalty
```

Requirement: jokes in the lobby must never be penalized. The 8-second grace period and the speech-reset are both mandatory.

**AFK penalty:**
```
IF no_input_activity AND no_speech
   FOR >= 60 continuous seconds
THEN apply penalty
```
Both conditions required. Either one alone is insufficient.

**Other negatives (same gating principle):** loading screens, flat-prosody monologue.

**Generalized rule:** all negative signals are conditional on audio being flat. Implement as a shared gating helper, not per-signal ad hoc logic.

## 6.5 Dual profiles plus the combined score

Two weight profiles, defined in `config/profiles/`:

**`entertainment.yaml`** — the primary profile. The operator's channel identity is comedy.
```yaml
weights:
  marker_definite:     3.0
  marker_maybe:        1.5
  laugh_party:         2.5      # external validation, weighted above own laugh
  laugh_operator:      2.0
  mic_f0_variance:     1.8      # prosodic range = comedic delivery
  overlap_speech:      1.5
  phrase_repeat:       1.5
  party_rms:           1.2
  mic_rms:             1.0
  phrase_excitement:   1.0
  speech_rate:         0.8
  sudden_silence:      0.8
  reaction_onset:      1.5
  kill:                0.1
  multikill:           0.3
  input_rate:          0.2
```

**`gameplay.yaml`** — secondary. Supporting material, occasional edit-style compilations.
```yaml
weights:
  marker_definite:     3.0
  flick_signature:     3.0
  clutch:              2.8
  multikill:           2.5
  multikill_with_ult:  2.8
  mvp_screen:          2.0
  mouse_velocity:      1.5
  input_rate:          1.2
  kill:                1.0
  mic_rms:             0.8
  laugh_operator:      0.2
  laugh_party:         0.2
```

### The combined score — the most valuable output

**`score_combined` is NOT a sum or average.** It must reward windows where *both* profiles fire, because that intersection is the operator's actual differentiator: a flashy Hawkeye play *plus* a genuinely funny reaction. That combination requires mechanics and personality simultaneously, which is not commoditized.

Use a product-like form that penalizes imbalance:

```python
def combined(e, g, alpha=0.5):
    # geometric-mean-like: both must be non-trivial
    e_n, g_n = normalize(e), normalize(g)
    return (e_n ** alpha) * (g_n ** (1 - alpha)) * (e_n + g_n)
```

Any formulation is acceptable provided it satisfies: **high+high >> high+low.**

**Surface combined-score winners FIRST in the review UI**, in their own section, above the per-profile lists.

### Profile selection

- Marvel Rivals streams: run **both** profiles plus combined.
- Casual/comedy streams (Valorant QP, Roblox, co-op): entertainment only; gameplay weights are near-meaningless.
- Drinking streams: use entertainment profile with **thresholds lowered further** and automatic-signal weights increased relative to markers — marker discipline degrades exactly when the most clippable material is being generated.

Store the profile used in `streams.profile_used` and `candidates.config_version`.

## 6.6 Spacing penalty

Prevents ten candidates from one 90-second stretch.

```
FOR each candidate, sorted by score descending:
    FOR each lower-scoring candidate within 30 s:
        multiply its score by a decay factor (e.g. 0.5)
```

The operator can always lengthen a clip during review, so aggressive spacing is safe.

## 6.7 Target output volume

**80–150 candidates per 3-hour stream.** Tune the peak-prominence threshold to hit this range. This follows directly from constraint C2 — recall over precision, with cheap review.

---

# 7. LAYER 4: REVIEW UI

## 7.1 Why this section matters most

Every other subsystem feeds this one screen. If review is slow, the tool goes unused. This is the part engineers reliably under-build because signal processing is interesting and building a fast list view is not.

**Hard target: 120 candidates in under 8 minutes.** That is ~4 seconds per candidate. If exceeded, fix the UI before adding any feature anywhere in the system.

## 7.2 Precomputed preview assets

**Every expensive operation happens during unattended batch processing. Nothing expensive happens while the operator is waiting.**

Generated in the `previews` stage for every candidate:

| Asset | Spec | Size |
|---|---|---|
| Preview clip | 2 s, centered on `t_peak`, 480p, VP9/webm, no audio normalization | ~150 KB |
| Thumb strip | 5 frames evenly spaced across window, JPEG, 160px wide | ~30 KB |
| Waveform PNG | Mic + party RMS over the window | ~10 KB |

Total: **~25 MB per stream for 120 candidates.** Negligible.

Generation command pattern:
```bash
ffmpeg -ss {t_peak-1} -i {proxy} -t 2 \
  -vf scale=480:-2 -c:v libvpx-vp9 -crf 40 -b:v 0 -an \
  previews/{candidate_id}.webm
```

**Note `-ss` before `-i`.** This is input seeking — ffmpeg uses the container index to jump directly to the timestamp, taking milliseconds regardless of file size. Placing `-ss` after `-i` causes decode-from-frame-zero and is orders of magnitude slower. **Hardcode the fast form; this must never be gotten wrong.**

## 7.3 Interaction spec

Keyboard-driven. The operator should never need the mouse.

| Key | Action |
|---|---|
| `j` / `↓` | Next candidate |
| `k` / `↑` | Previous candidate |
| `1` | Rate 0 — skip |
| `2` | Rate 1 — maybe |
| `3` | Rate 2 — clip it |
| `space` | Expand: play full window with audio |
| `[` / `]` | Nudge window start earlier / later (0.5 s) |
| `{` / `}` | Nudge window end earlier / later (0.5 s) |
| `t` | Add tag (autocomplete from existing tags) |
| `n` | Add note |
| `e` | Send to export queue |
| `?` | Show contributing signals breakdown |

**Behavior:**
- Preview autoplays on focus, loops silently.
- Rating advances automatically to the next candidate.
- Transcript text for the window displayed alongside.
- Contributing-signals breakdown available but collapsed by default.

## 7.4 Layout and ordering

Sections, in order:
1. **Combined-score winners** (both profiles high) — the best content, reviewed first
2. **Entertainment** ranked
3. **Gameplay** ranked
4. **Marker-anchored** candidates that did not rank highly (safety net — the operator marked these deliberately)

## 7.5 Instrumentation hooks

Record `review_ms` per candidate and total session duration into `tool_metrics`. This is how the C4 target is verified rather than assumed.

---

# 8. LAYER 5: AUTO-FINISH RENDERER

## 8.1 Philosophy

**Auto-finish every approved clip to postable quality. Then selectively upgrade.**

Workflow:
1. Auto-finish all approved clips, unattended
2. Operator watches the batch (~2 min for 5 clips)
3. Posts the ones that are fine as-is — most will be
4. Takes the single best one into CapCut for zoom punches, SFX, timing polish

**Rationale:** most clips do not deserve 30 minutes of attention. Spending equal effort on all five is how the pipeline dies. Auto-finish sets the floor; manual effort is spent only where it pays.

## 8.2 What automates (and beats manual)

| Task | Method | Quality vs. manual |
|---|---|---|
| Captions | WhisperX word timestamps → ASS → burn-in | **Better** — frame-accurate sync |
| Vertical reframe | Static crop template per OBS scene | **Better** — no tracking jitter |
| Loudness normalization | ffmpeg `loudnorm` to −14 LUFS | **Better** — avoids platform's worse normalization |
| Cut-point snapping | Snap to word boundaries | **Better** — no clipped syllables |
| Filler-word removal | Word timestamps + concat filter | Equal, far faster |
| Export presets | Per-platform encode settings | Equal |
| Profanity muting | Word timestamps → mute ranges | Equal — **see §8.6** |

## 8.3 Captions — full specification

**Format: ASS (Advanced SubStation Alpha).** Chosen because it supports inline color/style overrides mid-line, which is required for word-level highlighting and speaker coloring.

**Word-group display, not sentence display.** 3–5 words on screen at a time, with the currently-spoken word highlighted. This is the format that reads at short-form pacing.

**Speaker coloring — deterministic.** Because speaker assignment comes from track energy (§5.8), not a model, each speaker gets a fixed color:
```
operator → color A (e.g. white with dark outline)
party    → color B (e.g. light cyan)
both     → alternate per word by source track
```

This is a genuine differentiator — almost nobody does per-speaker caption coloring on two-person content, and the operator's multi-track setup makes it trivial.

**ASS inline override syntax:** `{\c&HBBGGRR&}` (note: BGR order, not RGB). Highlight the active word by wrapping it in a color override and restoring after.

**Generation:**
```python
for group in chunk_words(words, size=4):
    for i, active_word in enumerate(group):
        # one dialogue line per active-word state
        start, end = active_word.start, active_word.end
        text = "".join(
            f"{{\\c&H{HIGHLIGHT}&}}{w.text}{{\\c&H{BASE}&}} " if w is active_word
            else f"{w.text} "
            for w in group
        )
        emit_dialogue(start, end, text, style=speaker_style(active_word.speaker))
```

**Burn-in:**
```bash
ffmpeg -ss {start} -i {master} -t {duration} \
  -vf "crop=...,scale=1080:1920,ass=captions.ass" \
  -af "loudnorm=I=-14:TP=-1.5:LRA=11" \
  -c:v libx264 -crf 18 -preset slow \
  -c:a aac -b:a 192k \
  exports/{id}_shorts.mp4
```

## 8.4 Vertical reframing — static templates

**Gaming has a structural advantage here.** Commercial tools perform dynamic subject tracking because talking-head footage moves. The operator's layout does not move — facecam is always in the same corner, gameplay fills the same rectangle.

Therefore: **a static crop template per OBS scene.** No model, no tracking, deterministic, instant, identical every time.

`config/crop_templates.yaml`:
```yaml
marvel_rivals_facecam:
  source_resolution: [1920, 1080]
  output: [1080, 1920]
  regions:
    - name: facecam
      src: [1560, 0, 360, 270]      # x,y,w,h
      dst: [0, 0, 1080, 810]
    - name: gameplay
      src: [480, 140, 960, 800]
      dst: [0, 810, 1080, 1110]
```

Implemented as an ffmpeg `crop` + `scale` + `vstack` filter chain. Define one template per OBS scene layout, once, and it is done permanently.

## 8.5 Hook text generation

The hook — first 1–2 seconds plus on-screen text — is the single highest-leverage decision in short-form. It stays manual, **but generating candidate options is a genuine time save.**

Have the LLM propose 5 hook variants from the clip's transcript. The operator picks and rewrites. Store the chosen text in `exports.hook_text`.

## 8.6 Profanity muting

**Toggle. OFF by default.** (Explicit operator decision.)

Store mute ranges as data (derived from word timestamps + a profanity list); apply only at export time when the toggle is on. When needed, produce a dual export — clean version for platforms that require it, unmuted version elsewhere.

## 8.7 What stays manual, permanently

- **Which clips to post** — always the operator's call
- **The hook** — text and first-frame choice
- **Comedic timing** — a half-second beat before a punchline *is* the joke; models cannot hear this
- **Zoom punches, SFX, music**
- **Emphasis placement**

---

# 9. LAYER 6: DIGEST GENERATION

## 9.1 Purpose

Clip detection wants **peaks**. Video ideation wants **the shape of the whole session**. Feeding a raw 3-hour transcript to an LLM and asking for video ideas produces slop.

The digest is the intermediate artifact that bridges them: **3 hours → ~3,000 structured words.**

**The digest corpus is the system's compounding asset.** ~3k words per stream means a year of streaming is ~300k words — trivial to store, trivial to search, and month-12 ideation over 100 digests is a fundamentally different capability than month-1 ideation over four. Digests are **first-class rows, never regenerable cache.** Keep every version forever.

## 9.2 Digest structure

```json
{
  "stream_id": "...",
  "date": "...",
  "games": [...],
  "duration_s": 14400,
  "chapters": [
    {
      "index": 0,
      "t_start": 0, "t_end": 1820,
      "title": "...",
      "summary": "3-4 sentences",
      "game": "Marvel Rivals",
      "energy_mean": 0.62,
      "notable_segment_ids": [12, 45, 88]
    }
  ],
  "recurring_phrases": [
    {"phrase": "...", "count": 7, "segment_ids": [...]}
  ],
  "emotional_arc": [
    {"t_bin": 0, "energy": 0.4, "laughter_density": 0.1}
  ],
  "open_loops": [
    {"text": "...", "t": 4210, "kind": "promise", "segment_id": 331}
  ],
  "top_candidates": [
    {"candidate_id": 88, "score": 4.2, "label": "...", "quote": "..."}
  ],
  "themes_observed": ["...", "..."]
}
```

## 9.3 Chapter segmentation — deterministic first

Segment boundaries are found **without** an LLM:
1. Transcript embedding shift — cosine distance between consecutive rolling-window embeddings; peaks are topic boundaries
2. Game changes (from OCR or manual stream metadata)
3. Long silence gaps (> 60 s)
4. Scene changes (weak signal, tie-breaker only)

Merge boundaries within 120 s. Target chapter length: 10–30 minutes.

## 9.4 Digest generation — map-reduce

**This is the industry-standard pattern and there is no need to invent anything.**

**Map:** for each chapter independently, send the chapter transcript (with segment IDs) to the LLM. Request a 3–4 sentence summary, notable segment IDs, observed themes, and open loops. Each chapter fits comfortably in context.

**Reduce:** combine the chapter outputs plus deterministically-computed statistics (energy arc, n-grams, candidate list) into the final digest JSON.

**Chunking happens ONLY here.** All downstream reasoning operates on the ~3k-word digest, which fits in a single context window with no chunking, no rolling refresh, and no context management.

**Structured extraction first, reasoning second.** This is the pattern for the entire system.

## 9.5 Open-loop extraction

Not a separate subsystem — it is **a field in the digest prompt**: "list things the speaker said he would try, questions he asked aloud, and problems he encountered but did not solve."

Costs one line of prompt. Include it because it is free. Write results to the `open_loops` table; mark resolved when a later stream addresses them.

---

# 10. LAYER 7: YOUTUBE PIPELINES

## 10.1 One pipeline with an optional step

```
Stream digest(s)
        ↓
   Theme provided by operator?
   ├── NO  → Theme suggestion pass → 3–5 options → operator picks or edits
   └── YES ↓
Theme + digest → segment/candidate retrieval from DB
        ↓
Ordering / narrative pass
        ↓
Coverage report (which beats have footage, which are empty)
        ↓
FCPXML/EDL with gaps marked
```

## 10.2 THE critical framing

**The theme is a retrieval query, not a generation prompt.**

Given "Drinking with strangers on Valorant," the model must not imagine a video and then hunt for footage. It **selects and orders existing segments**, returns segment/candidate IDs, and quotes the transcript line justifying each pick.

If the model writes a treatment first and finds footage second, the output is slop.

## 10.3 The "imagine a video" mode — three passes

The operator explicitly wants the model to propose structure, not merely retrieve. This is achievable — the failure mode is not imagination, it is **imagining footage that does not exist.** Separating those makes it work.

**Pass 1 — IMAGINE (input: digest only).**
Model proposes a concept: premise, working title, thumbnail idea, and a beat structure (ordered list of what each beat must accomplish). It invents *structure*, constrained to material the digest proves exists.

**Pass 2 — GROUND (input: beat structure + candidate/segment index).**
For each beat, retrieve real candidates by ID. Model returns IDs only, with a justifying quote per selection.

**Pass 3 — REPORT COVERAGE (deterministic).**
```
Beat 1 (cold open):     3 candidates — STRONG
Beat 2 (setup):         2 candidates — OK
Beat 3 (escalation):    0 candidates — GAP
Beat 4 (payoff):        1 candidate  — THIN
Coverage: 65%
```

**The gaps are the feature, not the bug.** A 70%-coverage concept becomes a **shooting list**: "you need a cold open and a payoff for this — get them next stream." This is the model doing something the operator genuinely cannot do manually, and it converts imagination's weakness into forward planning.

**Always run plain retrieval alongside the imagined version.** Two candidates, operator picks. If the imagined structure is weak, the retrieval list is still useful.

## 10.4 Two video types

**Type A — Per-stream themed video.** Common format among creators. Harder, more likely to disappoint. Single-session themes are usually thin ("we got drunk and played Valorant" is a description, not a premise). Build it, but expect a rough assembly requiring heavy rewriting, not a postable cut.

**Type B — Cross-stream compilation.** Rare among creators, and **this one will work early.** It is ~90% SQL:

```sql
SELECT c.* FROM candidates c
JOIN ratings r ON r.candidate_id = c.id
JOIN streams s ON s.id = c.stream_id
WHERE json_extract(s.games,'$[0]') = 'Marvel Rivals'
  AND EXISTS (SELECT 1 FROM events e
              WHERE e.stream_id=c.stream_id
                AND e.t BETWEEN c.t_start AND c.t_end
                AND e.kind IN ('multikill','flick_signature','clutch'))
  AND r.rating >= 2
ORDER BY c.score_gameplay DESC
LIMIT 40;
```

The LLM contributes ordering and pacing only. Examples: a Hawkeye kill compilation spanning three months; a "bit" compilation once a running gag has accumulated.

**Cross-stream ideation is where footage-first idea generation earns its keep**, because it surfaces patterns the operator cannot see: "you've died to this boss 14 times across 4 sessions"; "you've said you want to try X in 5 separate streams"; "this bit has recurred 9 times over three weeks."

## 10.5 Output format — EDL/FCPXML, never cut files

For YouTube assembly, **do not extract clip files.**

Generate an FCPXML or EDL referencing master files with in/out points. Resolve and Premiere handle multi-source timelines natively — a cross-stream compilation is simply an EDL pointing at 15 different source files.

Advantages: no re-encoding, no keyframe-boundary artifacts, frame-accurate cuts, and a 40-clip compilation "renders" in milliseconds because it is a text file.

**Mark gaps explicitly in the timeline** — placeholder segments where the operator will record commentary or insert a transition.

Only extract standalone files for short-form, where a discrete upload artifact is genuinely required.

---

# 11. LAYER 8: TREND DETECTION AND IDEAS DATABASE

## 11.1 Cross-stream embedding clustering

**The mechanism that finds unknowns.** This is the only subsystem that surfaces content patterns the operator did not name, tag, or notice.

```
1. Collect all candidates with rating >= 1, across all streams
2. Represent each by the mean embedding of its overlapping transcript segments
3. HDBSCAN(min_cluster_size=5, metric='euclidean')
4. For each cluster: compute n_members, n_streams spanned, first/last seen
5. Send representative quotes from each cluster to the LLM for labeling
6. Write to clusters / cluster_members
```

**The key metric is `n_streams`, not `n_members`.** Fifteen members from one stream is a single long moment. Fifteen members across nine streams is a **recurring content type** — an unnamed bit.

**Timing honesty:** clustering over 15 streams finds noise. It needs **60+ streams** to find anything real. Build the logging now (it cannot be reconstructed); build the clustering pipeline when the corpus exists.

## 11.2 Recurring n-gram tracking (bit detection)

Much simpler than clustering — roughly 50 lines.

```
1. Extract 2–6 grams from all transcript segments
2. Filter stopwords and the operator's baseline verbal tics
   (compute baseline from the first ~10 streams; store as is_baseline_tic)
3. Count total occurrences and streams-spanned per phrase
4. Recency-weight:
      recency_score = Σ over occurrences of exp(-days_ago / 30)
5. Flag as EMERGING: absent from streams 1–20, present in >= 8 of last 15
```

**Recency weighting is what separates a catchphrase from a verbal tic.** A tic is uniformly distributed across all streams. A bit appears, grows, and recurs.

**On the value of bits:** running gags are the primary mechanism by which comedy audiences form attachment — the difference between "that was funny" and "that's *our* thing." But a bit has no callback value until the audience has seen the setup. At zero audience in month one, this produces nothing usable. Value arrives around **month 4–6, and only if logging started on day one.**

## 11.3 Open-loop tracking

Already covered — a field in the digest prompt (§9.5), not a subsystem. Unresolved loops are video premises with built-in setup.

## 11.4 Scheduled ideation pass

**This is not a detection mechanism. It is the cron job that runs §11.1–11.3 and updates idea evidence.** Without it, cluster output sits in a table nobody reads.

Monthly:
```
1. Run clustering; label clusters
2. Update n-gram recency scores; flag emerging phrases
3. Load last ~30 digests + cluster labels + open loops
   (30 × 3k words ≈ 120k tokens — fits one context, costs well under $1)
4. LLM pass: propose new ideas, re-score existing open ideas
5. For every open idea: re-run retrieval, update idea_evidence,
   recompute coverage_pct and gaps
6. Update ideas.status where coverage crosses threshold → 'ready'
```

## 11.5 Ideas as living records

**This is the mechanism that makes months of accumulated data pay off.**

Ideas persist and accumulate evidence:
- Month 2: "The Coward Bit" proposed — 3 supporting candidates, weak
- Month 3: 7 candidates, one strong cold open
- Month 5: 14 candidates, coverage complete → **status: ready**

An idea's status changes because the footage caught up, not because the operator remembered it.

**Dashboard:** ideas sorted by `coverage_pct` descending. Top items are shootable now. Items with a single specific gap become the shooting list for the next stream.

## 11.6 Semantic search — two entry points, one index

**Push (automatic):** clustering surfaces patterns on a schedule. *"14 moments across 9 streams cluster together; label: 'panicking and blaming your girlfriend.'"* Never searched for. The operator did not know it existed.

**Pull (manual):** free-text query — *"that time I did the sad voice"* — embedded and matched by cosine similarity across all segments, 5 months deep. Keyword search cannot do this; the operator remembers the vibe, not the words.

Same 300 MB index serves both. Push is the capability that cannot be replicated manually; pull is the one used daily.

---

# 12. LLM USAGE RULES

**These rules apply to every LLM call in the system without exception.**

## 12.1 The model never emits timestamps

Assign sequential integer IDs to transcript segments and candidates at ingest. Provide `{id, text}` pairs to the model. **The model returns IDs only.** All timestamps are resolved from the database.

## 12.2 Validation is mandatory

Every returned ID is checked for existence. Non-existent IDs are **silently dropped**, and the drop is logged to `tool_metrics` for monitoring hallucination rate.

## 12.3 Required output discipline

- **Constrained JSON schema** on every call (structured outputs / function calling where available)
- **Every selection must include a verbatim quote** from the referenced segment. Fabrication becomes immediately visible and is machine-checkable against the stored text.
- The model **judges and labels**; deterministic signals **find**. The LLM is never the search mechanism.

## 12.4 Context strategy

- Send only transcript around candidate windows (±30 s), never a full transcript, for clip ranking. ~95% token reduction; hallucination surface reduced to near zero.
- Chunking happens **only** in digest generation, via map-reduce over chapters.
- All downstream reasoning runs over the ~3k-word digest in a single context. **No rolling refresh, no context management, no chunking.**

**Cost reality (do not optimize for this):**

| Call | Approx tokens |
|---|---|
| Digest (map+reduce) | ~30k input |
| Theme suggestion | ~4k |
| Assembly | ~5k |
| **Total per stream** | **< 50k** |

Roughly **$0.10–0.30 per stream** on a frontier model. Cost is not a design constraint. Run transcription and embeddings locally because they are GPU-bound and the hardware exists; run reasoning through an API because quality matters and price is noise.

---

# 13. STORAGE, BACKUP, AND RETENTION

## 13.1 Storage tiers

| Tier | Contents | Size / stream | Retention |
|---|---|---|---|
| Master | Full-quality OBS recording | 40–55 GB | Keep all of year one |
| Proxy | 2 Mbps H.264, same timebase | ~4 GB | Forever |
| Previews | webm/jpg/png per candidate | ~25 MB | Forever |
| Metadata | DB rows, embeddings, digest | ~5 MB | Forever |

**On masters:** 100 streams ≈ 5 TB; an 8 TB external drive is ~$150. Do not build a retention policy that saves less money than a single stream's electricity. Revisit in year two.

**Proxies are generated from day one for speed, not storage.** Scrubbing 50 GB masters in the review UI is unusable; proxies scrub instantly and share an identical timebase, so timestamps are fully interchangeable.

## 13.2 Backup plan

**Tier 1 — SQLite database (~5 MB/stream). THE IRREPLACEABLE TIER.**

Five months of ratings, digests, embeddings, and idea history cannot be reconstructed. Signals can be re-extracted from footage; the operator's judgment calls cannot.

Nightly, automated:
```bash
sqlite3 data/clipforge.db "VACUUM INTO 'data/backups/clipforge_$(date +%F).db'"
gzip data/backups/clipforge_$(date +%F).db
# upload to B2/S3
```

`VACUUM INTO` is safe against a live database (SQLite 3.27+) and produces a clean, compacted copy.

**Retention:** 30 daily + 12 monthly.

**Destination:** Backblaze B2 — first 10 GB free, then ~$6/TB/month. DB backups live in the free tier essentially indefinitely. Google Drive (15 GB free) or Dropbox also work for a 5 MB nightly dump; B2 is better suited to scripted automation.

**Tier 2 — digests + ideas as plain markdown.** Exported alongside the DB. Human-readable, survives any schema change or full application rewrite. Cheap insurance against the app itself becoming the failure point.

**Tier 3 — proxies.** Local + external drive. Cloud only if budget allows.

**Tier 4 — masters.** Local + external drive. Not cloud; not worth the cost.

## 13.3 Test the restore path

**An untested backup is not a backup.** Once, early: restore a backup to a scratch directory, point the application at it, confirm it runs. Then trust it.

---

# 14. INSTRUMENTATION

Roughly 20 lines of code, and the only way to know whether any of this is working. Without it, weight tuning is blind.

Log to `tool_metrics`:

| Metric | Purpose |
|---|---|
| `review_session_duration_s` | Verify the C4 target (< 8 min / 120 candidates) |
| `review_ms_per_candidate` | Identify UI slowness |
| `approval_rate` | Approved / total candidates — is threshold correct? |
| `signal_firing_rate_by_rating` | Which signals fired on approved vs. rejected — **the primary weight-tuning input** |
| `marker_precision` | Fraction of marker-anchored candidates approved |
| `marker_recall_proxy` | Fraction of approved candidates that had no marker (i.e. what the operator missed live) |
| `stage_duration_s` | Processing bottleneck identification |
| `llm_invalid_id_rate` | Hallucination monitoring |
| `candidates_per_hour_of_stream` | Threshold calibration |

`marker_recall_proxy` is particularly valuable: it directly measures how many good clips the operator misses live, which is the exact worry that motivated automatic detection in the first place.

---

# 15. BUILD ORDER

**Constraint C5 governs this section: never build ahead of the data.** The operator currently has zero streams. Every weight, threshold, and window length in this document is an educated guess. Some are wrong in ways only real footage will reveal.

## Phase 0 — Capture setup (30 minutes, on streaming PC)
- OBS multi-track audio (mic / game / party separated)
- Marker hotkey daemon (F1/F2) writing JSONL
- Record-start anchor capture
- Input activity logger

**Blocking for data collection, not for development.** Phases 1–3 can be built and tested against any recorded gameplay footage before returning to the streaming PC.

## Phase 1 — Minimum viable pipeline (a weekend)
- Stream registration, ffprobe, proxy generation
- Audio split, mic RMS extraction
- Marker parsing with retroactive offset
- Naive scoring (markers + RMS only)
- Window generation with hysteresis
- **FCPXML export**
- Minimal review UI

**This phase alone probably captures ~70% of the total value the project will ever deliver.** Ship it, use it for ten streams, then decide what to add based on what actually caused friction.

## Phase 2 — Transcript layer
- WhisperX with VAD and vocabulary seeding
- Word-level timestamps, word-boundary snapping
- Deterministic speaker assignment via track energy
- Speech rate, passive phrase detection
- **ASS caption generation with speaker coloring**
- Semantic embeddings + pull search

## Phase 3 — Full signal set
- Pitch (F0) and pitch variance
- Sudden silence, overlap speech
- Laughter detection (mic + party)
- Input-derived signals
- Dual profiles + combined score
- Gated negative signals
- Spacing penalty
- Preview asset generation

## Phase 4 — Auto-finish renderer
- Crop templates and vertical reframe
- Loudness normalization
- Caption burn-in
- Export presets
- Hook text generation
- Filler removal, profanity muting (toggle, off)

## Phase 5 — Digest and ideation
- Deterministic chapter segmentation
- Map-reduce digest generation
- Theme suggestion pass
- Three-pass imagine/ground/report
- Cross-stream compilation queries
- EDL/FCPXML with gap markers

## Phase 6 — Trends (requires 60+ streams)
- N-gram tracking and bit detection (can start earlier — cheap)
- HDBSCAN clustering
- Idea evidence accumulation
- Scheduled ideation cron
- Idea dashboard

## Phase 7 — Vision (last, cuttable)
- Template capture per game
- Kill feed / scoreboard matching
- Multikill clustering, clutch detection, MVP screen
- Flick signature composite

**Defer to last. If it proves difficult, cut it.** Audio signals already fire during multikills because the operator reacts to them.

## The failure mode to actively guard against

The realistic risk is not technical. It is **month three, half the system built, zero videos published, and having become a person who builds video tooling rather than a person who makes videos.**

Mitigation: ship Phase 1, stream ten times, then choose Phase 2+ features based on observed friction. Everything in this document should be **reachable**, not **pre-built**.

---

# 16. EXPLICITLY REJECTED FEATURES

Recorded so they are not re-proposed during implementation.

| Feature | Status | Reason |
|---|---|---|
| Active voice trigger (saying a codeword) | **Rejected** | Operator does not want the behavior change. *Passive* phrase detection is included. |
| POTG screen detection | **Rejected** | Does not exist in Marvel Rivals (that is Overwatch). MVP/SVP sequence is the correct analogue. |
| Scene changes as a scoring signal | **Rejected as a scorer** | Low value. Retained as a chapter-boundary tie-breaker only. |
| Ult usage as an independent scorer | **Rejected as independent** | Retained as a correlation component of `multikill_with_ult`. |
| Headshot markers as an independent scorer | **Rejected as independent** | Retained as a correlation component of `flick_signature`. |
| Profanity muting by default | **Off by default** | Toggle only; dual export if a platform requires it. |
| pyannote diarization | **Rejected** | Separate audio tracks make speaker ID deterministic. No model, no HF token. |
| Chat velocity / emote density | **Deferred, not rejected** | Highest-signal input in general, but unavailable at zero audience. Implement when chat becomes active. |
| Vector database | **Rejected** | Brute-force numpy cosine over 200k vectors is ~50 ms. Unnecessary complexity. |
| VLM scrubbing full video | **Rejected** | Expensive, slow, less reliable than template matching for fixed UI. |
| Training on raw view counts | **Rejected** | Variance swamps signal; confounders (growth, game, posting time) dominate; no counterfactual; self-reinforcing. Operator's own ratings are the training signal. |

---

# 17. OPEN PARAMETERS REQUIRING EMPIRICAL TUNING

Every value below is a starting guess. All must be tuned against real footage. **Store them in config files, never hardcoded.**

| Parameter | Default | Tune against |
|---|---|---|
| `marker_retro_offset_s` | 20.0 | Fraction of marker-anchored windows where the moment is inside the window |
| `rolling_baseline_window_s` | 300 | Whether long-term energy drift leaks into scores |
| `min_window_s` / `max_window_s` | 8 / 60 | How often the operator nudges boundaries during review |
| `hysteresis_enter` / `hysteresis_exit` | 0.6 / 0.35 | Window length distribution |
| `spacing_penalty_window_s` | 30 | Clustered-candidate complaints |
| `spacing_penalty_factor` | 0.5 | Same |
| `peak_prominence_threshold` | tune to 80–150 candidates/3 h | Candidate count |
| `menu_grace_period_s` | 8 | Whether lobby jokes ever get suppressed |
| `afk_threshold_s` | 60 | False AFK penalties |
| All profile weights | See §6.5 | `signal_firing_rate_by_rating` from instrumentation |
| `combined_score_alpha` | 0.5 | Whether combined winners are actually the best clips |
| `laughter_band_hz` | 4–7 | Laughter detection precision/recall |
| `hdbscan_min_cluster_size` | 5 | Cluster coherence at 60+ streams |
| `ngram_recency_halflife_days` | 30 | Emerging-bit detection quality |

**Tuning procedure:** after every ~5 streams, pull `signal_firing_rate_by_rating` from `tool_metrics`, compare firing rates on rating-2 vs rating-0 candidates, and adjust weights toward signals that discriminate. Re-score the entire back catalog (cheap, by design) and confirm improvement.

---

# APPENDIX A: IMPLEMENTATION NOTES AND GOTCHAS

**A1. `-ss` placement in ffmpeg.** Before `-i` = input seeking via container index, milliseconds. After `-i` = decode from frame zero, minutes. Hardcode the fast form everywhere. This is the single most common performance mistake in this domain.

**A2. Fixed GOP for proxies.** `-g 30 -keyint_min 30 -sc_threshold 0`. Without `-sc_threshold 0`, x264 inserts keyframes on scene changes and seek granularity becomes unpredictable.

**A3. ASS color format is BGR, not RGB.** `{\c&HBBGGRR&}`. Getting this backwards produces silently wrong colors.

**A4. WhisperX VAD must be on.** Otherwise phantom transcription over silence and music poisons the transcript and every downstream signal.

**A5. SQLite WAL mode.** Required for concurrent read during batch write (review UI while processing continues).

**A6. Store signal arrays as BLOB, not rows.** Row-per-second-per-signal produces ~8.6M rows over 100 streams and makes queries slow for no benefit.

**A7. MKV for recording, not MP4.** MP4 is unrecoverable if OBS crashes mid-recording. Remux to MP4 afterward if needed.

**A8. Epoch milliseconds everywhere in capture.** Never local time, never formatted strings. Convert to VOD-relative seconds only at ingest.

**A9. Feature vectors are logged in full, always,** even for signals not currently weighted. They cannot be reconstructed retroactively and they are the input to all future tuning.

**A10. Every extraction stage writes to a temp path and atomically renames on success.** This is what makes resumability actually work rather than leaving half-written files that look complete.
