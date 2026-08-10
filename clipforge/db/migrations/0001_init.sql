-- ClipForge initial schema. Spec §3.2, with the deviations noted inline.
--
-- Core principle (§3.1): video files are immutable blobs, everything else is
-- metadata pointing into them. A clip is never a file until export; it is
-- (stream_id, t_start, t_end).
--
-- Tables for phases that are not built yet are created anyway (C6). An empty
-- table costs nothing; retrofitting one costs a migration and a backfill.
--
-- PATHS: every path column except streams.master_path holds a path RELATIVE to
-- paths.data_root. The database is the irreplaceable backup tier (§13.2) and it
-- has to survive moving between machines and drive letters.

-- ============================================================
-- STREAMS
-- ============================================================
CREATE TABLE streams (
    id                    TEXT PRIMARY KEY,      -- e.g. '2026-08-14_marvel'
    date                  TEXT NOT NULL,         -- ISO date
    title                 TEXT,
    games                 TEXT,                  -- JSON array
    profile_used          TEXT,                  -- weight profile name
    master_path           TEXT NOT NULL,         -- absolute; see `clipforge relink`
    proxy_path            TEXT,                  -- relative to data_root

    -- DEVIATION from §3.2, which declares this NOT NULL. Arbitrary footage has
    -- no OBS anchor, and Phase 1 must be testable against arbitrary footage
    -- before the capture scripts exist (§15 Phase 0 is unbuilt). NULL means
    -- "no wall-clock anchor", which is only legal when marker_time_base='vod'.
    record_start_epoch_ms INTEGER,

    -- NOT IN §3.2. 'epoch' = marker JSONL carries Unix ms and needs the anchor
    -- to convert (§4.1). 'vod' = timestamps are already seconds into the file,
    -- which is what synthesized test markers use.
    marker_time_base      TEXT NOT NULL DEFAULT 'epoch'
                          CHECK (marker_time_base IN ('epoch', 'vod')),

    duration_s            REAL,
    fps                   REAL,                  -- display only

    -- NOT IN §3.2. FCPXML times must be exact integer multiples of a rational
    -- frameDuration: 29.97 is 1001/30000, not 1/29.97. A float fps cannot
    -- express that, and Resolve either rejects the file or silently rounds.
    fps_num               INTEGER,
    fps_den               INTEGER,

    -- NOT IN §3.2. §5.2 asserts the proxy shares the master's timebase, which
    -- is false for a VFR master. Recorded at probe time so the claim can be
    -- checked rather than assumed.
    is_vfr                INTEGER NOT NULL DEFAULT 0,

    resolution            TEXT,
    audio_track_map       TEXT,                  -- JSON: {"mic":1,"game":2,"party":3}
    obs_scene_log_path    TEXT,
    notes                 TEXT,
    created_at            TEXT DEFAULT (datetime('now'))
    -- An 'epoch' time base with no anchor cannot be converted to VOD seconds,
    -- but that is not expressible as a row constraint without coupling it to
    -- when probe happens to run. Enforced at marker-parse time instead, where
    -- the error can say what to do about it.
);

-- ============================================================
-- PIPELINE STATE (idempotency / resumability)
-- ============================================================
CREATE TABLE pipeline_stages (
    stream_id     TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    stage         TEXT NOT NULL,     -- 'proxy','audio_split','whisperx',...
    status        TEXT NOT NULL CHECK (status IN ('pending','running','done','failed')),
    started_at    TEXT,
    finished_at   TEXT,
    output_hash   TEXT,
    error         TEXT,
    attempt_count INTEGER DEFAULT 0,

    -- NOT IN §3.2. C7 says a stage skips when its output exists. But "exists"
    -- is not the same as "was produced by the current settings" -- re-encoding
    -- a proxy at a different bitrate must re-run, not skip. This hashes the
    -- stage's effective inputs.
    params_hash   TEXT,

    -- NOT IN §3.2. A crash leaves status='running' forever. These let the next
    -- invocation tell a live run from a dead one instead of refusing to start.
    host          TEXT,
    pid           INTEGER,
    heartbeat_at  TEXT,

    PRIMARY KEY (stream_id, stage)
);

-- ============================================================
-- CONTINUOUS SIGNALS  (stored as BLOB arrays, NOT rows)
-- ============================================================
CREATE TABLE signal_series (
    stream_id      TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    kind           TEXT NOT NULL,    -- 'mic_rms','mic_f0','speech_rate',...
    sample_rate_hz REAL NOT NULL,    -- typically 1.0 or 10.0
    t0             REAL DEFAULT 0.0, -- offset of first sample, seconds
    n_samples      INTEGER NOT NULL,
    dtype          TEXT DEFAULT 'float32',
    data           BLOB NOT NULL,    -- raw numpy array bytes

    -- NOT IN §3.2. How this series was computed: hop, frame length, floor,
    -- source track. Without it, re-extracting at a different hop silently
    -- overwrites a series with an incomparable one and nothing records that
    -- the two halves of the back catalogue no longer match.
    params         TEXT,
    created_at     TEXT DEFAULT (datetime('now')),

    PRIMARY KEY (stream_id, kind)
);
-- RATIONALE (A6): row-per-second-per-signal would produce ~8.6M rows over
-- 100 streams. As float32 arrays this is a few hundred MB and loads in
-- milliseconds.

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
-- RATIONALE: uniform shape means adding a new discrete signal requires ZERO
-- schema migration. New sensor = new `source` value.

-- ============================================================
-- TRANSCRIPT  (Phase 2)
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
-- DEVIATION from §3.2. §6.1 promises scoring is "infinitely rerunnable at zero
-- cost", but §3.2 gives candidates no stable identity and hangs ratings off
-- candidate_id with ON DELETE CASCADE. Re-scoring under that schema deletes the
-- operator's judgment calls -- the one thing §13.2 says cannot be
-- reconstructed.
--
-- So candidates are APPEND-ONLY GENERATIONS. Each score run inserts a new
-- generation for the stream; older rows and their ratings persist forever.
-- is_current marks the generation the review UI serves.
CREATE TABLE candidates (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id            TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    generation           INTEGER NOT NULL,       -- 1, 2, 3... per stream
    is_current           INTEGER NOT NULL DEFAULT 1,
    profile              TEXT NOT NULL,          -- which profile produced this row

    t_start              REAL NOT NULL,
    t_end                REAL NOT NULL,
    t_peak               REAL NOT NULL,

    score_entertainment  REAL NOT NULL,
    score_gameplay       REAL NOT NULL,
    score_combined       REAL NOT NULL,          -- §6.5 -- NOT a simple sum

    contributing_signals TEXT,                   -- JSON {signal: contribution}
    feature_vector       TEXT NOT NULL,          -- JSON, FULL vector, always (A9)

    -- NOT IN §3.2. Which revision of config/feature_schema.yaml defined the
    -- key set in feature_vector. Without it a Phase 1 vector and a Phase 3
    -- vector cannot be told apart by shape.
    feature_schema_version INTEGER NOT NULL,

    config_version       TEXT NOT NULL,          -- <profile>@<hash>

    -- NOT IN §3.2. Set when export rounds the boundaries onto the frame grid,
    -- so the UI and the timeline never disagree about where a cut is.
    frame_snapped        INTEGER NOT NULL DEFAULT 0,

    preview_path         TEXT,                   -- relative to data_root
    thumbstrip_path      TEXT,
    waveform_path        TEXT,
    created_at           TEXT DEFAULT (datetime('now')),

    CHECK (t_end > t_start),
    CHECK (t_peak >= t_start AND t_peak <= t_end),
    UNIQUE (stream_id, generation, profile, t_peak)
);
CREATE INDEX idx_candidates_stream  ON candidates(stream_id);
CREATE INDEX idx_candidates_scores  ON candidates(score_combined DESC);
CREATE INDEX idx_candidates_current ON candidates(stream_id, is_current, score_combined DESC);
CREATE INDEX idx_candidates_window  ON candidates(stream_id, t_start, t_end);

-- ============================================================
-- RATINGS  (the primary training signal)
-- ============================================================
CREATE TABLE ratings (
    candidate_id INTEGER PRIMARY KEY REFERENCES candidates(id) ON DELETE CASCADE,
    rating       INTEGER NOT NULL CHECK (rating IN (0, 1, 2)),  -- skip/maybe/clip it
    tags         TEXT,               -- JSON array, free-form
    note         TEXT,
    adjusted_start REAL,             -- if operator trimmed
    adjusted_end   REAL,

    -- NOT IN §3.2. When a re-score produces a new generation, ratings are
    -- carried forward onto the matching new candidate rather than lost.
    -- 'inherited' rows are copies; §14's signal_firing_rate_by_rating must
    -- filter to 'operator' so a carried-forward rating is never counted twice
    -- as training signal.
    rating_source TEXT NOT NULL DEFAULT 'operator'
                  CHECK (rating_source IN ('operator', 'inherited')),
    inherited_from INTEGER REFERENCES candidates(id) ON DELETE SET NULL,

    rated_at     TEXT DEFAULT (datetime('now')),
    review_ms    INTEGER             -- time spent, for instrumentation (§7.5)
);

-- ============================================================
-- EXPORTS / PUBLISHED
-- ============================================================
-- DEVIATION from §3.2: `exports` there has a single candidate_id, which cannot
-- describe an FCPXML timeline referencing forty clips across fifteen source
-- files (§10.5). candidate_id stays for single-clip short-form exports;
-- multi-clip timelines use export_items.
CREATE TABLE exports (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id   INTEGER REFERENCES candidates(id),   -- single-clip exports
    stream_id      TEXT REFERENCES streams(id),         -- NULL for cross-stream
    kind           TEXT NOT NULL DEFAULT 'clip'
                   CHECK (kind IN ('clip', 'fcpxml', 'edl')),
    path           TEXT NOT NULL,                       -- relative to data_root
    preset         TEXT,                                -- 'shorts','tiktok','reels'
    hook_text      TEXT,
    config_version TEXT,
    item_count     INTEGER,
    exported_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE export_items (
    export_id    INTEGER NOT NULL REFERENCES exports(id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,       -- position on the timeline
    candidate_id INTEGER REFERENCES candidates(id) ON DELETE SET NULL,
    t_start      REAL NOT NULL,          -- frame-snapped, as written to the file
    t_end        REAL NOT NULL,
    role         TEXT,                   -- 'clip','gap' -- §10.5 marks gaps explicitly
    label        TEXT,
    PRIMARY KEY (export_id, seq)
);

CREATE TABLE performance (
    export_id        INTEGER PRIMARY KEY REFERENCES exports(id),
    platform         TEXT,
    views            INTEGER,
    retention_pct    REAL,
    shares           INTEGER,
    normalized_score REAL,           -- vs rolling median of last 20
    measured_at      TEXT
);

-- ============================================================
-- DIGESTS  (Phase 5)
-- ============================================================
CREATE TABLE digests (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id  TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    version    INTEGER NOT NULL,
    content    TEXT NOT NULL,        -- JSON, structured (§9.2)
    markdown   TEXT,                 -- human-readable mirror
    model_used TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX idx_digest_ver ON digests(stream_id, version);

-- ============================================================
-- IDEAS  (Phase 5/6 -- living records, accumulate evidence over months)
-- ============================================================
CREATE TABLE ideas (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    kind           TEXT NOT NULL,    -- 'per_stream','cross_stream','bit','compilation'
    title          TEXT NOT NULL,
    premise        TEXT NOT NULL,
    beat_structure TEXT,             -- JSON array of beats
    status         TEXT DEFAULT 'open',  -- 'open','ready','shot','published','dead'
    coverage_pct   REAL,
    gaps           TEXT,             -- JSON: which beats lack footage
    origin         TEXT,             -- 'llm_ideation','cluster','ngram','manual'
    created_at     TEXT DEFAULT (datetime('now')),
    last_scored    TEXT
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
-- TRENDS  (Phase 6 -- requires 60+ streams to say anything real)
-- ============================================================
CREATE TABLE clusters (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    label      TEXT,                 -- LLM-generated
    n_members  INTEGER,
    n_streams  INTEGER,              -- streams spanned -- the key metric
    first_seen TEXT,
    last_seen  TEXT,
    centroid   BLOB,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE cluster_members (
    cluster_id   INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    distance     REAL,
    PRIMARY KEY (cluster_id, candidate_id)
);

CREATE TABLE ngrams (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    phrase          TEXT NOT NULL,
    n               INTEGER NOT NULL,
    total_count     INTEGER,
    stream_count    INTEGER,
    first_seen      TEXT,
    last_seen       TEXT,
    recency_score   REAL,            -- §11.2
    is_baseline_tic INTEGER DEFAULT 0,
    UNIQUE (phrase)
);

CREATE TABLE open_loops (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id          TEXT REFERENCES streams(id),
    t                  REAL,
    text               TEXT NOT NULL,
    kind               TEXT,         -- 'promise','question','unsolved'
    status             TEXT DEFAULT 'open',  -- 'open','resolved','dead'
    resolved_stream_id TEXT,
    created_at         TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- INSTRUMENTATION  (§14 -- the only way to know any of this works)
-- ============================================================
CREATE TABLE tool_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id   TEXT,
    metric      TEXT NOT NULL,
    value       REAL,
    meta        TEXT,
    recorded_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_tool_metrics_metric ON tool_metrics(metric, recorded_at);
