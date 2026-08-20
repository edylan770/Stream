-- The rubric, plus three gaps closed while every affected table is still empty.
--
-- ONE migration rather than four. `open_loops`, `digests` and `performance`
-- have no rows anywhere -- nothing reads or writes them yet -- and a column is
-- vastly cheaper to add before that stops being true. 0001_init.sql already
-- states the principle for creating unused tables ("an empty table costs
-- nothing; retrofitting one costs a migration and a backfill"); this is the
-- same argument one level down, applied to their columns.

-- ============================================================
-- RUBRICS  (the learning layer -- no § reference, the spec has no section)
-- ============================================================
--
-- A versioned plain-language document the operator writes after a review batch:
-- what worked, what did not, what to watch for. It is fed into the LLM ranking
-- and ideation prompts, and it is the PRIMARY learning mechanism here rather
-- than fitted weights -- it works at n=1, it is interpretable, it survives a
-- re-score, and it holds judgements no feature vector can carry.
--
-- APPEND-ONLY, and never UPDATEd. §9.1 keeps digests forever and never
-- regenerates them, so a digest made under rubric v3 is a different artifact
-- from one made under v4 and v3 has to stay readable. `digests.rubric_version`
-- below is what records which was in force.
--
-- ONE free-text column, deliberately. Named sections (what_worked / what_didnt
-- / watch_for) would decide the shape of the operator's thinking before a
-- single rubric has been written, and markdown headings inside `text` cost
-- nothing if a shape turns out to want one. C5, applied to a table.
--
-- NOT a config file, unlike prompts.yaml / phrases.yaml / crop_templates.yaml.
-- Those are git-tracked, hand-edited and read-only at runtime; this is written
-- by the review server, per review batch, and needs version history. A server
-- writing into its own installed package directory would also be wrong.
CREATE TABLE rubrics (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    version    INTEGER NOT NULL UNIQUE,   -- 1, 2, 3... the row's identity
    text       TEXT NOT NULL,

    -- How much evidence stood behind this version, recorded at write time so a
    -- rubric written after four streams is legible as such a year later rather
    -- than reading as considered advice. Nullable: a rubric written before any
    -- stream exists is a legitimate thing to write.
    n_streams_at_write INTEGER,
    n_ratings_at_write INTEGER,

    created_at TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- THE THREE GAPS
-- ============================================================

-- §9.2's digest JSON gives every open loop a `segment_id`; the table had
-- nowhere to put it. §12.3 requires a verbatim quote per selection and the
-- validator checks it against the named segment's own text -- so without this
-- the check runs at ingest and then its evidence is discarded, and a loop can
-- never be re-audited against the line it came from.
--
-- A REFERENCES clause is legal in ALTER TABLE ADD COLUMN only when the default
-- is NULL, which is what this wants anyway: §11.3 leaves loops resolvable by
-- hand, and a loop whose segment was deleted is still a loop.
ALTER TABLE open_loops ADD COLUMN segment_id INTEGER REFERENCES segments(id);

-- §9.1: "Digests are first-class rows, never regenerable cache. Keep every
-- version forever." Two digests of one stream made by two different prompts --
-- or under two different rubrics -- must be distinguishable in origin, and
-- `model_used` alone cannot do it. `llm.prompts.digest_of` has existed since
-- commit 42 with nowhere to store its result.
ALTER TABLE digests ADD COLUMN prompt_hash TEXT;
ALTER TABLE digests ADD COLUMN rubric_version INTEGER;

-- §3.2 gives `performance` `export_id` as its whole primary key, so ONE export
-- can carry ONE performance row. But the unit of an export here is a rendered
-- clip (HANDOFF: "the unit is a rendered clip"), and the three presets differ
-- only by `max_duration_s` -- so the same file posted to Shorts and to TikTok
-- is one export with two performances, and the schema refused to record the
-- second.
--
-- A primary key cannot be altered in SQLite, so this is a rebuild. Safe here:
-- the table is empty, nothing references it, and the INSERT...SELECT is
-- correct anyway if some database somewhere does have rows.
--
-- `platform` becomes NOT NULL because it is now half the key, and SQLite
-- permits NULLs in a non-INTEGER PRIMARY KEY column -- which would let two
-- rows for one export both slip in with a null platform and defeat the point
-- of the change.
CREATE TABLE performance_new (
    export_id        INTEGER NOT NULL REFERENCES exports(id),
    platform         TEXT NOT NULL,
    views            INTEGER,
    retention_pct    REAL,
    shares           INTEGER,
    normalized_score REAL,           -- vs rolling median of last 20
    measured_at      TEXT,
    PRIMARY KEY (export_id, platform)
);

INSERT INTO performance_new
    (export_id, platform, views, retention_pct, shares, normalized_score, measured_at)
SELECT export_id, COALESCE(platform, 'unknown'), views, retention_pct, shares,
       normalized_score, measured_at
  FROM performance;

DROP TABLE performance;
ALTER TABLE performance_new RENAME TO performance;
