-- Staleness needs an ordering that clock resolution cannot blur.
--
-- The runner treats a stage as stale when one of its dependencies completed
-- *after* it did. The obvious key for that is `finished_at`, but §3.2 defines
-- those timestamps as `datetime('now')`, which SQLite resolves to whole
-- seconds. The cheap stages -- marker_events parsing a few hundred lines,
-- score running over stored arrays -- routinely finish inside the same second
-- as the stage they depend on, and a same-second comparison reports "not
-- stale" for a dependency that genuinely re-ran.
--
-- The failure that produces is the quiet kind: the pipeline reports success
-- while serving artifacts built from superseded inputs. A monotonic counter
-- removes the ambiguity entirely and costs one integer per row.

ALTER TABLE pipeline_stages ADD COLUMN completed_seq INTEGER;

-- Backfill any pre-existing completions in finished_at order, so historical
-- rows at least compare consistently with each other. On a fresh database this
-- matches zero rows.
UPDATE pipeline_stages
SET completed_seq = (
    SELECT COUNT(*)
    FROM pipeline_stages AS earlier
    WHERE earlier.status = 'done'
      AND (
            earlier.finished_at < pipeline_stages.finished_at
            OR (earlier.finished_at = pipeline_stages.finished_at
                AND earlier.rowid <= pipeline_stages.rowid)
          )
)
WHERE status = 'done';
