"""What the review screen reads and writes.

Kept out of the web layer so the shape of a candidate payload can be tested
without starting a server, and so the ordering rules live somewhere a person
can find them.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from dataclasses import dataclass, field

from clipforge import signals

#: Signals the sparkline can draw, in preference order. mic_rms is the one that
#: matters in Phase 1; the others exist so the panel keeps working when Phase 3
#: adds them.
SPARKLINE_KINDS = ("mic_rms", "party_rms", "game_rms")

#: Samples in the rendered sparkline. Enough to see the shape of a 60 s window,
#: few enough that the payload for 150 candidates stays small.
SPARKLINE_POINTS = 120


def list_streams(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT s.id, s.date, s.title, s.duration_s, s.resolution, s.proxy_path,
               s.audio_track_map, s.profile_used,
               (SELECT COUNT(*) FROM candidates c
                 WHERE c.stream_id = s.id AND c.is_current = 1) AS candidates,
               (SELECT COUNT(*) FROM candidates c JOIN ratings r ON r.candidate_id = c.id
                 WHERE c.stream_id = s.id AND c.is_current = 1
                   AND r.rating_source = 'operator') AS rated,
               (SELECT COUNT(*) FROM pipeline_stages p
                 WHERE p.stream_id = s.id AND p.status = 'done') AS stages_done
          FROM streams s
         ORDER BY s.date DESC, s.id
        """
    ).fetchall()

    out = []
    for row in rows:
        track_map = json.loads(row["audio_track_map"]) if row["audio_track_map"] else {}
        out.append({
            "id": row["id"],
            "date": row["date"],
            "title": row["title"],
            "duration_s": row["duration_s"],
            "resolution": row["resolution"],
            "has_proxy": bool(row["proxy_path"]),
            "profile_used": row["profile_used"],
            "candidates": row["candidates"],
            "rated": row["rated"],
            "stages_done": row["stages_done"],
            # The §4.2 contamination notice belongs in front of the operator,
            # not buried in a log they will never open.
            "warnings": track_map.get("warnings", []),
        })
    return out


def stream_detail(conn: sqlite3.Connection, stream_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM streams WHERE id = ?", (stream_id,)).fetchone()
    if row is None:
        return None
    track_map = json.loads(row["audio_track_map"]) if row["audio_track_map"] else {}
    return {
        "id": row["id"],
        "title": row["title"],
        "date": row["date"],
        "duration_s": row["duration_s"],
        "resolution": row["resolution"],
        "fps": row["fps"],
        "is_vfr": bool(row["is_vfr"]),
        "has_proxy": bool(row["proxy_path"]),
        "profile_used": row["profile_used"],
        "warnings": track_map.get("warnings", []),
        "roles": track_map.get("roles", {}),
    }


@dataclass
class CandidateView:
    """One candidate as the UI needs it."""

    id: int
    index: int
    t_start: float
    t_end: float
    t_peak: float
    score: float
    generation: int
    profile: str
    config_version: str
    contributions: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    marker_anchored: bool = False
    markers: list[float] = field(default_factory=list)
    sparkline: list[float] = field(default_factory=list)
    sparkline_kind: str | None = None
    sparkline_range: list[float] = field(default_factory=list)
    rating: int | None = None
    rating_source: str | None = None
    note: str | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "index": self.index,
            "t_start": round(self.t_start, 3),
            "t_end": round(self.t_end, 3),
            "t_peak": round(self.t_peak, 3),
            "duration_s": round(self.t_end - self.t_start, 2),
            "score": round(self.score, 4),
            "generation": self.generation,
            "profile": self.profile,
            "config_version": self.config_version,
            "contributions": self.contributions,
            "context": self.context,
            "marker_anchored": self.marker_anchored,
            "markers": [round(t, 2) for t in self.markers],
            "sparkline": self.sparkline,
            "sparkline_kind": self.sparkline_kind,
            "sparkline_range": self.sparkline_range,
            "rating": self.rating,
            "rating_source": self.rating_source,
            "note": self.note,
        }


def load_candidates(conn: sqlite3.Connection, stream_id: str) -> list[CandidateView]:
    """Current candidates, ranked, with everything the screen needs.

    Loaded in one pass and sent in one payload: C4's target is four seconds per
    candidate, and a network round trip per `j` press would spend most of that
    on latency.
    """
    rows = conn.execute(
        """
        SELECT c.*, r.rating, r.rating_source, r.note
          FROM candidates c
          LEFT JOIN ratings r ON r.candidate_id = c.id
         WHERE c.stream_id = ? AND c.is_current = 1
         ORDER BY c.score_combined DESC, c.t_peak
        """,
        (stream_id,),
    ).fetchall()
    if not rows:
        return []

    markers = [
        float(r["t"]) for r in conn.execute(
            "SELECT t FROM events WHERE stream_id = ? AND source = 'marker' ORDER BY t",
            (stream_id,),
        )
    ]
    series = _sparkline_series(conn, stream_id)

    out = []
    for index, row in enumerate(rows):
        payload = json.loads(row["contributing_signals"] or "{}")
        contributions = {k: v for k, v in payload.items() if not k.startswith("_")}
        context = {k.lstrip("_"): v for k, v in payload.items() if k.startswith("_")}

        inside = [t for t in markers if row["t_start"] <= t <= row["t_end"]]
        spark, low_high = _sparkline_for(series, row["t_start"], row["t_end"])

        out.append(CandidateView(
            id=row["id"], index=index,
            t_start=row["t_start"], t_end=row["t_end"], t_peak=row["t_peak"],
            score=row["score_combined"], generation=row["generation"],
            profile=row["profile"], config_version=row["config_version"],
            contributions=contributions, context=context,
            # §7.4's fourth section: the operator marked these deliberately, so
            # they are the safety net when the weights rank them low.
            marker_anchored=bool(inside) or contributions.get("marker_definite", 0) > 0
                            or contributions.get("marker_maybe", 0) > 0,
            markers=inside,
            sparkline=spark, sparkline_kind=series[0] if series else None,
            sparkline_range=low_high,
            rating=row["rating"], rating_source=row["rating_source"], note=row["note"],
        ))
    return out


def _sparkline_series(conn: sqlite3.Connection, stream_id: str):
    """The best available signal to draw, as (kind, Series)."""
    available = set(signals.kinds(conn, stream_id))
    for kind in SPARKLINE_KINDS:
        if kind in available:
            series = signals.load(conn, stream_id, kind)
            if series is not None and len(series):
                return kind, series
    return None


def _sparkline_for(series, t_start: float, t_end: float) -> tuple[list[float], list[float]]:
    """Downsample the window to a fixed number of points, plus its dB range.

    Drawn from `signal_series`, which is already in the database — no ffmpeg,
    no files, none of §7.2's Phase 3 assets. It answers "where is the loud part"
    without playing anything.
    """
    if series is None:
        return [], []

    _kind, data = series
    chunk = data.slice(t_start, t_end)
    if chunk.size == 0:
        return [], []

    values = chunk.astype(float)
    low, high = float(values.min()), float(values.max())
    if len(values) > SPARKLINE_POINTS:
        # Max within each bucket: an envelope should show peaks, and averaging
        # would flatten exactly the transients being looked for.
        buckets = [
            values[i * len(values) // SPARKLINE_POINTS:(i + 1) * len(values) // SPARKLINE_POINTS]
            for i in range(SPARKLINE_POINTS)
        ]
        values = [float(b.max()) for b in buckets if b.size]
    else:
        values = [float(v) for v in values]

    return [round(v, 2) for v in values], [round(low, 2), round(high, 2)]


# --------------------------------------------------------------------------
# writes
# --------------------------------------------------------------------------


def save_rating(
    conn: sqlite3.Connection, candidate_id: int, rating: int, review_ms: int | None,
    note: str | None = None,
) -> None:
    """Record an operator rating (§3.2 ratings, §7.5 instrumentation).

    Always `rating_source='operator'`: this is a human pressing a key, which is
    the distinction §14's weight tuning depends on when a re-score has carried
    inherited copies alongside.
    """
    if rating not in (0, 1, 2):
        raise ValueError(f"rating must be 0, 1 or 2 (§7.3), got {rating!r}")

    conn.execute(
        """
        INSERT INTO ratings (candidate_id, rating, note, review_ms, rating_source)
        VALUES (?, ?, ?, ?, 'operator')
        ON CONFLICT(candidate_id) DO UPDATE SET
            rating = excluded.rating,
            note = COALESCE(excluded.note, ratings.note),
            review_ms = excluded.review_ms,
            rating_source = 'operator',
            inherited_from = NULL,
            rated_at = datetime('now')
        """,
        (candidate_id, rating, note, review_ms),
    )


def record_session(
    conn: sqlite3.Connection, stream_id: str, duration_s: float, reviewed: int
) -> None:
    """§7.5 / §14: how the C4 target gets verified rather than assumed."""
    conn.execute(
        "INSERT INTO tool_metrics (stream_id, metric, value, meta) VALUES (?, ?, ?, ?)",
        (stream_id, "review_session_duration_s", float(duration_s),
         json.dumps({"reviewed": reviewed})),
    )


def review_metrics(conn: sqlite3.Connection, stream_id: str) -> dict:
    """§7.1's target, measured.

    Reported as a **median**. Leave the tab open over lunch and one candidate
    reads forty minutes; a mean would be swamped by it and the four-second
    target would become unmeasurable. The raw values stay in the database
    because they are the honest observation.
    """
    times = [
        int(r["review_ms"]) for r in conn.execute(
            """
            SELECT r.review_ms FROM ratings r JOIN candidates c ON c.id = r.candidate_id
             WHERE c.stream_id = ? AND r.rating_source = 'operator'
               AND r.review_ms IS NOT NULL
            """,
            (stream_id,),
        )
    ]
    sessions = [
        dict(r) for r in conn.execute(
            "SELECT value, meta, recorded_at FROM tool_metrics "
            "WHERE stream_id = ? AND metric = 'review_session_duration_s' "
            "ORDER BY recorded_at",
            (stream_id,),
        )
    ]

    counts = conn.execute(
        """
        SELECT r.rating, COUNT(*) AS n
          FROM candidates c JOIN ratings r ON r.candidate_id = c.id
         WHERE c.stream_id = ? AND c.is_current = 1 AND r.rating_source = 'operator'
         GROUP BY r.rating
        """,
        (stream_id,),
    ).fetchall()
    by_rating = {int(r["rating"]): int(r["n"]) for r in counts}
    total_rated = sum(by_rating.values())
    total = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE stream_id = ? AND is_current = 1",
        (stream_id,),
    ).fetchone()[0]

    return {
        "candidates": total,
        "rated": total_rated,
        "by_rating": by_rating,
        # §14: approved / total, "is the threshold correct?"
        "approval_rate": round(by_rating.get(2, 0) / total, 4) if total else None,
        "median_review_ms": int(statistics.median(times)) if times else None,
        "mean_review_ms": int(statistics.fmean(times)) if times else None,
        "sessions": sessions,
    }
