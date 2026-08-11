"""Continuous signal series: the `signal_series` table as a domain type.

A6 is the reason this is a BLOB and not rows:

> "Row-per-second-per-signal produces ~8.6M rows over 100 streams and makes
> queries slow for no benefit."

Both extraction and scoring construct these, so the type lives at package root
rather than under `db/` — §2.3 lists a `db/queries.py`, but a series is a thing
the pipeline passes around, not a query.

**Timestamps.** Sample *i* is centred at `t0 + i / sample_rate_hz`. `t0` is not
decoration: frame-based signals like RMS measure a *window*, and the honest
timestamp for that measurement is the window's centre, not its leading edge.
Recording the offset here means no downstream stage has to remember a framing
convention that was decided in an extraction module it never reads.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Little-endian float32, written explicitly.
#:
#: §3.2 defaults the `dtype` column to the string 'float32', which does not say
#: which byte order — and `ndarray.tobytes()` is native-endian. Every machine
#: this will realistically run on is little-endian, so this is cheap insurance
#: rather than a live concern, but the database is the tier §13.2 calls
#: irreplaceable and its blobs should describe themselves.
DTYPE = "<f4"


@dataclass
class Series:
    """One continuous signal over one stream."""

    kind: str
    values: np.ndarray
    sample_rate_hz: float
    t0: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.ascontiguousarray(self.values, dtype=DTYPE)

    def __len__(self) -> int:
        return int(self.values.size)

    @property
    def duration_s(self) -> float:
        """Span covered, from the first sample's centre to the last."""
        if len(self) <= 1:
            return 0.0
        return (len(self) - 1) / self.sample_rate_hz

    def time_of(self, index: int) -> float:
        """Timestamp of sample `index`, in seconds into the VOD."""
        return self.t0 + index / self.sample_rate_hz

    def times(self) -> np.ndarray:
        return self.t0 + np.arange(len(self), dtype=np.float64) / self.sample_rate_hz

    def index_at(self, t: float) -> int:
        """Nearest sample to time `t`, clamped to the series."""
        if len(self) == 0:
            raise IndexError(f"series {self.kind!r} is empty")
        raw = round((t - self.t0) * self.sample_rate_hz)
        return int(min(max(raw, 0), len(self) - 1))

    def value_at(self, t: float) -> float:
        return float(self.values[self.index_at(t)])

    def slice(self, t_start: float, t_end: float) -> np.ndarray:
        """Samples whose centres fall within [t_start, t_end]."""
        if len(self) == 0:
            return self.values[:0]
        lo = max(0, int(np.ceil((t_start - self.t0) * self.sample_rate_hz)))
        hi = min(len(self), int(np.floor((t_end - self.t0) * self.sample_rate_hz)) + 1)
        return self.values[lo:max(lo, hi)]


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def store(conn: sqlite3.Connection, stream_id: str, series: Series) -> None:
    """Write (or replace) one series. Caller owns the transaction."""
    conn.execute(
        """
        INSERT INTO signal_series
            (stream_id, kind, sample_rate_hz, t0, n_samples, dtype, data, params)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stream_id, kind) DO UPDATE SET
            sample_rate_hz = excluded.sample_rate_hz,
            t0             = excluded.t0,
            n_samples      = excluded.n_samples,
            dtype          = excluded.dtype,
            data           = excluded.data,
            params         = excluded.params,
            created_at     = datetime('now')
        """,
        (
            stream_id, series.kind, float(series.sample_rate_hz), float(series.t0),
            len(series), DTYPE, series.values.tobytes(),
            json.dumps(series.params, sort_keys=True),
        ),
    )


def load(conn: sqlite3.Connection, stream_id: str, kind: str) -> Series | None:
    row = conn.execute(
        "SELECT * FROM signal_series WHERE stream_id = ? AND kind = ?",
        (stream_id, kind),
    ).fetchone()
    return _from_row(row) if row is not None else None


def load_all(conn: sqlite3.Connection, stream_id: str) -> dict[str, Series]:
    rows = conn.execute(
        "SELECT * FROM signal_series WHERE stream_id = ? ORDER BY kind", (stream_id,)
    ).fetchall()
    return {row["kind"]: _from_row(row) for row in rows}


def kinds(conn: sqlite3.Connection, stream_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT kind FROM signal_series WHERE stream_id = ? ORDER BY kind", (stream_id,)
    ).fetchall()
    return [row["kind"] for row in rows]


def _from_row(row: sqlite3.Row) -> Series:
    dtype = row["dtype"] or DTYPE
    # .copy() because np.frombuffer wraps the sqlite bytes and is read-only,
    # and ascontiguousarray passes an already-correct array straight through.
    # Scoring resamples and smooths these in place; a read-only array would
    # fail there rather than here, a long way from the cause.
    values = np.frombuffer(row["data"], dtype=np.dtype(dtype)).copy()
    if values.size != row["n_samples"]:
        raise ValueError(
            f"signal_series {row['kind']!r} for {row['stream_id']!r} declares "
            f"{row['n_samples']} samples but the blob holds {values.size}"
        )
    return Series(
        kind=row["kind"],
        values=values,
        sample_rate_hz=float(row["sample_rate_hz"]),
        t0=float(row["t0"] or 0.0),
        params=json.loads(row["params"]) if row["params"] else {},
    )


def summarize(series: Series) -> dict[str, float]:
    """Stats for `clipforge signals` and for instrumentation."""
    if len(series) == 0:
        return {"n": 0}
    values = series.values.astype(np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": len(series)}
    return {
        "n": len(series),
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }
