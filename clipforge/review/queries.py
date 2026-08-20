"""What the review screen reads and writes.

Kept out of the web layer so the shape of a candidate payload can be tested
without starting a server, and so the ordering rules live somewhere a person
can find them.
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import numpy as np

from clipforge import moments, signals
from clipforge.render import words
from clipforge.score import combined as combined_score

#: Signals the sparkline draws, in stacking order. §7.2 asks for a pre-rendered
#: "mic + party RMS" PNG per candidate; this is that asset, drawn as SVG from
#: data already in the database instead — see `clipforge/review/previews.py` for
#: why the file is not written.
#:
#: The ROLE IS THE KIND NAME. `extract/features.py` writes one series per §4.2
#: role (`SIGNAL_FOR_ROLE = {"mic": "mic_rms", ...}`), so provenance needs no
#: `track_roles` lookup the way the transcript panel and the captions do — there
#: is no track index to resolve, and a kind cannot be wrong about its own role.
#:
#: `game_rms` is last and is drawn like the others; it is included because A9's
#: instinct applies to the screen too — a moment that is loud only in the game
#: is a different moment from one that is loud on the mic, and hiding the
#: difference makes the two look identical.
SPARKLINE_KINDS = ("mic_rms", "party_rms", "game_rms")

#: Samples in the rendered sparkline. Enough to see the shape of a 60 s window,
#: few enough that the payload for 150 candidates stays small.
SPARKLINE_POINTS = 120

#: §7.1's target, in ms per candidate: "120 candidates in under 8 minutes".
#: A fallback only — the caller passes `review.target_ms_per_candidate`. It
#: exists so this module can be called without a config object in a test.
DEFAULT_TARGET_MS = 4000

#: §14 names no metric for boundary nudges, so this name is INVENTED. §17 asks
#: for "how often the operator nudges boundaries during review" and that phrase
#: describes a count, which cannot tune anything on its own: it does not say
#: which way to move `min_window_s`. The row therefore carries direction, size,
#: and whether the detector's window was sitting on one of the two clamps —
#: an operator repeatedly extending windows that came out at exactly
#: `max_window_s` IS `max_window_s` being too low, stated in one row.
NUDGE_METRIC = "window_nudge_s"

#: Tolerance for "this window was sitting on a clamp". Window bounds come back
#: through SQLite REAL and through §6.3's snapping, so an exact equality test
#: against the configured value would report False on a window that is plainly
#: clamped. One millisecond is far below `nudge_step_s` and far above the error.
CLAMP_EPSILON_S = 0.001


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
        # §7.3 wants the transcript beside the window. Phase 2 ships
        # `extract.whisperx.enabled: false`, so a stream processed with defaults
        # has no segments at all — and a review screen that always reserved a
        # third of its width for an empty panel would be worse than one that
        # does not. The client uses this to decide.
        "has_transcript": bool(conn.execute(
            "SELECT EXISTS(SELECT 1 FROM segments WHERE stream_id = ?)",
            (stream_id,),
        ).fetchone()[0]),
    }


# --------------------------------------------------------------------------
# transcript (§7.3)
# --------------------------------------------------------------------------


def role_colours(cfg) -> dict[str, str]:
    """The caption palette, so review and the finished clip agree.

    Read from `render.captions.styles` rather than duplicated in CSS. If the
    two ever disagreed, the operator would rate a moment believing one person
    said it and then watch the export attribute it to the other — and nothing
    would report the mismatch.
    """
    styles = cfg.get("render.captions.styles", {}) or {}
    fallback = cfg.get("render.captions.colour", "#FFFFFF")
    return {
        name: (settings or {}).get("colour", fallback)
        for name, settings in styles.items()
    }


def load_transcript(conn: sqlite3.Connection, stream_id: str) -> list[dict]:
    """Every segment of a stream, once, with its §8.3 role resolved.

    Loaded whole and sliced per candidate in Python rather than queried per
    window: `load_candidates` exists to put everything the screen needs in one
    response, because a round trip per `j` press would spend most of C4's four
    seconds on latency. 120 windows would otherwise be 120 queries.

    The role comes from `segments.track`, not `segments.speaker` — the same
    provenance rule `render/words.py` uses for caption colour, and for the same
    reason: `speaker` is an energy comparison over the segment's span and can
    disagree with which track actually produced the words.
    """
    row = conn.execute(
        "SELECT audio_track_map FROM streams WHERE id = ?", (stream_id,)
    ).fetchone()
    roles = words.track_roles(row["audio_track_map"] if row else None)

    return [
        {
            "seq": int(segment["seq"]),
            "t": round(float(segment["t_start"]), 2),
            "t_end": round(float(segment["t_end"]), 2),
            "role": words.role_for(segment["track"], segment["speaker"], roles),
            "speaker": segment["speaker"],
            "text": segment["text"],
        }
        for segment in conn.execute(
            "SELECT seq, t_start, t_end, text, speaker, track FROM segments "
            "WHERE stream_id = ? ORDER BY t_start, seq",
            (stream_id,),
        )
    ]


def lines_in(transcript: list[dict], t_start: float, t_end: float) -> list[dict]:
    """The lines a clip of this window would contain.

    OVERLAP, not containment. A segment that starts before the window still
    puts speech inside it, and the clip will carry that speech — hiding it
    would misreport what is in the clip.
    """
    return [
        line for line in transcript
        if line["t_end"] >= t_start and line["t"] <= t_end
    ]


@dataclass
class CandidateView:
    """One candidate as the UI needs it."""

    id: int
    index: int
    t_start: float
    t_end: float
    t_peak: float
    score: float
    #: §3.2's other two score columns. The rail ranks §7.4's sections by these,
    #: and the `?` panel shows all three beside the breakdown that explains the
    #: primary one.
    score_entertainment: float = 0.0
    score_gameplay: float = 0.0
    #: §7.4's section, assigned server-side and DISJOINT — see `assign_sections`.
    section: str = ""
    generation: int = 0
    profile: str = ""
    config_version: str = ""
    contributions: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    marker_anchored: bool = False
    markers: list[float] = field(default_factory=list)
    #: One entry per available role: {"kind", "role", "points"}. A LIST rather
    #: than the single track this used to carry, because §7.2's waveform asset
    #: is specified as "mic + party" and picking whichever came first meant a
    #: two-person moment was drawn as if one person made it.
    sparklines: list[dict] = field(default_factory=list)
    #: ONE range shared by every track above, and that is the whole point:
    #: scaling each track to its own min/max would draw a silent party track at
    #: the same height as a shouting mic, which is a lie about the only thing
    #: the picture is for.
    sparkline_range: list[float] = field(default_factory=list)
    #: §7.2's precomputed assets, as URLs the browser can fetch. None until the
    #: `previews` stage has run — the screen falls back to seeking the proxy.
    preview_url: str | None = None
    thumbstrip_url: str | None = None
    rating: int | None = None
    rating_source: str | None = None
    note: str | None = None
    #: §7.3's nudged boundaries, when the operator has moved them. NULL means
    #: the detector's window still stands. Sent so that reopening a stream shows
    #: the operator's own trim rather than silently reverting to the detector's
    #: — and so `space` plays what the export will contain.
    adjusted_start: float | None = None
    adjusted_end: float | None = None
    #: §7.3's "transcript text for the window displayed alongside". Empty when
    #: the stream has no segments — see `stream_detail`'s `has_transcript`.
    transcript: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "index": self.index,
            "t_start": round(self.t_start, 3),
            "t_end": round(self.t_end, 3),
            "t_peak": round(self.t_peak, 3),
            "duration_s": round(self.t_end - self.t_start, 2),
            "adjusted_start": (None if self.adjusted_start is None
                               else round(self.adjusted_start, 3)),
            "adjusted_end": (None if self.adjusted_end is None
                             else round(self.adjusted_end, 3)),
            "score": round(self.score, 4),
            "score_entertainment": round(self.score_entertainment, 4),
            "score_gameplay": round(self.score_gameplay, 4),
            "section": self.section,
            "generation": self.generation,
            "profile": self.profile,
            "config_version": self.config_version,
            "contributions": self.contributions,
            "context": self.context,
            "marker_anchored": self.marker_anchored,
            "markers": [round(t, 2) for t in self.markers],
            "sparklines": self.sparklines,
            "sparkline_range": self.sparkline_range,
            "preview_url": self.preview_url,
            "thumbstrip_url": self.thumbstrip_url,
            "rating": self.rating,
            "rating_source": self.rating_source,
            "note": self.note,
            "transcript": self.transcript,
        }


#: §7.4's sections, in the order it lists them. The key is what the payload
#: carries; the label is what the rail prints.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("combined", "Combined winners"),
    ("entertainment", "Entertainment"),
    ("gameplay", "Gameplay"),
    ("markers", "Marker-anchored"),
)

#: A stream scored with one profile has no combined score distinct from its
#: primary and no gameplay ranking at all, so §7.4's four headers would be a lie
#: about what the numbers mean. `candidates.profile` holds the profile-SET name,
#: so the row itself says which case this is.
def is_sectioned(profile_set: str) -> bool:
    return len(str(profile_set or "").split("+")) > 1


def assign_sections(views: list[CandidateView], top_n: int) -> list[CandidateView]:
    """§7.4's four sections: disjoint, ordered, and returned in review order.

    > 1. Combined-score winners (both profiles high) — the best content,
    >    reviewed first
    > 2. Entertainment ranked
    > 3. Gameplay ranked
    > 4. Marker-anchored candidates that did not rank highly (safety net — the
    >    operator marked these deliberately)

    DISJOINT is the part §7.4 does not say and that matters most: a candidate
    appears in the first section it qualifies for. Listing one moment under
    three headers would spend §7.1's four-second budget rating the same window
    three times, on the screen C4 calls the critical path.

    Section 4 takes every marker-anchored candidate that missed the combined
    cut, which is what "did not rank highly" means once section 1 is the only
    thing above it. A key the operator pressed at that moment is a different
    kind of evidence from a weighted sum, and it should not be buried in the
    tail of a list sorted by weights.

    Sections 2 and 3 split what is left by which profile is more confident about
    it, compared as a SHARE of each profile's own best — raw composites are not
    comparable across profiles, since §6.5's two sum 20.7 and 21.0 of weight of
    which very different amounts have a producer today.
    """
    if not views:
        return []

    if not is_sectioned(views[0].profile):
        for view in views:
            view.section = ""
        return views

    entertainment = combined_score.normalise(
        np.array([v.score_entertainment for v in views])
    )
    gameplay = combined_score.normalise(np.array([v.score_gameplay for v in views]))

    ranked = sorted(views, key=lambda v: (-v.score, v.t_peak))
    order = {id(view): i for i, view in enumerate(views)}
    for place, view in enumerate(ranked):
        i = order[id(view)]
        if place < int(top_n):
            view.section = "combined"
        elif view.marker_anchored:
            view.section = "markers"
        elif entertainment[i] >= gameplay[i]:
            view.section = "entertainment"
        else:
            view.section = "gameplay"

    within = {
        "combined": lambda v: (-v.score, v.t_peak),
        "entertainment": lambda v: (-v.score_entertainment, v.t_peak),
        "gameplay": lambda v: (-v.score_gameplay, v.t_peak),
        "markers": lambda v: (-v.score, v.t_peak),
    }
    out: list[CandidateView] = []
    for key, _label in SECTIONS:
        out.extend(sorted((v for v in views if v.section == key), key=within[key]))

    for index, view in enumerate(out):
        view.index = index
    return out


def load_candidates(
    conn: sqlite3.Connection, stream_id: str, section_top_n: int = 20
) -> list[CandidateView]:
    """Current candidates, ranked, with everything the screen needs.

    Loaded in one pass and sent in one payload: C4's target is four seconds per
    candidate, and a network round trip per `j` press would spend most of that
    on latency.
    """
    rows = conn.execute(
        """
        SELECT c.*, r.rating, r.rating_source, r.note,
               r.adjusted_start, r.adjusted_end
          FROM candidates c
          LEFT JOIN ratings r ON r.candidate_id = c.id
         WHERE c.stream_id = ? AND c.is_current = 1
         ORDER BY c.score_combined DESC, c.t_peak
        """,
        (stream_id,),
    ).fetchall()
    if not rows:
        return []

    markers = moments.marker_times(conn, stream_id)
    series = _sparkline_series(conn, stream_id)
    # ONE query for the whole stream, sliced per window below.
    transcript = load_transcript(conn, stream_id)

    out = []
    for index, row in enumerate(rows):
        payload = json.loads(row["contributing_signals"] or "{}")
        contributions = {k: v for k, v in payload.items() if not k.startswith("_")}
        context = {k.lstrip("_"): v for k, v in payload.items() if k.startswith("_")}

        anchoring = moments.marker_anchoring(
            row["t_start"], row["t_end"], markers, contributions)
        tracks, low_high = _sparklines_for(series, row["t_start"], row["t_end"])

        out.append(CandidateView(
            id=row["id"], index=index,
            t_start=row["t_start"], t_end=row["t_end"], t_peak=row["t_peak"],
            score=row["score_combined"],
            score_entertainment=row["score_entertainment"],
            score_gameplay=row["score_gameplay"],
            generation=row["generation"],
            profile=row["profile"], config_version=row["config_version"],
            contributions=contributions, context=context,
            # §7.4's fourth section: the operator marked these deliberately, so
            # they are the safety net when the weights rank them low. `.any` is
            # the LOOSE reading — near a press, or scored by one. §14's
            # `marker_precision` deliberately takes `.press_inside` instead; see
            # `moments.MarkerAnchoring` for why they must not be the same test.
            marker_anchored=anchoring.any,
            markers=list(anchoring.press_inside),
            sparklines=tracks, sparkline_range=low_high,
            # §7.2's assets, when `previews` has run. Served by path rather
            # than embedded: a 2 s webm is ~250 KB and 120 of them inlined
            # would be a 30 MB JSON payload on the screen C4 calls the
            # critical path.
            preview_url=_asset_url(stream_id, row["preview_path"]),
            thumbstrip_url=_asset_url(stream_id, row["thumbstrip_path"]),
            rating=row["rating"], rating_source=row["rating_source"], note=row["note"],
            adjusted_start=row["adjusted_start"], adjusted_end=row["adjusted_end"],
            # Sliced on the DETECTOR's window, not the adjusted one. A nudge can
            # reach outside it and the extra lines are not in this payload —
            # which is what commit 31's `nudge_context_s` padding is for.
            transcript=lines_in(transcript, row["t_start"], row["t_end"]),
        ))
    return assign_sections(out, section_top_n)


def _asset_url(stream_id: str, relative: str | None) -> str | None:
    """A preview asset as a URL, or None when `previews` has not run.

    Only the file NAME travels: the route resolves it under the stream's own
    previews directory, so a stored path that had been tampered with cannot
    address anything outside it.
    """
    if not relative:
        return None
    return f"/media/{stream_id}/preview/{PurePosixPath(str(relative)).name}"


def _sparkline_series(conn: sqlite3.Connection, stream_id: str) -> list[tuple[str, object]]:
    """EVERY available role signal, as [(kind, Series)] in stacking order.

    This used to return the first of the three and draw it alone, which was
    invisible while `mic_rms` was the only one that existed — and wrong the
    moment a stream had two people in it. §7.2 specifies the waveform asset as
    "mic + party RMS"; drawing only the mic loses the half that says whether the
    moment was a conversation.
    """
    available = set(signals.kinds(conn, stream_id))
    loaded: list[tuple[str, object]] = []
    for kind in SPARKLINE_KINDS:
        if kind not in available:
            continue
        series = signals.load(conn, stream_id, kind)
        if series is not None and len(series):
            loaded.append((kind, series))
    return loaded


#: `mic_rms` -> `mic`. The signal kinds are named by §4.2 role, so this is the
#: whole of the provenance rule (see SPARKLINE_KINDS).
def _role_of(kind: str) -> str:
    return kind.rsplit("_", 1)[0]


def _sparklines_for(series: list[tuple[str, object]], t_start: float, t_end: float
                    ) -> tuple[list[dict], list[float]]:
    """Downsample the window for each track, over ONE shared dB range.

    Drawn from `signal_series`, which is already in the database — no ffmpeg and
    no files. It answers "where is the loud part, and who was making it" without
    playing anything, which is §7.2's waveform asset arriving as vector rather
    than as 120 PNGs per stream.

    The shared range is load-bearing. Per-track normalisation would draw a party
    track sitting at its noise floor with the same amplitude as a mic that is
    being shouted into, so the one comparison the picture exists to support —
    who is louder — would be the one thing it cannot show.
    """
    if not series:
        return [], []

    tracks: list[dict] = []
    low, high = None, None
    for kind, data in series:
        chunk = data.slice(t_start, t_end)
        if chunk.size == 0:
            continue
        values = chunk.astype(float)
        low = float(values.min()) if low is None else min(low, float(values.min()))
        high = float(values.max()) if high is None else max(high, float(values.max()))
        if len(values) > SPARKLINE_POINTS:
            # Max within each bucket: an envelope should show peaks, and
            # averaging would flatten exactly the transients being looked for.
            points = [
                float(bucket.max())
                for bucket in (
                    values[i * len(values) // SPARKLINE_POINTS:
                           (i + 1) * len(values) // SPARKLINE_POINTS]
                    for i in range(SPARKLINE_POINTS)
                )
                if bucket.size
            ]
        else:
            points = [float(v) for v in values]
        tracks.append({
            "kind": kind,
            "role": _role_of(kind),
            "points": [round(v, 2) for v in points],
        })

    if low is None or high is None:
        return [], []
    return tracks, [round(low, 2), round(high, 2)]


# --------------------------------------------------------------------------
# writes
# --------------------------------------------------------------------------


def save_rating(
    conn: sqlite3.Connection, candidate_id: int, rating: int, review_ms: int | None,
    note: str | None = None,
    adjusted_start: float | None = None, adjusted_end: float | None = None,
) -> None:
    """Record an operator rating (§3.2 ratings, §7.5 instrumentation).

    Always `rating_source='operator'`: this is a human pressing a key, which is
    the distinction §14's weight tuning depends on when a re-score has carried
    inherited copies alongside.

    §7.3's nudged boundaries ride along with the rating rather than getting a
    write of their own, because `ratings.rating` is NOT NULL: an unrated nudge
    could only be stored by inventing a rating, and a fabricated rating poisons
    §14's `signal_firing_rate_by_rating` — the primary weight-tuning input.
    The nudge *metric* is recorded separately (`record_nudge`), so §17's number
    survives even when the operator nudges and moves on without rating.

    Passing None for the pair means "no opinion", not "clear it": re-rating a
    candidate that was trimmed in an earlier session must not silently restore
    the detector's window. Hence COALESCE rather than assignment.
    """
    if rating not in (0, 1, 2):
        raise ValueError(f"rating must be 0, 1 or 2 (§7.3), got {rating!r}")

    conn.execute(
        """
        INSERT INTO ratings (candidate_id, rating, note, review_ms, rating_source,
                             adjusted_start, adjusted_end)
        VALUES (?, ?, ?, ?, 'operator', ?, ?)
        ON CONFLICT(candidate_id) DO UPDATE SET
            rating = excluded.rating,
            note = COALESCE(excluded.note, ratings.note),
            review_ms = excluded.review_ms,
            rating_source = 'operator',
            inherited_from = NULL,
            adjusted_start = COALESCE(excluded.adjusted_start, ratings.adjusted_start),
            adjusted_end = COALESCE(excluded.adjusted_end, ratings.adjusted_end),
            rated_at = datetime('now')
        """,
        (candidate_id, rating, note, review_ms, adjusted_start, adjusted_end),
    )


def candidate_window(conn: sqlite3.Connection, candidate_id: int) -> sqlite3.Row | None:
    """A candidate's own window plus its stream's duration, for validation."""
    return conn.execute(
        """
        SELECT c.id, c.stream_id, c.t_start, c.t_end, c.t_peak, s.duration_s
          FROM candidates c JOIN streams s ON s.id = c.stream_id
         WHERE c.id = ?
        """,
        (candidate_id,),
    ).fetchone()


def check_window(start: float, end: float, duration_s: float | None) -> None:
    """Reject a nudged window that is arithmetically impossible. Nothing else.

    **Deliberately not checked against `score.window.min_window_s` /
    `max_window_s`.** Those are the two numbers §17 tunes *against these
    nudges*, so refusing a window outside their range would make the
    measurement circular — the operator could never record "I wanted this
    shorter than 8 seconds", which is precisely the observation being
    collected. A window shorter than `min_window_s` is a finding, not an error.
    """
    for name, value in (("start", start), ("end", end)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")
    if start < 0:
        raise ValueError(f"start ({start}) is before the beginning of the recording")
    if duration_s is not None and end > float(duration_s) + CLAMP_EPSILON_S:
        raise ValueError(f"end ({end}) is past the end of the recording ({duration_s})")


def record_nudge(
    conn: sqlite3.Connection, candidate_id: int, *,
    adjusted_start: float, adjusted_end: float, presses: int,
    min_window_s: float | None = None, max_window_s: float | None = None,
) -> dict:
    """§17's missing observation, as one row per nudged candidate.

    §17 tunes `min_window_s`/`max_window_s` against "how often the operator
    nudges boundaries during review". A bare count cannot do that job — it does
    not say which direction to move a bound. So the row carries:

    * `value` — the change in window LENGTH, signed. Negative means the operator
      wanted less than the detector gave.
    * `start_delta` / `end_delta` — which edge moved, and which way. Trimming
      dead air off the front is a different finding from extending past the end.
    * `presses` — how many keystrokes it took, which is the literal reading of
      "how often".
    * `at_min` / `at_max` — whether the DETECTOR's window was sitting on a
      clamp. This is the one that tunes: an operator repeatedly extending
      windows that came out at exactly `max_window_s` is `max_window_s` being
      too low, and no other field says so.

    Writes to `tool_metrics` only, never to `ratings` — so it is safe on a
    candidate the operator has not rated and never will.
    """
    row = candidate_window(conn, candidate_id)
    if row is None:
        raise ValueError(f"no candidate {candidate_id}")

    check_window(adjusted_start, adjusted_end, row["duration_s"])

    original = float(row["t_end"]) - float(row["t_start"])
    adjusted = adjusted_end - adjusted_start
    meta = {
        "candidate_id": int(candidate_id),
        "start_delta": round(adjusted_start - float(row["t_start"]), 3),
        "end_delta": round(adjusted_end - float(row["t_end"]), 3),
        "original_s": round(original, 3),
        "adjusted_s": round(adjusted, 3),
        "presses": int(presses),
        # Whether the operator's window still contains the peak the detector
        # found. `candidates` forbids this by CHECK; `ratings` does not, and a
        # trim to the second half of a moment is a legitimate thing to want.
        "keeps_peak": bool(adjusted_start <= float(row["t_peak"]) <= adjusted_end),
        "at_min": (min_window_s is not None
                   and abs(original - float(min_window_s)) <= CLAMP_EPSILON_S),
        "at_max": (max_window_s is not None
                   and abs(original - float(max_window_s)) <= CLAMP_EPSILON_S),
    }
    conn.execute(
        "INSERT INTO tool_metrics (stream_id, metric, value, meta) VALUES (?, ?, ?, ?)",
        (row["stream_id"], NUDGE_METRIC, round(adjusted - original, 3), json.dumps(meta)),
    )
    return meta


def record_session(
    conn: sqlite3.Connection, stream_id: str, duration_s: float, reviewed: int,
    nudged: int = 0,
) -> None:
    """§7.5 / §14: how the C4 target gets verified rather than assumed.

    `nudged` goes in this row's meta rather than becoming a metric of its own,
    because it is the denominator for `reviewed`: §17 asks how *often* the
    operator nudges, and a count with its own timestamp somewhere else could
    only be matched back to a session by guessing.
    """
    conn.execute(
        "INSERT INTO tool_metrics (stream_id, metric, value, meta) VALUES (?, ?, ?, ?)",
        (stream_id, "review_session_duration_s", float(duration_s),
         json.dumps({"reviewed": reviewed, "nudged": int(nudged)})),
    )


def nudge_summary(conn: sqlite3.Connection, stream_id: str) -> dict:
    """§17's tuning input for `min_window_s` / `max_window_s`.

    Deliberately reports the two clamp counts separately from the averages. An
    average delta near zero can mean "the windows are right" or "half were too
    long and half too short", and those call for opposite changes; the clamp
    counts are what distinguish them.
    """
    rows = [
        json.loads(r["meta"] or "{}") for r in conn.execute(
            "SELECT meta FROM tool_metrics WHERE stream_id = ? AND metric = ? "
            "ORDER BY recorded_at",
            (stream_id, NUDGE_METRIC),
        )
    ]
    if not rows:
        return {"nudged": 0}

    starts = [float(r.get("start_delta", 0.0)) for r in rows]
    ends = [float(r.get("end_delta", 0.0)) for r in rows]
    return {
        "nudged": len(rows),
        "presses": sum(int(r.get("presses", 0)) for r in rows),
        "mean_start_delta_s": round(statistics.fmean(starts), 3),
        "mean_end_delta_s": round(statistics.fmean(ends), 3),
        "mean_length_delta_s": round(
            statistics.fmean(
                float(r.get("adjusted_s", 0.0)) - float(r.get("original_s", 0.0))
                for r in rows
            ), 3),
        "extended": sum(
            1 for r in rows
            if float(r.get("adjusted_s", 0)) > float(r.get("original_s", 0))
        ),
        "trimmed": sum(
            1 for r in rows
            if float(r.get("adjusted_s", 0)) < float(r.get("original_s", 0))
        ),
        # The two that actually move a bound.
        "was_at_min": sum(1 for r in rows if r.get("at_min")),
        "was_at_max": sum(1 for r in rows if r.get("at_max")),
        "dropped_peak": sum(1 for r in rows if r.get("keeps_peak") is False),
    }


def review_metrics(
    conn: sqlite3.Connection, stream_id: str, target_ms: int = DEFAULT_TARGET_MS
) -> dict:
    """§7.1's target, measured.

    Reported as a **median**. Leave the tab open over lunch and one candidate
    reads forty minutes; a mean would be swamped by it and the four-second
    target would become unmeasurable. The raw values stay in the database
    because they are the honest observation.

    `target_ms` is returned rather than left to the caller because the browser
    used to carry its own copy of it: `review.js` graded the session against a
    hardcoded 4000 while `clipforge metrics` read
    `review.target_ms_per_candidate` from config. Changing the config moved one
    and not the other, and nothing said so.
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
        "target_ms": int(target_ms),
        "sessions": sessions,
        # §17's input for min_window_s / max_window_s — GUESSES gap 1.
        "nudges": nudge_summary(conn, stream_id),
    }
