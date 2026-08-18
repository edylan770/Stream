"""The review server (§7).

C4 makes this the critical path for the whole system: *"Every other subsystem
feeds this one screen. If review is slow, the tool goes unused and the entire
system is dead weight."* The hard target is 120 candidates in under 8 minutes,
about four seconds each.

Two design consequences follow, and both are visible in the routes below:

**Everything the screen needs arrives in one payload.** Candidates, their
contribution breakdowns and their sparklines are loaded once. A round trip per
`j` press would spend most of that four-second budget on latency.

**The only per-action request is the rating**, and it is fire-and-forget from
the client's side — the keyboard never waits on the network.

Around that screen sits the app shell: browsing for a recording, registering
it, and running the pipeline. Those routes are why `guard.py` exists — a file
browser and a run trigger are reachable by every page open in the operator's
browser, which a read-only API was not.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from clipforge import db, paths, search
from clipforge.ingest import register
from clipforge.pipeline import runner
from clipforge.review import browse, guard, jobs, media, queries

STATIC = Path(__file__).parent / "static"


def create_app(cfg) -> FastAPI:
    app = FastAPI(title="ClipForge", docs_url=None, redoc_url=None)
    app.state.cfg = cfg
    #: Per-app rather than module-level, so two apps in one test process do not
    #: share a registry and refuse each other's runs.
    app.state.jobs = jobs.JobRegistry()
    guard.install(app, cfg)

    def connect() -> sqlite3.Connection:
        return db.open_db(cfg.db_path, migrate_to_latest=False)

    def require_stream(conn: sqlite3.Connection, stream_id: str) -> None:
        row = conn.execute("SELECT id FROM streams WHERE id = ?", (stream_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"no stream {stream_id!r}")

    # -- pages ------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # -- data -------------------------------------------------------------

    @app.get("/api/streams")
    def api_streams() -> JSONResponse:
        conn = connect()
        try:
            return JSONResponse({"streams": queries.list_streams(conn)})
        finally:
            conn.close()

    @app.get("/api/streams/{stream_id}")
    def api_stream(stream_id: str) -> JSONResponse:
        conn = connect()
        try:
            detail = queries.stream_detail(conn, stream_id)
            if detail is None:
                raise HTTPException(status_code=404, detail=f"no stream {stream_id!r}")
            return JSONResponse(detail)
        finally:
            conn.close()

    @app.get("/api/streams/{stream_id}/candidates")
    def api_candidates(stream_id: str) -> JSONResponse:
        conn = connect()
        try:
            detail = queries.stream_detail(conn, stream_id)
            if detail is None:
                raise HTTPException(status_code=404, detail=f"no stream {stream_id!r}")
            found = queries.load_candidates(
                conn, stream_id,
                section_top_n=int(cfg.get("review.sections.combined_top_n", 20)),
            )
            return JSONResponse({
                "stream": detail,
                "candidates": [c.to_json() for c in found],
                # §7.4's section labels travel with the payload rather than
                # living in the markup, for the same reason the caption colours
                # do: two places naming the same thing is how they drift, and a
                # section the server stopped emitting would otherwise still
                # have a header in the rail.
                "sections": [{"key": key, "label": label}
                             for key, label in queries.SECTIONS],
                # §8.3's speaker palette, so the transcript beside the video is
                # coloured the same way the burned-in captions will be. Sent
                # rather than duplicated in CSS: if the two disagreed, the
                # operator would rate a moment believing one person said it and
                # watch the export attribute it to the other.
                "role_colours": queries.role_colours(cfg),
                # §7.3's nudge step, sent for the same reason `target_ms` is:
                # a copy of it in the browser would drift from the config and
                # nothing would say so.
                "nudge_step_s": float(cfg.get("review.nudge_step_s", 0.5)),
            })
        finally:
            conn.close()

    @app.get("/api/streams/{stream_id}/metrics")
    def api_metrics(stream_id: str) -> JSONResponse:
        conn = connect()
        try:
            return JSONResponse(queries.review_metrics(
                conn, stream_id,
                target_ms=int(cfg.get("review.target_ms_per_candidate",
                                      queries.DEFAULT_TARGET_MS)),
            ))
        finally:
            conn.close()

    # -- writes -----------------------------------------------------------

    def _adjusted(conn: sqlite3.Connection, candidate_id: int, body: dict):
        """§7.3's nudged window out of a request body, or (None, None).

        Absent means "no opinion" and leaves any stored adjustment alone; it
        does not mean "restore the detector's window". Present means both
        values, validated together — a half-specified window has no meaning.
        """
        start, end = body.get("adjusted_start"), body.get("adjusted_end")
        if start is None and end is None:
            return None, None
        if start is None or end is None:
            raise ValueError("adjusted_start and adjusted_end must be sent together")

        start, end = float(start), float(end)
        row = queries.candidate_window(conn, candidate_id)
        if row is None:
            raise ValueError(f"no candidate {candidate_id}")
        queries.check_window(start, end, row["duration_s"])
        return start, end

    @app.post("/api/candidates/{candidate_id}/rating")
    async def api_rate(candidate_id: int, request: Request) -> JSONResponse:
        body = await request.json()
        conn = connect()
        try:
            start, end = _adjusted(conn, candidate_id, body)
            with db.transaction(conn):
                queries.save_rating(
                    conn, candidate_id,
                    rating=int(body["rating"]),
                    review_ms=body.get("review_ms"),
                    note=body.get("note"),
                    adjusted_start=start,
                    adjusted_end=end,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            conn.close()
        return JSONResponse({"ok": True})

    @app.post("/api/candidates/{candidate_id}/nudge")
    async def api_nudge(candidate_id: int, request: Request) -> JSONResponse:
        """§17's missing observation. Writes a metric and nothing else.

        Separate from the rating route because a nudge and a judgment are
        different events: `ratings.rating` is NOT NULL, so storing a nudge on an
        unrated candidate would mean inventing a rating — and a fabricated
        rating corrupts §14's `signal_firing_rate_by_rating`, which §17 calls
        the primary weight-tuning input. So the *window* rides along with the
        rating and the *observation* comes here, which means it survives the
        operator nudging a boundary and then moving on without rating.
        """
        body = await request.json()
        conn = connect()
        try:
            with db.transaction(conn):
                meta = queries.record_nudge(
                    conn, candidate_id,
                    adjusted_start=float(body["adjusted_start"]),
                    adjusted_end=float(body["adjusted_end"]),
                    presses=int(body.get("presses", 1)),
                    min_window_s=cfg.get("score.window.min_window_s", None),
                    max_window_s=cfg.get("score.window.max_window_s", None),
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            conn.close()
        return JSONResponse({"ok": True, "recorded": meta})

    @app.post("/api/streams/{stream_id}/session")
    async def api_session(stream_id: str, request: Request) -> JSONResponse:
        body = await request.json()
        conn = connect()
        try:
            with db.transaction(conn):
                queries.record_session(
                    conn, stream_id,
                    duration_s=float(body.get("duration_s", 0.0)),
                    reviewed=int(body.get("reviewed", 0)),
                    nudged=int(body.get("nudged", 0)),
                )
        finally:
            conn.close()
        return JSONResponse({"ok": True})

    # -- adding a recording -----------------------------------------------

    @app.get("/api/browse")
    def api_browse(path: str | None = None) -> JSONResponse:
        """List one directory.

        The browser cannot hand a web page a filesystem path, so this is what
        stands in for a file dialog. See browse.py for why it is not a drop
        target.
        """
        listing = browse.list_dir(
            path or browse.default_start(),
            media_extensions=cfg.get("review.browse.media_extensions", ()),
            max_entries=int(cfg.get("review.browse.max_entries", 500)),
        )
        conn = connect()
        try:
            recent = browse.registered_directories(conn)
        finally:
            conn.close()
        # "Recent locations" with no new state to keep: the answer is already in
        # `streams.master_path`, and a derived list cannot go stale.
        return JSONResponse({**listing.to_json(), "recent": recent})

    @app.post("/api/browse/locate")
    async def api_locate(request: Request) -> JSONResponse:
        """Find a dropped file on disk, by name and size. NOTHING IS UPLOADED.

        The page sends `File.name` and `File.size` — metadata every browser
        exposes without reading a byte — and gets back the path the operator
        would otherwise have navigated to. §13.1 puts a master at 40-55 GB and
        it is already on this machine's disk; copying it to `data/` to discover
        where it lives would be minutes of I/O for nothing.

        A POST rather than a GET because a filename is the operator's data and
        does not belong in a URL, a log line, or a browser history entry.
        """
        body = await request.json()
        name = str(body.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="no file name given")

        size = body.get("size")
        conn = connect()
        try:
            directories = [
                # Where recordings have come from before is the best guess by a
                # distance, and it costs nothing to keep.
                *browse.registered_directories(conn),
                str(body.get("path") or "") or None,
                *(cfg.get("review.browse.search_dirs", []) or []),
                # Last: a guess about somebody else's OBS configuration, which
                # is only there so the FIRST drop on a fresh install — no
                # history, nothing browsed yet — has somewhere to look.
                *browse.default_search_dirs(),
            ]
        finally:
            conn.close()

        found = browse.locate(
            name, int(size) if isinstance(size, (int, float)) else None,
            directories=directories,
        )
        return JSONResponse(found.to_json())

    def _spec(body: dict) -> register.RegisterSpec:
        master = str(body.get("master") or "").strip()
        if not master:
            raise HTTPException(status_code=400, detail="no master path given")
        return register.RegisterSpec(
            master=Path(master).expanduser(),
            stream_id=body.get("id") or None,
            date=body.get("date") or None,
            title=body.get("title") or None,
            games=body.get("games") or None,
            notes=body.get("notes") or None,
        )

    @app.post("/api/register/preflight")
    async def api_preflight(request: Request) -> JSONResponse:
        """What registering this file would do, without doing it.

        Worth a round trip of its own because of the anchor. §4.1 makes
        `record_start_epoch_ms` the entire sync solution, and a stream
        registered without it silently reads §4.3's epoch-millisecond marker
        presses as VOD seconds. After the row is written is too late to notice.
        """
        conn = connect()
        try:
            # `with_audio`: §4.2's track map, read from the container header
            # before anything is written. Two or three audio tracks are refused
            # by `audio_split` rather than guessed at, and §4.2 calls confusing
            # game audio for the mic unrecoverable — so the operator sees which
            # tracks the file actually carries while the choice is still theirs.
            pre = register.preflight(
                cfg, conn, _spec(await request.json()), with_audio=True
            )
        except register.RegisterError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            conn.close()
        return JSONResponse(pre.to_json())

    @app.post("/api/register")
    async def api_register(request: Request) -> JSONResponse:
        conn = connect()
        try:
            result = register.register_stream(cfg, conn, _spec(await request.json()))
        except register.AlreadyRegistered as exc:
            # Not a failure: re-registration is a real operation that happens to
            # invalidate artifacts, so it stays a deliberate `--force` at a
            # terminal rather than a button.
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except register.RegisterError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            conn.close()
        return JSONResponse(result.to_json())

    # -- running the pipeline ---------------------------------------------

    @app.get("/api/streams/{stream_id}/plan")
    def api_plan(stream_id: str) -> JSONResponse:
        """§5.1's decision table — what would run, and why.

        Each stage carries its `pipeline_stages` status alongside the decision
        so the screen can tell "not built until Phase 2" from "failed" from
        "done", and can withhold the review link until `score` has actually
        completed rather than handing over an empty screen.
        """
        conn = connect()
        try:
            require_stream(conn, stream_id)
            engine = runner.Runner(cfg=cfg, conn=conn, stream_id=stream_id)
            entries = []
            for decision in engine.plan():
                row = engine.row(decision.stage)
                entries.append({
                    **jobs.decision_json(decision),
                    "status": row["status"] if row is not None else None,
                    "finished_at": row["finished_at"] if row is not None else None,
                    "error": row["error"] if row is not None else None,
                })
        finally:
            conn.close()

        job = app.state.jobs.for_stream(stream_id)
        return JSONResponse({
            "stream_id": stream_id,
            "stages": entries,
            "will_run": [e["stage"] for e in entries if e["will_run"]],
            "job": job.snapshot()["id"] if job is not None else None,
            "job_live": bool(job is not None and job.live),
        })

    @app.post("/api/streams/{stream_id}/run")
    def api_run(stream_id: str) -> JSONResponse:
        return _start_job(stream_id, jobs.KIND_RUN)

    @app.post("/api/streams/{stream_id}/render")
    def api_render(stream_id: str) -> JSONResponse:
        """§8's auto-finish, started from the browser.

        Same machinery as a run, and deliberately the same one-job-per-stream
        rule: rendering reads the master while a pipeline run may be rewriting
        artifacts beside it, and the log UI can only follow one job.
        """
        return _start_job(stream_id, jobs.KIND_RENDER)

    def _start_job(stream_id: str, kind: str) -> JSONResponse:
        conn = connect()
        try:
            require_stream(conn, stream_id)
        finally:
            conn.close()

        try:
            job = app.state.jobs.start(cfg, stream_id, kind)
        except jobs.JobBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(job.snapshot())

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str, since: int = 0) -> JSONResponse:
        """Polled while a run is live. `since` is an absolute line cursor."""
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        return JSONResponse(job.snapshot(since=since))

    # -- §11.6's pull search ----------------------------------------------

    @app.get("/api/search/corpus")
    def api_search_corpus() -> JSONResponse:
        """What is searchable, before anything is typed.

        A separate call from the search itself so the screen can say "nothing is
        indexed" up front rather than looking ready and explaining afterwards.
        On shipped defaults that IS the state — §5.7's transcription is off — so
        it is the common answer, not an edge case.
        """
        conn = connect()
        try:
            corpus = search.survey(conn)
            model = search.default_model(conn, cfg)
        finally:
            conn.close()
        return JSONResponse({
            "segments": corpus.segments,
            "embedded": corpus.embedded,
            "streams": corpus.streams,
            "models": corpus.models,
            "model": model,
            "searchable": corpus.searchable,
        })

    @app.get("/api/search")
    def api_search(q: str, limit: int | None = None,
                   stream: str | None = None) -> JSONResponse:
        conn = connect()
        try:
            hits, corpus, model = search.run_query(
                conn, cfg, q, limit=limit, stream_id=stream)
        except search.Unavailable as exc:
            # 503, not 500: the search did not run, and the operator can fix it
            # by starting Ollama. A generic error would send them to the logs.
            raise HTTPException(status_code=503, detail=str(exc)) from None
        except search.SearchError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        finally:
            conn.close()

        return JSONResponse({
            "query": q,
            "model": model,
            "searchable": corpus.searchable,
            "embedded": corpus.embedded,
            "hits": [hit.to_json() for hit in hits],
        })

    # -- media ------------------------------------------------------------

    @app.get("/media/{stream_id}/proxy")
    def api_proxy(stream_id: str, request: Request):
        """The proxy, range-served.

        A2's fixed GOP plus byte ranges is what makes seeking instant, which is
        what lets Phase 1 skip §7.2's pre-rendered previews entirely.
        """
        conn = connect()
        try:
            row = conn.execute(
                "SELECT proxy_path FROM streams WHERE id = ?", (stream_id,)
            ).fetchone()
        finally:
            conn.close()

        if row is None or not row["proxy_path"]:
            raise HTTPException(
                status_code=404,
                detail=f"{stream_id} has no proxy — run `clipforge run {stream_id}`",
            )

        path = paths.StreamPaths(cfg.data_root, stream_id).absolute(row["proxy_path"])
        return media.serve(path, request.headers.get("range"))

    @app.get("/media/{stream_id}/preview/{name}")
    def api_preview(stream_id: str, name: str, request: Request):
        """One of §7.2's precomputed assets.

        The name is resolved under the stream's own previews directory and the
        result is checked to still be inside it. The candidate payload only ever
        emits names this server wrote, but the route is reachable directly and
        `..` in a path segment is the oldest trick there is — loopback is not a
        security boundary here (see `review/guard.py`), so this cannot rely on
        the caller being the bundled UI.
        """
        directory = paths.StreamPaths(cfg.data_root, stream_id).previews_dir
        try:
            path = (directory / name).resolve()
            path.relative_to(directory.resolve())
        except (ValueError, OSError):
            raise HTTPException(status_code=404, detail="not found") from None
        return media.serve(path, request.headers.get("range"))

    return app


def serve(cfg, host: str | None = None, port: int | None = None, open_browser: bool = True):
    """Run the server. Binds loopback only."""
    import threading
    import webbrowser

    import uvicorn

    host = host or str(cfg.get("review.host"))
    port = int(port or cfg.get("review.port"))
    url = f"http://{host}:{port}/"

    print(f"ClipForge review  {url}")
    print(f"database          {cfg.db_path}")
    print("\nkeys: j/k move  1 skip  2 maybe  3 clip it  space play  ? signals  q finish")
    print("Ctrl-C to stop\n")

    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")
