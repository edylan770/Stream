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
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from clipforge import db, paths
from clipforge.review import media, queries

STATIC = Path(__file__).parent / "static"


def create_app(cfg) -> FastAPI:
    app = FastAPI(title="ClipForge", docs_url=None, redoc_url=None)
    app.state.cfg = cfg

    def connect() -> sqlite3.Connection:
        return db.open_db(cfg.db_path, migrate_to_latest=False)

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
            found = queries.load_candidates(conn, stream_id)
            return JSONResponse({
                "stream": detail,
                "candidates": [c.to_json() for c in found],
            })
        finally:
            conn.close()

    @app.get("/api/streams/{stream_id}/metrics")
    def api_metrics(stream_id: str) -> JSONResponse:
        conn = connect()
        try:
            return JSONResponse(queries.review_metrics(conn, stream_id))
        finally:
            conn.close()

    # -- writes -----------------------------------------------------------

    @app.post("/api/candidates/{candidate_id}/rating")
    async def api_rate(candidate_id: int, request: Request) -> JSONResponse:
        body = await request.json()
        conn = connect()
        try:
            with db.transaction(conn):
                queries.save_rating(
                    conn, candidate_id,
                    rating=int(body["rating"]),
                    review_ms=body.get("review_ms"),
                    note=body.get("note"),
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            conn.close()
        return JSONResponse({"ok": True})

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
                )
        finally:
            conn.close()
        return JSONResponse({"ok": True})

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
