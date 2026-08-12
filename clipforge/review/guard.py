"""Who is allowed to talk to this server.

§2.2 says "local web app (FastAPI + vanilla JS)" and nothing else about
serving, which is fine right up to the point where the routes stop being
read-only. A file browser and a "run the pipeline" trigger are reachable by
**every page open in the operator's browser**: binding to loopback keeps other
machines out, not other tabs.

Two vectors, needing two different answers.

**The cross-origin simple POST.** A hostile page can send
`POST /api/streams/x/run` with `Content-Type: text/plain` and no preflight at
all, and FastAPI's `await request.json()` parses the body regardless of the
declared type — so the run happens even though the reply cannot be read.
Requiring a custom header forces a CORS preflight, which that page cannot pass.
GETs do not need the header: a cross-origin page can issue one, but cannot read
the response without CORS headers this server never sends.

**DNS rebinding.** A name that resolves first to the attacker's address and then
to 127.0.0.1 makes their page genuinely same-origin, at which point the header
is sent willingly and `Origin` looks correct. The only thing that still
separates them is the `Host` they had to ask for, so that is checked on every
request.

None of this makes the server safe to expose deliberately. It makes it safe to
have running while the operator browses the web, which is the actual threat.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from starlette.responses import PlainTextResponse

#: Header a request must carry to change anything. The value is never checked —
#: its presence is the whole mechanism, because sending it at all is what a
#: cross-origin page cannot do without a preflight.
HEADER = "x-clipforge"

#: Always allowed, whatever `review.host` is set to.
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

#: Methods that cannot change anything, and so do not need the header.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _hostname(value: str) -> str:
    """The host part of a `Host` header or an `Origin`, without its port.

    `urlsplit` needs a scheme to populate `.hostname`, and bracketed IPv6 hosts
    must not be split on their colons — hence going through it rather than
    partitioning on ':'.
    """
    value = value.strip()
    if "//" not in value:
        value = "//" + value
    try:
        return (urlsplit(value).hostname or "").lower()
    except ValueError:
        return ""


def allowed_hosts(cfg) -> frozenset[str]:
    names = set(LOOPBACK)
    configured = cfg.get("review.host", None)
    if configured:
        names.add(_hostname(str(configured)))
    for extra in cfg.get("review.allow_hosts", None) or []:
        names.add(_hostname(str(extra)))
    return frozenset(n for n in names if n)


def check(cfg, method: str, headers) -> str | None:
    """The reason to refuse this request, or None to let it through."""
    permitted = allowed_hosts(cfg)

    host = _hostname(headers.get("host", ""))
    if host not in permitted:
        return (
            f"Host {host or '(missing)'!r} is not one this server answers to. "
            f"ClipForge serves {', '.join(sorted(permitted))} only — a request "
            f"arriving under another name is how DNS rebinding reaches a local "
            f"server. Set review.allow_hosts if you meant this."
        )

    origin = headers.get("origin")
    if origin and origin.lower() != "null" and _hostname(origin) not in permitted:
        return f"cross-origin request from {origin!r} refused"

    if method.upper() not in SAFE_METHODS and cfg.get("review.require_header", True):
        if HEADER not in headers:
            return (
                f"missing {HEADER} header. Requests that change something must send "
                f"it, which is what stops another page in your browser from posting "
                f"here without your knowledge."
            )

    return None


def install(app, cfg) -> None:
    """Add the check to every request this app serves."""

    @app.middleware("http")
    async def _guard(request, call_next):
        refusal = check(cfg, request.method, request.headers)
        if refusal is not None:
            return PlainTextResponse(refusal, status_code=403)
        return await call_next(request)
