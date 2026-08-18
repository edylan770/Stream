"""`clipforge search "that time I did the sad voice"` — §11.6's pull search.

> "Keyword search cannot do this; the operator remembers the vibe, not the
> words."

The command exists alongside the review UI's search screen because the two get
used at different moments: the screen is for browsing back into a stream, and
this is for answering "did I ever..." from a terminal without opening anything.

**An empty result explains itself.** Three different things produce zero rows and
they need three different sentences — nothing is indexed, Ollama is not running,
or the search ran fine and this library genuinely has no such moment. Printing
"no results" for all three is how a broken search looks healthy, which matters
more here than usual because a library processed with shipped defaults has NO
embeddings at all: §5.7's transcript ships `extract.whisperx.enabled: false`.
"""

from __future__ import annotations

import textwrap

from clipforge import config, db, search


def add_arguments(parser) -> None:
    parser.add_argument("query", nargs="+", help="free text, in your own words")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="results to show (default: search.limit)",
    )
    parser.add_argument(
        "--stream", metavar="ID",
        help="search one stream instead of the whole library",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="print each segment in full rather than a snippet",
    )
    config.add_config_arguments(parser)


def main(args) -> int:
    cfg = config.from_args(args)
    if not cfg.db_path.exists():
        print(f"no database at {cfg.db_path}")
        return 1

    query = " ".join(args.query)
    conn = db.open_db(cfg.db_path, migrate_to_latest=False)
    try:
        try:
            hits, corpus, model = search.run_query(
                conn, cfg, query, limit=args.limit, stream_id=args.stream)
        except search.Unavailable as exc:
            print(exc)
            return 1
        except search.SearchError as exc:
            print(exc)
            return 1

        return _report(cfg, args, query, hits, corpus, model)
    finally:
        conn.close()


def _report(cfg, args, query, hits, corpus, model) -> int:
    if not corpus.searchable:
        print(f"nothing is indexed, so there is nothing to search for {query!r}.")
        print()
        if corpus.segments:
            print(f"  {corpus.segments} transcript segment(s) exist but none has an "
                  f"embedding.")
            print("  Run the `embeddings` stage: clipforge run <stream_id>")
        else:
            print("  No transcript segments at all. §5.7's transcription ships OFF")
            print("  (`extract.whisperx.enabled: false`) because it costs a multi-GB")
            print("  download; §11.6's search is one of the things turning it on buys.")
        return 1

    print(f'"{query}"  —  {len(hits)} of {corpus.embedded} embedded segment(s) '
          f'across {corpus.streams} stream(s), model {model}')
    print()

    if not hits:
        print("  Nothing matched. The index is fine; this library has no such moment.")
        return 0

    width = int(cfg.get("search.snippet_chars", 240))
    for rank, hit in enumerate(hits, start=1):
        where = f"{hit.stream_id} @ {_clock(hit.t_start)}"
        # The score is printed because it is the only ordering information there
        # is, NOT because a number is meaningful on its own: the measured band
        # is 0.345-0.610 and nothing should be compared against a threshold.
        print(f"{rank:>3}. {hit.score:.3f}  {where}"
              + (f"  [candidate {hit.candidate_id}]" if hit.candidate_id
                 else "  (no candidate covers this)"))
        text = hit.text.strip()
        body = text if args.full else textwrap.shorten(text, width=width,
                                                       placeholder=" …")
        for line in textwrap.wrap(body, width=88,
                                  initial_indent=" " * 6, subsequent_indent=" " * 6):
            print(line)
        print()

    print("Scores rank; they do not measure. The measured spread across unrelated")
    print("pairs is 0.345-0.610, so a low number here is not evidence of a bad match.")
    return 0


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
