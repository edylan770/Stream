"""The written rubric — storage, versioning, and the seam into prompts.

The rubric has no § reference: the spec never gave the learning layer a
section. So unlike most of this suite there is no spec text to assert against,
and what is tested instead is the set of properties that make it safe to keep
forever — append-only, versioned, and structurally unable to reach the scoring
path.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from clipforge import config, db, rubric
from clipforge.llm import LLMError, prompts

SCORE_DIR = Path(__file__).resolve().parent.parent / "clipforge" / "score"


@pytest.fixture
def conn(tmp_path):
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'data').as_posix()}"])
    connection = db.open_db(cfg.db_path)
    yield cfg, connection
    connection.close()


def _write(connection, text):
    with db.transaction(connection):
        return rubric.write(connection, text)


# --------------------------------------------------------------------------
# append-only, and the version is the identity
# --------------------------------------------------------------------------


def test_versions_increment_and_nothing_is_overwritten(conn):
    """§9.1's reasoning, one table over: a digest made under v1 has to stay
    explicable once v2 exists, so v1's text may never change."""
    _, connection = conn
    first = _write(connection, "silences before a punchline")
    second = _write(connection, "silences, and her reaction over mine")

    assert (first.version, second.version) == (1, 2)
    assert rubric.get(connection, 1).text == "silences before a punchline"
    assert rubric.current(connection).version == 2
    assert [entry.version for entry in rubric.versions(connection)] == [1, 2]


def test_there_is_no_way_to_edit_or_delete_a_version():
    """Asserted on the module's surface rather than by trying it: the absence
    of the verb is the guarantee, and a test that only checked behaviour would
    pass a module that grew an `update()` tomorrow."""
    assert not hasattr(rubric, "update")
    assert not hasattr(rubric, "delete")
    assert not hasattr(rubric, "replace")


def test_an_empty_rubric_is_refused(conn):
    """Writing nothing is not a version, it is a deletion wearing one."""
    _, connection = conn
    with pytest.raises(ValueError):
        _write(connection, "   \n  ")


def test_the_corpus_behind_a_version_is_recorded_at_write_time(conn):
    """So a rubric written after four streams is legible as such a year later,
    rather than reading as considered advice."""
    _, connection = conn
    connection.execute(
        "INSERT INTO streams (id, date, master_path) VALUES ('s', '2026-08-14', 'm.mkv')")
    entry = _write(connection, "written from one stream")

    assert entry.n_streams_at_write == 1
    assert entry.n_ratings_at_write == 0
    assert "1 stream(s)" in entry.evidence


def test_only_operator_ratings_count_towards_the_corpus(conn):
    """The reading `clipforge/moments.py` uses: an inherited copy is not a
    second judgement, and after commit 43 the operator's own row routinely
    lives on a superseded generation — so `is_current` is not consulted."""
    from tests.conftest import hand_rows

    _, connection = conn
    connection.execute(
        "INSERT INTO streams (id, date, master_path) VALUES ('s', '2026-08-14', 'm.mkv')")
    hand_rows(connection, "s", [
        (1, 0, 0.0, 10.0, 2, "2026-08-14 10:00:00", "operator"),
        (2, 1, 0.0, 10.0, 2, "2026-08-14 10:01:00", "inherited"),
    ])

    assert _write(connection, "one judgement, two rows").n_ratings_at_write == 1


# --------------------------------------------------------------------------
# the seam into prompts
# --------------------------------------------------------------------------


def test_an_absent_rubric_still_yields_a_usable_sentence(conn):
    """`Template.substitute` raises on anything unsupplied, so `$rubric` always
    needs a value — and an empty one leaves a blank section under a heading,
    which reads as though something went missing and invites a model to invent
    guidance that is not there."""
    _, connection = conn
    assert rubric.for_prompt(connection) == rubric.ABSENT
    assert rubric.ABSENT.strip()


def test_for_prompt_returns_the_newest_text(conn):
    _, connection = conn
    _write(connection, "older")
    _write(connection, "newer")
    assert rubric.for_prompt(connection) == "newer"


def test_a_prompt_can_carry_a_rubric_placeholder(tmp_path):
    """Proved against a TEMP prompt file rather than by adding `$rubric` to the
    shipped hook prompt, which commit 42 went to some trouble to leave
    byte-identical to commit 41's."""
    prompt_file = tmp_path / "p.yaml"
    prompt_file.write_text(
        "prompts:\n  demo: |\n    Guidance from the operator:\n    $rubric\n",
        encoding="utf-8")
    cfg = config.load(overrides=[f"llm.prompts.file={prompt_file.as_posix()}"])

    rendered = prompts.render(cfg, "demo", rubric="her reaction over mine")
    assert "her reaction over mine" in rendered
    assert "$rubric" not in rendered


def test_a_prompt_asking_for_a_rubric_that_is_not_supplied_raises(tmp_path):
    """The strictness commit 42 chose deliberately: an unfilled `$name` must
    never survive into text a model then reads."""
    prompt_file = tmp_path / "p.yaml"
    prompt_file.write_text("prompts:\n  demo: |\n    $rubric\n", encoding="utf-8")
    cfg = config.load(overrides=[f"llm.prompts.file={prompt_file.as_posix()}"])

    with pytest.raises(LLMError):
        prompts.render(cfg, "demo")


def test_the_shipped_prompts_do_not_use_a_rubric_yet():
    """45a builds the seam and wires nothing to it. A prompt quietly gaining
    `$rubric` here would break `render/hooks.py`, whose caller supplies four
    named values and no more."""
    for name, text in prompts.load(config.load()).items():
        assert "$rubric" not in text, name


# --------------------------------------------------------------------------
# it must never reach scoring
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module", sorted(p.name for p in SCORE_DIR.glob("*.py")))
def test_nothing_in_score_imports_the_rubric(module):
    """THE structural guarantee. Scoring is deterministic (C1) and §6.1 promises
    re-scoring is free and infinitely repeatable. A rubric reaching that path
    would make every re-score depend on prose `config_version` cannot describe,
    because the text lives in a table rather than in config.

    An AST walk rather than a habit, the shape `test_capture.py` uses for §2.1's
    independence rule."""
    # utf-8-sig, not utf-8: some modules here carry a BOM, and `ast.parse` of
    # already-decoded text rejects U+FEFF as a non-printable character.
    tree = ast.parse((SCORE_DIR / module).read_text(encoding="utf-8-sig"))
    reached = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            reached.append(node.module)
        elif isinstance(node, ast.Import):
            reached.extend(alias.name for alias in node.names)

    for name in reached:
        assert "rubric" not in name, f"{module} imports {name}, which reaches scoring"


def test_editing_the_rubric_subtree_cannot_invalidate_a_candidate():
    from clipforge.config import VERSIONED_SUBTREES

    assert "rubric" not in VERSIONED_SUBTREES
    base = config.load()
    moved = config.load(overrides=["rubric.warn_chars=1", "rubric.export_dir=./elsewhere"])
    assert base.version == moved.version


# --------------------------------------------------------------------------
# size: warned about, never truncated
# --------------------------------------------------------------------------


def test_an_oversized_rubric_is_reported_and_left_whole(conn):
    """Every downstream prompt carries this text, so its size is worth knowing
    — but silently dropping the end of the operator's own judgement is the
    exact failure the rubric exists to prevent."""
    cfg, connection = conn
    limit = int(cfg.get("rubric.warn_chars"))
    entry = _write(connection, "x" * (limit + 25))

    assert rubric.oversized(entry, cfg) == 25
    assert len(rubric.for_prompt(connection)) == limit + 25


def test_a_rubric_within_the_limit_reports_nothing(conn):
    cfg, connection = conn
    entry = _write(connection, "short")
    assert rubric.oversized(entry, cfg) == 0
    assert rubric.oversized(None, cfg) == 0


# --------------------------------------------------------------------------
# §13.2's tier 2
# --------------------------------------------------------------------------


def test_the_markdown_mirror_carries_every_version_and_its_evidence(conn):
    """"Human-readable, survives any schema change or full application
    rewrite." A mirror holding only the current version would not."""
    _, connection = conn
    _write(connection, "first thoughts")
    _write(connection, "second thoughts")

    text = rubric.to_markdown(rubric.versions(connection))
    assert "## v1" in text and "## v2" in text
    assert "first thoughts" in text and "second thoughts" in text
    assert "0 stream(s)" in text


# --------------------------------------------------------------------------
# the three gaps closed in the same migration
# --------------------------------------------------------------------------


def test_one_export_can_now_carry_two_platforms(conn):
    """§3.2 keyed `performance` on `export_id` alone, so a clip posted to
    Shorts and TikTok could record one result and not the other. The unit of an
    export here is a rendered clip, and the three presets differ only by
    `max_duration_s`."""
    _, connection = conn
    connection.execute(
        "INSERT INTO streams (id, date, master_path) VALUES ('s', '2026-08-14', 'm.mkv')")
    connection.execute(
        "INSERT INTO exports (stream_id, kind, path) VALUES ('s', 'clip', 'a.mp4')")
    export_id = connection.execute("SELECT id FROM exports").fetchone()[0]

    with db.transaction(connection):
        connection.executemany(
            "INSERT INTO performance (export_id, platform, views) VALUES (?, ?, ?)",
            [(export_id, "shorts", 10), (export_id, "tiktok", 20)],
        )

    assert connection.execute("SELECT COUNT(*) FROM performance").fetchone()[0] == 2


def test_the_same_export_and_platform_twice_is_still_refused(conn):
    """The key still means something: one measurement per clip per platform."""
    import sqlite3

    _, connection = conn
    connection.execute(
        "INSERT INTO streams (id, date, master_path) VALUES ('s', '2026-08-14', 'm.mkv')")
    connection.execute(
        "INSERT INTO exports (stream_id, kind, path) VALUES ('s', 'clip', 'a.mp4')")
    export_id = connection.execute("SELECT id FROM exports").fetchone()[0]

    connection.execute(
        "INSERT INTO performance (export_id, platform) VALUES (?, 'shorts')", (export_id,))
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO performance (export_id, platform) VALUES (?, 'shorts')", (export_id,))


def test_an_open_loop_can_record_the_segment_its_quote_came_from(conn):
    """§9.2's JSON gives every loop a `segment_id` and the table had nowhere to
    put it, so §12.3's verbatim-quote check ran at ingest and then discarded its
    own evidence."""
    _, connection = conn
    connection.execute(
        "INSERT INTO streams (id, date, master_path) VALUES ('s', '2026-08-14', 'm.mkv')")
    connection.execute(
        "INSERT INTO segments (stream_id, seq, t_start, t_end, text) "
        "VALUES ('s', 1, 0.0, 2.0, 'I will try Namor next game')")
    segment_id = connection.execute("SELECT id FROM segments").fetchone()[0]

    connection.execute(
        "INSERT INTO open_loops (stream_id, t, text, kind, segment_id) "
        "VALUES ('s', 1.0, 'I will try Namor next game', 'promise', ?)", (segment_id,))

    row = connection.execute("SELECT segment_id FROM open_loops").fetchone()
    assert row["segment_id"] == segment_id


def test_a_digest_can_record_which_prompt_and_rubric_produced_it(conn):
    """§9.1 keeps digests forever and never regenerates them, so two made under
    two prompts — or two rubrics — must be distinguishable in origin.
    `prompts.digest_of` has existed since commit 42 with nowhere to store it."""
    _, connection = conn
    connection.execute(
        "INSERT INTO streams (id, date, master_path) VALUES ('s', '2026-08-14', 'm.mkv')")
    connection.execute(
        "INSERT INTO digests (stream_id, version, content, prompt_hash, rubric_version) "
        "VALUES ('s', 1, '{}', 'abc123def456', 3)")

    row = connection.execute(
        "SELECT prompt_hash, rubric_version FROM digests").fetchone()
    assert (row["prompt_hash"], row["rubric_version"]) == ("abc123def456", 3)
