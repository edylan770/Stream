"""Transaction semantics.

These exist because the sqlite3 module's default transaction handling is
surprising, and every extraction stage in this pipeline depends on getting it
right: A10 requires that a crash mid-stage never leaves work that looks
complete.
"""

from __future__ import annotations

import sqlite3

import pytest

from clipforge import db


@pytest.fixture
def conn(tmp_path):
    connection = db.open_db(tmp_path / "tx.db")
    yield connection
    connection.close()


def test_no_implicit_transaction_is_opened(conn):
    conn.execute("INSERT INTO ideas (kind, title, premise) VALUES ('bit', 't', 'p')")
    assert not conn.in_transaction


def test_writes_outside_a_transaction_are_durable_immediately(tmp_path):
    path = tmp_path / "durable.db"
    writer = db.open_db(path)
    writer.execute("INSERT INTO ideas (kind, title, premise) VALUES ('bit', 't', 'p')")
    # No commit() call. A second connection must still see it.
    reader = db.connect(path)
    assert reader.execute("SELECT count(*) FROM ideas").fetchone()[0] == 1
    reader.close()
    writer.close()


def test_transaction_commits_on_success(conn):
    with db.transaction(conn):
        conn.execute("INSERT INTO ideas (kind, title, premise) VALUES ('bit', 'a', 'p')")
        conn.execute("INSERT INTO ideas (kind, title, premise) VALUES ('bit', 'b', 'p')")
    assert conn.execute("SELECT count(*) FROM ideas").fetchone()[0] == 2
    assert not conn.in_transaction


def test_transaction_rolls_back_every_write_on_failure(conn):
    with pytest.raises(sqlite3.IntegrityError):
        with db.transaction(conn):
            conn.execute("INSERT INTO ideas (kind, title, premise) VALUES ('bit', 'a', 'p')")
            conn.execute("INSERT INTO ideas (id, kind, title, premise) VALUES (1, 'b', 'c', 'd')")
    assert conn.execute("SELECT count(*) FROM ideas").fetchone()[0] == 0
    assert not conn.in_transaction


def test_transaction_can_be_reentered_after_a_rollback(conn):
    """A failed stage must not poison the connection for the retry."""
    with pytest.raises(RuntimeError):
        with db.transaction(conn):
            conn.execute("INSERT INTO ideas (kind, title, premise) VALUES ('bit', 'a', 'p')")
            raise RuntimeError("stage crashed")

    with db.transaction(conn):
        conn.execute("INSERT INTO ideas (kind, title, premise) VALUES ('bit', 'b', 'p')")
    assert conn.execute("SELECT count(*) FROM ideas").fetchone()[0] == 1


def test_failed_migration_leaves_no_partial_schema(tmp_path, monkeypatch):
    """Half a schema with user_version still 0 is the worst possible outcome."""
    broken = tmp_path / "migrations"
    broken.mkdir()
    (broken / "0001_broken.sql").write_text(
        "CREATE TABLE good (id INTEGER);\nCREATE TABLE bad (id INTEGER;\n", encoding="utf-8"
    )
    monkeypatch.setattr(db, "MIGRATIONS_DIR", broken)

    conn = db.connect(tmp_path / "broken.db")
    with pytest.raises(db.MigrationError, match="0001_broken.sql"):
        db.migrate(conn)

    assert db.user_version(conn) == 0
    assert "good" not in db.table_names(conn)
    conn.close()


def test_migration_numbering_gaps_are_rejected(tmp_path, monkeypatch):
    gapped = tmp_path / "migrations"
    gapped.mkdir()
    (gapped / "0001_a.sql").write_text("SELECT 1;", encoding="utf-8")
    (gapped / "0003_c.sql").write_text("SELECT 1;", encoding="utf-8")
    monkeypatch.setattr(db, "MIGRATIONS_DIR", gapped)
    with pytest.raises(db.MigrationError, match="no gaps"):
        db.migration_files()
