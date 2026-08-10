"""Shared fixtures.

The config layer deliberately reads a machine-local `local.yaml` and
`$CLIPFORGE_CONFIG`. That is correct at runtime and poison in tests: the suite
would pass or fail depending on what the developer happens to have on disk.
Every test therefore runs against the packaged defaults unless it says
otherwise.
"""

from __future__ import annotations

import pytest

from clipforge import config


@pytest.fixture(autouse=True)
def hermetic_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOCAL_FILE", tmp_path / "no-such-local.yaml")
    monkeypatch.delenv(config.ENV_CONFIG, raising=False)
