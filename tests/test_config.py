"""Config layering, versioning, and §17 completeness."""

from __future__ import annotations

import json

import pytest
import yaml

from clipforge import config


@pytest.fixture
def cfg():
    return config.load()


# --------------------------------------------------------------------------
# §17 — every tunable must be reachable from config, never hardcoded
# --------------------------------------------------------------------------

#: Spec §17 row -> the config key that implements it. If a parameter is ever
#: dropped or renamed without updating config, this test is what catches it.
#: "Store them in config files, never hardcoded."
SPEC_17_PARAMETERS = {
    "marker_retro_offset_s": "score.markers.retro_offset_s",
    "rolling_baseline_window_s": "score.rolling_baseline_window_s",
    "min_window_s": "score.window.min_window_s",
    "max_window_s": "score.window.max_window_s",
    "hysteresis_enter": "score.window.hysteresis_enter",
    "hysteresis_exit": "score.window.hysteresis_exit",
    "spacing_penalty_window_s": "score.spacing.window_s",
    "spacing_penalty_factor": "score.spacing.factor",
    "peak_prominence_threshold": "score.peak.prominence",
    "menu_grace_period_s": "deferred.negatives.menu_grace_period_s",
    "afk_threshold_s": "deferred.negatives.afk_threshold_s",
    "combined_score_alpha": "score.combined.alpha",
    "laughter_band_hz": "deferred.laughter.band_hz",
    "hdbscan_min_cluster_size": "deferred.trends.hdbscan_min_cluster_size",
    "ngram_recency_halflife_days": "deferred.trends.ngram_recency_halflife_days",
}


@pytest.mark.parametrize(("spec_name", "key"), sorted(SPEC_17_PARAMETERS.items()))
def test_spec_17_parameter_exists(cfg, spec_name, key):
    cfg.get(key)  # raises ConfigError if absent


def test_spec_17_defaults_match_the_table(cfg):
    """The defaults are guesses, but they should be the spec's guesses."""
    assert cfg.get("score.markers.retro_offset_s") == 20.0
    assert cfg.get("score.rolling_baseline_window_s") == 300
    assert cfg.get("score.window.min_window_s") == 8
    assert cfg.get("score.window.max_window_s") == 60
    assert cfg.get("score.window.hysteresis_enter") == 0.6
    assert cfg.get("score.window.hysteresis_exit") == 0.35
    assert cfg.get("score.spacing.window_s") == 30
    assert cfg.get("score.spacing.factor") == 0.5
    assert cfg.get("score.combined.alpha") == 0.5
    assert cfg.get("deferred.negatives.menu_grace_period_s") == 8
    assert cfg.get("deferred.negatives.afk_threshold_s") == 60
    assert cfg.get("deferred.laughter.band_hz") == [4.0, 7.0]
    assert cfg.get("deferred.trends.hdbscan_min_cluster_size") == 5
    assert cfg.get("deferred.trends.ngram_recency_halflife_days") == 30


def test_final_profile_weights_match_the_spec(cfg):
    """§6.5 entertainment weights for the signals Phase 1 actually computes."""
    assert cfg.profile.weights == {
        "marker_definite": 3.0,
        "marker_maybe": 1.5,
        "mic_rms": 1.0,
    }
    assert cfg.profile.weight("laugh_party") == 0.0  # not in this profile yet


def test_a2_proxy_gop_settings(cfg):
    assert cfg.get("ingest.proxy.gop") == 30
    assert cfg.get("ingest.proxy.keyint_min") == 30
    assert cfg.get("ingest.proxy.sc_threshold") == 0
    # Not in §5.2; without it the "identical timebase" claim is false for VFR.
    assert cfg.get("ingest.proxy.force_cfr") is True


def test_audio_extraction_is_16k_mono(cfg):
    """§5.3: 'Do not extract at higher rates.'"""
    assert cfg.get("ingest.audio.sample_rate_hz") == 16000
    assert cfg.get("ingest.audio.channels") == 1


# --------------------------------------------------------------------------
# layering and provenance
# --------------------------------------------------------------------------


def test_defaults_layer_is_recorded(cfg):
    assert cfg.layers[0][0] == "default"
    assert cfg.provenance["score.window.min_window_s"] == "default"


def test_explicit_file_overrides_defaults(tmp_path):
    override = tmp_path / "over.yaml"
    override.write_text(
        yaml.safe_dump({"score": {"window": {"min_window_s": 3}}}), encoding="utf-8"
    )
    cfg = config.load(config_file=override)
    assert cfg.get("score.window.min_window_s") == 3
    assert cfg.provenance["score.window.min_window_s"] == "file"
    # Siblings survive the merge.
    assert cfg.get("score.window.max_window_s") == 60
    assert cfg.provenance["score.window.max_window_s"] == "default"


def test_set_flag_wins_over_file(tmp_path):
    override = tmp_path / "over.yaml"
    override.write_text(
        yaml.safe_dump({"score": {"window": {"min_window_s": 3}}}), encoding="utf-8"
    )
    cfg = config.load(config_file=override, overrides=["score.window.min_window_s=11"])
    assert cfg.get("score.window.min_window_s") == 11
    assert cfg.provenance["score.window.min_window_s"] == "flag"


def test_env_config_is_honoured(tmp_path, monkeypatch):
    override = tmp_path / "env.yaml"
    override.write_text(yaml.safe_dump({"review": {"port": 9999}}), encoding="utf-8")
    monkeypatch.setenv(config.ENV_CONFIG, str(override))
    assert config.load().get("review.port") == 9999


def test_missing_config_file_is_an_error(tmp_path):
    with pytest.raises(config.ConfigError, match="not found"):
        config.load(config_file=tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a.b=1", ("a.b", 1)),
        ("a.b=1.5", ("a.b", 1.5)),
        ("a.b=true", ("a.b", True)),
        ("a.b=null", ("a.b", None)),
        ("a.b=hello", ("a.b", "hello")),
        ("a.b=[1, 2]", ("a.b", [1, 2])),
    ],
)
def test_override_parsing_preserves_yaml_types(text, expected):
    assert config.parse_override(text) == expected


def test_override_without_equals_is_rejected():
    with pytest.raises(config.ConfigError, match="key.path=value"):
        config.parse_override("nonsense")


# --------------------------------------------------------------------------
# config_version
# --------------------------------------------------------------------------


def test_version_is_profile_at_hash(cfg):
    name, _, digest = cfg.version.partition("@")
    assert name == "naive"
    assert len(digest) == 8
    assert digest == digest.lower()


def test_version_is_stable_across_loads():
    assert config.load().version == config.load().version


def test_version_ignores_numeric_formatting(tmp_path):
    """YAML 8 and 8.0 are the same configuration, not two versions."""
    same = tmp_path / "same.yaml"
    same.write_text(yaml.safe_dump({"score": {"window": {"min_window_s": 8.0}}}), encoding="utf-8")
    assert config.load(config_file=same).version == config.load().version


def test_version_ignores_irrelevant_keys(tmp_path):
    """The review port cannot change what a scoring run produces."""
    noise = tmp_path / "noise.yaml"
    noise.write_text(yaml.safe_dump({"review": {"port": 1234}}), encoding="utf-8")
    assert config.load(config_file=noise).version == config.load().version


def test_version_changes_with_a_scoring_parameter():
    changed = config.load(overrides=["score.window.min_window_s=9"])
    assert changed.version != config.load().version


def test_version_changes_with_an_extraction_parameter():
    """Signals feed scoring, so a different hop means different candidates."""
    changed = config.load(overrides=["extract.rms.hop_s=0.25"])
    assert changed.version != config.load().version


def test_version_changes_with_a_weight(tmp_path, monkeypatch):
    """The whole point: retuning weights produces a distinguishable version."""
    baseline = config.load().version

    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "naive.yaml").write_text(
        yaml.safe_dump(
            {"name": "naive", "weights": {"mic_rms": 2.0, "marker_definite": 3.0}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROFILES_DIR", profile_dir)

    retuned = config.load().version
    assert retuned != baseline
    # Still the same profile name — only the hash moves.
    assert retuned.startswith("naive@")


def test_canonical_json_is_key_order_independent():
    a = config.canonical_json({"b": 1, "a": {"d": 2, "c": 3}})
    b = config.canonical_json({"a": {"c": 3, "d": 2}, "b": 1})
    assert a == b


def test_canonical_json_keeps_bools_distinct_from_ints():
    assert config.canonical_json(True) != config.canonical_json(1)


# --------------------------------------------------------------------------
# profiles and feature schema
# --------------------------------------------------------------------------


def test_unknown_profile_lists_the_available_ones():
    with pytest.raises(config.ConfigError, match="naive"):
        config.load(profile="does-not-exist")


def test_profile_with_non_numeric_weight_is_rejected(tmp_path):
    directory = tmp_path / "profiles"
    directory.mkdir()
    (directory / "bad.yaml").write_text(
        yaml.safe_dump({"name": "bad", "weights": {"mic_rms": "loud"}}), encoding="utf-8"
    )
    with pytest.raises(config.ConfigError, match="non-numeric"):
        config.load_profile("bad", directory)


def test_feature_schema_covers_every_spec_signal(cfg):
    """A9 vectors are only comparable across phases if the key set is fixed."""
    schema = cfg.feature_schema
    expected = {
        # §5.4.1 continuous
        "mic_rms", "mic_f0", "mic_f0_variance", "speech_rate", "sudden_silence",
        "game_rms", "party_rms", "party_f0", "overlap_speech", "input_rate",
        "mouse_velocity", "input_stillness",
        # §5.6 mentions swearing density as a continuous signal
        "swear_density",
        # §5.4.2 discrete
        "marker_maybe", "marker_definite", "laugh_operator", "laugh_party",
        "phrase_excitement", "phrase_repeat", "kill", "multikill", "clutch",
        "mvp_screen", "ult_used", "headshot", "scene_change", "menu_screen", "afk",
        # §5.4.3 composite
        "flick_signature", "multikill_with_ult", "reaction_onset",
    }
    assert set(schema.keys) == expected


def test_empty_vector_is_full_width_and_null(cfg):
    vector = cfg.feature_schema.empty_vector()
    assert set(vector) == set(cfg.feature_schema.keys)
    assert all(value is None for value in vector.values())


def test_phase_1_signals_are_marked_as_such(cfg):
    phase_1 = {k for k, v in cfg.feature_schema.signals.items() if v.get("phase") == 1}
    assert phase_1 == {"mic_rms", "marker_maybe", "marker_definite"}


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


def test_relative_paths_resolve_against_project_root(cfg):
    assert cfg.data_root.is_absolute()
    assert cfg.data_root == cfg.project_root / "data"


def test_project_root_is_found_from_a_subdirectory(cfg):
    from_sub = config.find_project_root(cfg.project_root / "clipforge" / "score")
    assert from_sub == cfg.project_root


def test_absolute_data_root_is_left_alone(tmp_path):
    override = tmp_path / "abs.yaml"
    target = tmp_path / "elsewhere"
    override.write_text(yaml.safe_dump({"paths": {"data_root": str(target)}}), encoding="utf-8")
    assert config.load(config_file=override).data_root == target.resolve()


def test_db_path_defaults_under_data_root(cfg):
    assert cfg.db_path == cfg.data_root / "clipforge.db"


def test_explicit_db_path_wins(tmp_path):
    override = tmp_path / "db.yaml"
    override.write_text(
        yaml.safe_dump({"paths": {"db": str(tmp_path / "other.db")}}), encoding="utf-8"
    )
    assert config.load(config_file=override).db_path == (tmp_path / "other.db").resolve()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_config_show_reports_provenance(capsys):
    from clipforge import cli

    assert cli.main(["config", "show", "--set", "review.port=4321"]) == 0
    out = capsys.readouterr().out
    assert "config_version" in out
    assert "review.port" in out
    assert "flag" in out


def test_config_version_action_prints_only_the_version(capsys):
    from clipforge import cli

    assert cli.main(["config", "version"]) == 0
    assert capsys.readouterr().out.strip() == config.load().version


def test_config_json_is_parseable(capsys):
    from clipforge import cli

    assert cli.main(["config", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config_version"] == config.load().version
    assert payload["profile"]["name"] == "naive"
    assert payload["resolved"]["data_root"]
