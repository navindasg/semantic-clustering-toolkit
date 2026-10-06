from datetime import timedelta

import pytest
from pydantic import ValidationError

from toolkit.config import MODEL_PROFILES, Config, format_duration, load_config, parse_duration


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("7d", timedelta(days=7)),
        ("1h30m", timedelta(hours=1, minutes=30)),
        ("90s", timedelta(seconds=90)),
        ("2w", timedelta(weeks=2)),
        ("3600", timedelta(hours=1)),
        (120, timedelta(minutes=2)),
        ("P7D", timedelta(days=7)),
        ("PT5M", timedelta(minutes=5)),
        ("P1DT2H", timedelta(days=1, hours=2)),
        (timedelta(hours=3), timedelta(hours=3)),
    ],
)
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize("bad", ["", "7x", "abc", "1d junk", "P"])
def test_parse_duration_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_duration(bad)


def test_format_duration_round_trips():
    for text in ("7d", "14d", "2h", "15m", "45s"):
        assert format_duration(parse_duration(text)) == text


def test_defaults_match_prd():
    cfg = Config(embedder="hashing:64")
    assert cfg.exemplars_per_cluster == 32
    assert cfg.match_top_k == 5
    assert cfg.threshold_percentile == 5
    assert cfg.threshold_bounds == (0.55, 0.90)
    assert cfg.close_after == timedelta(days=7)
    assert cfg.effective_reopen_grace == timedelta(days=14)
    assert cfg.merge_threshold == 0.85
    assert cfg.random_seed == 42


def test_model_profile_applies_only_when_unset():
    default = Config()
    profile = MODEL_PROFILES[default.embedder]
    assert default.threshold_bounds == profile["threshold_bounds"]
    explicit = Config(threshold_bounds=(0.6, 0.8))
    assert explicit.threshold_bounds == (0.6, 0.8)
    assert explicit.merge_threshold == profile["merge_threshold"]


def test_with_overrides_reapplies_profile_for_new_embedder():
    switched = Config().with_overrides(embedder="hashing:64")
    assert switched.threshold_bounds == (0.55, 0.90)
    assert switched.merge_threshold == 0.85


def test_with_overrides_keeps_explicit_values():
    cfg = Config(embedder="hashing:64", close_after="2d").with_overrides(min_cluster_size=7)
    assert cfg.close_after == timedelta(days=2)
    assert cfg.min_cluster_size == 7


def test_reopen_grace_explicit():
    assert Config(close_after="1d", reopen_grace="3d").effective_reopen_grace == timedelta(days=3)


@pytest.mark.parametrize("bounds", [(0.9, 0.5), (0, 0.5), (0.5, 1.5)])
def test_invalid_bounds(bounds):
    with pytest.raises(ValidationError):
        Config(threshold_bounds=bounds)


def test_bounds_from_string():
    assert Config(threshold_bounds="0.6 to 0.8").threshold_bounds == (0.6, 0.8)


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        Config(not_a_field=1)


def test_load_config_layers(tmp_path):
    path = tmp_path / "toolkit.yaml"
    path.write_text("close_after: 3d\nmin_cluster_size: 8\ncolumns:\n  text: body\n")
    cfg = load_config(path, environ={"TOOLKIT_MIN_CLUSTER_SIZE": "12", "OTHER": "x"}, match_top_k=3)
    assert cfg.close_after == timedelta(days=3)
    assert cfg.min_cluster_size == 12
    assert cfg.match_top_k == 3
    assert cfg.columns.text == "body"


def test_load_config_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml", environ={})
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ValueError):
        load_config(bad, environ={})


def test_example_config_loads():
    from pathlib import Path

    cfg = load_config(Path(__file__).parent.parent / "toolkit.example.yaml", environ={})
    assert cfg.embedder == "fastembed:BAAI/bge-small-en-v1.5"
    assert cfg.threshold_bounds == MODEL_PROFILES[cfg.embedder]["threshold_bounds"]
    assert cfg.effective_reopen_grace == timedelta(days=14)
