import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from toolkit.config import ColumnMap, PreprocessConfig
from toolkit.io import InputError, parse_timestamp, read_items
from toolkit.labeling import CTfidfLabeler, make_labeler
from toolkit.labeling.llm import LLMLabeler
from toolkit.preprocess import build_preprocessor, normalize_text, redact_pii

DOCS = {
    "a": ["printer paper jam", "paper jam in the printer tray", "printer toner empty"],
    "b": ["wifi signal drops", "router wifi keeps dropping", "no wifi signal upstairs"],
}


def test_ctfidf_picks_distinctive_terms():
    labels = CTfidfLabeler(top_n=3).label({"a": DOCS["a"]}, DOCS)
    assert "printer" in labels["a"] or "paper" in labels["a"]
    assert "wifi" not in labels["a"]


def test_ctfidf_edge_cases():
    labeler = CTfidfLabeler()
    assert labeler.top_terms({}) == {}
    assert labeler.top_terms({"x": ["the and of"]}) == {"x": []}
    assert labeler.label({"x": []}, {}) == {"x": ""}


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def reply(text, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])


def test_llm_labeler_titles_and_falls_back():
    client = FakeClient(
        [reply('"Printer paper jams."\nextra'), reply("", "refusal"), RuntimeError("api down")]
    )
    labeler = LLMLabeler(client=client, fallback=CTfidfLabeler())
    targets = {"a": DOCS["a"], "b": DOCS["b"], "c": ["camera lens cracked", "lens broken"]}
    labels = labeler.label(targets, {**DOCS, "c": targets["c"]})
    assert labels["a"] == "Printer paper jams"
    assert labels["b"] and labels["b"] != ""  # refusal -> c-TF-IDF label
    assert labels["c"]  # API error -> c-TF-IDF label
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default"
    assert call["output_config"] == {"effort": "low"}


def test_make_labeler():
    assert isinstance(make_labeler("ctfidf"), CTfidfLabeler)
    with pytest.raises(ValueError):
        make_labeler("nope")


def test_preprocessing_steps():
    assert normalize_text("  a  b\n c ") == "a b c"
    redacted = redact_pii(
        "mail a.b@example.com, call +1 415-555-1234, card 4111 1111 1111 1111, ip 10.0.0.1, "
        "ssn 123-45-6789, see https://x.example/y"
    )
    for token in ("[EMAIL]", "[PHONE]", "[CARD]", "[IP]", "[SSN]", "[URL]"):
        assert token in redacted
    run = build_preprocessor(PreprocessConfig(normalize=True, redact_pii=True, max_chars=10))
    assert run("  hello   me@x.io ") == "hello [EMA"
    assert build_preprocessor(PreprocessConfig()) is None


def test_parse_timestamp_forms():
    assert parse_timestamp("2026-01-01T00:00:00Z") == datetime(2026, 1, 1, tzinfo=UTC)
    assert parse_timestamp(0) == datetime(1970, 1, 1, tzinfo=UTC)
    assert parse_timestamp("86400").day == 2
    with pytest.raises(ValueError):
        parse_timestamp(" ")


def test_read_csv_with_custom_columns(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("key,body,when,topic,source\n1,printer jam,2026-01-01T00:00:00,p,web\n")
    [row] = read_items(path, ColumnMap(id="key", text="body", timestamp="when", label="topic"))
    assert row.item.id == "1" and row.label == "p" and row.item.metadata == {"source": "web"}


def test_read_jsonl(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text(json.dumps({"id": 1, "text": "a", "timestamp": "2026-01-01"}) + "\n\n")
    [row] = read_items(path)
    assert row.item.id == "1" and row.label is None


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("bad.jsonl", "{not json}\n"),
        ("list.jsonl", "[1, 2]\n"),
        ("missing.csv", "id,text\n1,a\n"),
        ("empty.csv", "id,text,timestamp\n1,,2026-01-01\n"),
        ("ts.csv", "id,text,timestamp\n1,a,yesterday\n"),
        ("data.xml", "<x/>"),
    ],
)
def test_input_errors(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    with pytest.raises(InputError):
        read_items(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_items(tmp_path / "nope.csv")
