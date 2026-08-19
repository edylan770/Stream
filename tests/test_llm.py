"""§12's four rules, and the two transports that carry them.

Nothing here reaches a network, and nothing here evaluates a prompt. **What is
asserted is refusal**: §12 exists because a model that answers confidently
about material it never read is the failure mode, and the only defences against
it are an id that has to exist and a quote that has to be present.

The API source is exercised against a fake client. That proves the request is
assembled from config and that a refusal is not mistaken for an answer; it
proves nothing about whether the real API accepts that request, and it cannot
until a key exists. `test_the_request_carries_no_sampling_parameters` is the
one guard that can be written blind, because those parameters are a documented
400 rather than a matter of taste.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from clipforge import config, db, llm, paths
from clipforge.llm import prompts, sources, validate
from clipforge.render import hooks


@pytest.fixture
def cfg():
    return config.load()


@pytest.fixture
def conn(tmp_path):
    """A database with one stream, for the `tool_metrics` row §12.2 requires."""
    cfg = config.load(overrides=[f"paths.data_root={(tmp_path / 'd').as_posix()}"])
    db.open_db(cfg.db_path).close()
    connection = db.open_db(cfg.db_path, migrate_to_latest=False)
    paths.StreamPaths(cfg.data_root, "s").ensure()
    connection.execute(
        "INSERT INTO streams (id, date, master_path, marker_time_base, "
        "duration_s, resolution) VALUES "
        "('s', '2026-08-18', 'm.mkv', 'vod', 95.0, '640x360')")
    connection.commit()
    yield connection
    connection.close()


#: Three ids standing for three pieces of text. Deliberately not 1, 2, 3 --
#: §12.2's argument for real handles is that an invented number has somewhere
#: to fail, and a test whose ids start at 1 cannot show that.
KNOWN = {
    41: "Iron Fist is diving our backline again",
    58: "I can't believe that actually worked.",
    77: "Okay, I'm going Jeff.",
}


def run(entries, **kwargs):
    return validate.validate_selections(
        entries, KNOWN, id_field="seq", text_for=lambda text: text, **kwargs)


# --------------------------------------------------------------------------
# §12.2 — an id that was never handed out has to fail
# --------------------------------------------------------------------------


def test_a_real_id_with_a_real_quote_is_accepted():
    result = run([{"seq": 41, "quote": "diving our backline"}])

    assert [s.key for s in result.selections] == [41]
    assert result.dropped == []
    assert result.invalid_id_rate == 0.0


def test_an_invented_id_is_dropped_and_counted():
    """§12.2: "Non-existent IDs are silently dropped, and the drop is logged"."""
    result = run([{"seq": 9999, "quote": "anything"}])

    assert result.selections == []
    assert result.dropped[0][0] == "unknown-id"
    assert result.returned == 1
    assert result.invalid_id_rate == 1.0


def test_the_rate_is_the_share_of_returned_entries_that_named_nothing():
    result = run([
        {"seq": 41, "quote": "diving our backline"},
        {"seq": 9999, "quote": "anything"},
        {"seq": 8888, "quote": "anything"},
        {"seq": 58, "quote": "actually worked"},
    ])

    assert len(result.selections) == 2
    assert result.returned == 4
    assert result.invalid_id_rate == 0.5


def test_an_id_returned_as_a_string_still_resolves():
    """JSON has no int/str distinction a model reliably honours, and `"41"`
    names a handle it was actually given. Rescuing that is not the same as
    rescuing an invention -- the next test is the one that matters."""
    result = run([{"seq": "41", "quote": "diving our backline"}])

    assert [s.key for s in result.selections] == [41]


def test_a_string_that_is_not_an_id_is_still_a_hallucination():
    result = run([{"seq": "the first clip", "quote": "diving our backline"}])

    assert result.dropped[0][0] == "unknown-id"
    assert result.invalid_id_rate == 1.0


def test_an_unhashable_id_is_dropped_rather_than_raising():
    """A model that answers with a list where an id belongs is wrong, not
    exceptional. `41 in {...}` on an unhashable value raises TypeError, and a
    validator that dies on bad input has stopped validating."""
    result = run([{"seq": [41], "quote": "diving our backline"}])

    assert result.dropped[0][0] == "unknown-id"


def test_a_missing_id_field_is_a_hallucination_not_a_crash():
    result = run([{"quote": "diving our backline"}])

    assert result.dropped[0][0] == "unknown-id"


# --------------------------------------------------------------------------
# §12.3 — the quote has to be in the material
# --------------------------------------------------------------------------


def test_a_fabricated_quote_is_dropped():
    result = run([{"seq": 41, "quote": "and then the whole lobby exploded"}])

    assert result.selections == []
    assert result.dropped[0][0] == "bad-quote"


def test_a_real_quote_belonging_to_a_DIFFERENT_id_is_dropped():
    """The interesting adversarial case, and the one a lazy check misses: the
    words are genuinely in the transcript, just not in the one the model named.
    Checking the quote against the corpus instead of against the referenced
    material would accept this, and §12.3's whole purpose is to catch a model
    answering about something it did not read."""
    result = run([{"seq": 41, "quote": "Okay, I'm going Jeff."}])

    assert result.dropped[0][0] == "bad-quote"
    assert result.invalid_id_rate == 0.0     # it named a real id; it lied later


def test_a_missing_quote_is_dropped():
    result = run([{"seq": 41}])

    assert result.dropped[0][0] == "no-quote"


def test_a_whitespace_only_quote_is_dropped():
    result = run([{"seq": 41, "quote": "   "}])

    assert result.dropped[0][0] == "no-quote"


def test_rewrapping_a_quote_is_still_verbatim():
    """A model that re-wrapped a line has quoted it; one that rewrote the words
    has not. So whitespace and case are normalised and nothing else is."""
    result = run([{"seq": 41, "quote": "diving   our\n  BACKLINE"}])

    assert len(result.selections) == 1


def test_changing_the_punctuation_is_not_verbatim():
    """Deliberately strict. "Verbatim" is the requirement, and punctuation is
    part of what was said -- an apostrophe dropped is a word altered."""
    result = run([{"seq": 58, "quote": "I cant believe that actually worked"}])

    assert result.dropped[0][0] == "bad-quote"


# --------------------------------------------------------------------------
# malformed replies
# --------------------------------------------------------------------------


def test_a_non_object_entry_is_malformed_and_does_not_count_as_returned():
    """It is not an answer about anything, so counting it would put noise in
    the denominator of a hallucination rate."""
    result = run(["just a string", 7, None])

    assert [reason for reason, _ in result.dropped] == ["malformed"] * 3
    assert result.returned == 0
    assert result.invalid_id_rate == 0.0


def test_the_extra_check_runs_before_the_quote_is_examined():
    """Per-prompt requirements are the caller's. The validator gives them a
    place to fail that keeps the drop reasons in one list."""
    def needs_options(entry, key, _text):
        return None if entry.get("options") else ("no-options", f"seq {key}")

    result = run([{"seq": 41, "quote": "nonsense that is not in the text"}],
                 check=needs_options)

    assert result.dropped[0][0] == "no-options"


def test_nothing_returned_is_a_rate_of_zero_not_a_division_by_zero():
    assert validate.Validated().invalid_id_rate == 0.0


# --------------------------------------------------------------------------
# the tolerant parser
# --------------------------------------------------------------------------


def test_json_wrapped_in_prose_is_found():
    reply = ('Sure! Here are the hooks I came up with.\n\n'
             '{"hooks": [{"seq": 41}]}\n\n'
             'Let me know if you would like different angles.')

    assert parse(reply) == {"hooks": [{"seq": 41}]}


def test_the_last_object_wins():
    """A chat window often restates a shape before giving the real answer."""
    reply = 'Format: {"hooks": []}\n\nAnswer:\n{"hooks": [{"seq": 77}]}'

    assert parse(reply)["hooks"] == [{"seq": 77}]


def test_a_fenced_block_needs_no_special_handling():
    reply = '```json\n{"hooks": [{"seq": 58}]}\n```'

    assert parse(reply)["hooks"] == [{"seq": 58}]


def test_a_reply_with_no_json_is_none_rather_than_an_exception():
    assert parse("I am not able to help with that.") is None


def test_broken_json_is_none():
    assert parse('{"hooks": [{"seq": 41,,,}]}') is None


def parse(text):
    return validate.parse_reply(text)


# --------------------------------------------------------------------------
# §12.2's metric
# --------------------------------------------------------------------------


def test_the_metric_is_the_name_ss14_gives_it():
    """§14's table calls it `llm_invalid_id_rate`. A different name here means
    whatever reads it later finds nothing."""
    assert validate.INVALID_ID_METRIC == "llm_invalid_id_rate"


def test_the_rate_is_written_to_tool_metrics(conn):
    result = run([{"seq": 9999, "quote": "x"}, {"seq": 41, "quote": "diving"}])

    validate.record(conn, "s", result)

    row = conn.execute("SELECT value, meta FROM tool_metrics WHERE metric = ?",
                       (validate.INVALID_ID_METRIC,)).fetchone()
    assert row["value"] == 0.5
    meta = json.loads(row["meta"])
    assert meta["returned"] == 2 and meta["accepted"] == 1
    assert meta["dropped"] == ["unknown-id"]


# --------------------------------------------------------------------------
# one implementation, not two
# --------------------------------------------------------------------------


def test_hooks_reuses_the_shared_checks_rather_than_copying_them():
    """The whole point of the move. `is` rather than `==`: a copy would still
    compare equal on the values that matter and drift on the next §12 edit."""
    assert hooks.parse_reply is validate.parse_reply
    assert hooks.normalise is validate.normalise
    assert hooks.record is validate.record
    assert hooks.INVALID_ID_METRIC is validate.INVALID_ID_METRIC


def test_the_hook_validator_is_the_shared_loop_with_hook_shaped_arguments():
    """`hooks.validate` may keep its own signature; it must not keep its own
    loop. If it stopped calling the shared one, this fails."""
    calls = []
    original = validate.validate_selections

    def spy(*args, **kwargs):
        calls.append(kwargs.get("id_field"))
        return original(*args, **kwargs)

    llm.validate_selections = spy
    try:
        hooks.validate({"hooks": [{"export_id": 1, "quote": "x"}]}, [])
    finally:
        llm.validate_selections = original

    assert calls == ["export_id"]


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------


def test_the_manual_source_is_always_available(cfg):
    """It needs a person and a browser, not a credential — so unlike the Ollama
    and WhisperX checks, there is nothing to defer on."""
    usable, why = sources.ManualSource().available(cfg)

    assert usable and why == ""


def test_the_manual_source_cannot_call_anything(cfg):
    """It is a transport for a human, and a `complete` that silently did
    nothing would be worse than one that says what it is."""
    with pytest.raises(sources.LLMError, match="paste"):
        sources.ManualSource().complete(cfg, "a prompt")


def test_the_api_source_is_unavailable_here_and_says_why(cfg, monkeypatch):
    """MEASURED on this machine: no key, and the package is not installed."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    usable, why = sources.AnthropicSource().available(cfg)

    assert not usable
    assert "needs a key" in why
    assert "ANTHROPIC_API_KEY" in why


def test_the_two_reasons_are_reported_separately(cfg, monkeypatch):
    """A missing package and a missing key need different fixes, and a single
    "not configured" would hide which one is wrong."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(sources, "find_spec", lambda name: None)
    both = sources.AnthropicSource().available(cfg)[1]

    monkeypatch.setattr(sources, "find_spec", lambda name: object())
    key_only = sources.AnthropicSource().available(cfg)[1]

    assert "not installed" in both and "needs a key" in both
    assert "not installed" not in key_only and "needs a key" in key_only


def test_a_key_and_a_package_is_available(cfg, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-real-key")
    monkeypatch.setattr(sources, "find_spec", lambda name: object())

    assert sources.AnthropicSource().available(cfg) == (True, "")


def test_the_key_variable_is_named_by_config_and_never_holds_the_key(monkeypatch):
    """This file is in the repository. A key in it would be a key in git."""
    monkeypatch.setenv("SOMEWHERE_ELSE", "sk-not-a-real-key")
    monkeypatch.setattr(sources, "find_spec", lambda name: object())
    cfg = config.load(overrides=["llm.anthropic.api_key_env=SOMEWHERE_ELSE"])

    assert sources.AnthropicSource().available(cfg)[0]


def test_an_unknown_source_name_is_refused_and_lists_what_exists(cfg):
    bogus = config.load(overrides=["llm.source=gpt"])

    with pytest.raises(sources.LLMError, match="manual"):
        sources.source_for(bogus)


def test_hooks_reads_its_own_source_key(cfg):
    """§8.5's key predates the shared subtree and is already in local configs."""
    chosen = config.load(overrides=["render.hooks.source=anthropic",
                                    "llm.source=manual"])

    assert hooks.source_for(chosen).name == "anthropic"


# --------------------------------------------------------------------------
# the request that has never been sent
# --------------------------------------------------------------------------


def test_the_request_is_built_from_config(cfg):
    tuned = config.load(overrides=[
        "llm.anthropic.model=claude-haiku-4-5",
        "llm.anthropic.effort=low",
        "llm.anthropic.max_tokens=512",
    ])

    request = sources.build_request(tuned, "hello")

    assert request["model"] == "claude-haiku-4-5"
    assert request["max_tokens"] == 512
    assert request["output_config"]["effort"] == "low"
    assert request["messages"] == [{"role": "user", "content": "hello"}]


def test_thinking_is_adaptive_rather_than_a_token_budget(cfg):
    """A fixed `budget_tokens` is rejected outright by the model this config
    names, and the depth control is `effort`."""
    request = sources.build_request(cfg, "hello")

    assert request["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in json.dumps(request)


def test_the_request_carries_no_sampling_parameters(cfg):
    """The one thing about this request that can be checked without a key:
    `temperature`, `top_p` and `top_k` are a documented 400 on the configured
    model, not a matter of taste."""
    request = sources.build_request(cfg, "hello")

    for parameter in ("temperature", "top_p", "top_k"):
        assert parameter not in request


def test_a_schema_becomes_a_server_side_constraint(cfg):
    """§12.3's "constrained JSON schema on every call", enforced by the API
    rather than only asked for in the prompt."""
    schema = {"type": "object", "properties": {"hooks": {"type": "array"}}}

    request = sources.build_request(cfg, "hello", schema=schema)

    assert request["output_config"]["format"] == {
        "type": "json_schema", "schema": schema}


def test_no_schema_means_no_format_key(cfg):
    assert "format" not in sources.build_request(cfg, "hello")["output_config"]


def test_the_fallback_can_be_turned_off_in_one_edit(cfg):
    with_it = sources.build_request(cfg, "hello")
    without = sources.build_request(
        config.load(overrides=["llm.anthropic.fallbacks="]), "hello")

    assert with_it["fallbacks"] == "default" and with_it["betas"]
    assert "fallbacks" not in without and "betas" not in without


# --------------------------------------------------------------------------
# reading a reply that is not a reply
# --------------------------------------------------------------------------


def block(kind, text=""):
    return types.SimpleNamespace(type=kind, text=text)


def message(stop_reason="end_turn", content=(), category=None):
    return types.SimpleNamespace(
        stop_reason=stop_reason, content=list(content),
        stop_details=types.SimpleNamespace(category=category) if category else None)


def test_a_refusal_is_not_an_answer():
    """It arrives as a perfectly successful response with nothing in it.
    Reading `content[0]` first turns that into an IndexError three layers from
    the cause."""
    with pytest.raises(sources.LLMError, match="declined"):
        sources.read_text(message(stop_reason="refusal", category="cyber"))


def test_a_refusal_with_no_stated_category_still_refuses():
    with pytest.raises(sources.LLMError, match="declined"):
        sources.read_text(message(stop_reason="refusal"))


def test_thinking_blocks_are_not_the_answer():
    """Only text blocks are. A reply that is all thinking has said nothing."""
    with pytest.raises(sources.LLMError, match="no text"):
        sources.read_text(message(content=[block("thinking")]))


def test_text_blocks_come_back_as_text_for_the_shared_parser():
    """The API path hands back the same kind of thing a person pastes, which is
    what lets both transports run through one validator."""
    text = sources.read_text(message(content=[
        block("thinking"), block("text", '{"hooks": '), block("text", "[]}")]))

    assert validate.parse_reply(text) == {"hooks": []}


# --------------------------------------------------------------------------
# the call itself, against a fake client
# --------------------------------------------------------------------------


class _Stream:
    def __init__(self, reply):
        self._reply = reply

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._reply


class _Messages:
    def __init__(self, reply):
        self._reply = reply
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _Stream(self._reply)


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Enough of the SDK to see what would have been sent. It proves the call
    is assembled from config; it proves nothing about whether the real API
    accepts it, and cannot until a key exists."""
    plain = _Messages(message(content=[block("text", '{"hooks": []}')]))
    beta = _Messages(message(content=[block("text", '{"hooks": []}')]))
    client = types.SimpleNamespace(
        messages=plain, beta=types.SimpleNamespace(messages=beta))
    module = types.ModuleType("anthropic")
    module.Anthropic = lambda *a, **k: client

    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setattr(sources, "find_spec", lambda name: object())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-real-key")
    return plain, beta


def test_the_call_streams_rather_than_waiting(fake_anthropic):
    """A digest map is long in and long out, and the SDK refuses a
    non-streaming request it estimates will outlive the HTTP timeout."""
    plain, beta = fake_anthropic

    text = sources.AnthropicSource().complete(config.load(), "a prompt")

    assert text == '{"hooks": []}'
    assert len(plain.calls) + len(beta.calls) == 1


def test_the_beta_endpoint_is_used_only_when_a_beta_flag_is_in_play(fake_anthropic):
    plain, beta = fake_anthropic

    sources.AnthropicSource().complete(config.load(), "a prompt")
    assert len(beta.calls) == 1 and plain.calls == []

    sources.AnthropicSource().complete(
        config.load(overrides=["llm.anthropic.fallbacks="]), "a prompt")
    assert len(plain.calls) == 1


def test_no_call_is_made_without_a_key(cfg, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(sources.LLMError, match="needs a key"):
        sources.AnthropicSource().complete(cfg, "a prompt")


# --------------------------------------------------------------------------
# prompts live in a file
# --------------------------------------------------------------------------


def test_the_packaged_file_holds_the_hook_prompt(cfg):
    assert hooks.PROMPT_NAME in prompts.load(cfg)


def test_the_hook_prompt_states_every_rule_it_is_checked_against(cfg):
    """§12's checks run whatever the prompt says. Saying it too is what makes a
    well-behaved model comply rather than be caught."""
    text = prompts.get(cfg, hooks.PROMPT_NAME)

    assert "ONE JSON object" in text          # §12.3 constrained output
    assert "Do not invent ids" in text        # §12.2 real handles
    assert "VERBATIM" in text                 # §12.3 the quote
    assert "discarded" in text                # and that it is enforced


def test_the_prompt_file_never_asks_for_a_timestamp(cfg):
    """§12.1. Structural, because it has to hold for every prompt added later,
    not just the one that exists today."""
    for name, text in prompts.load(cfg).items():
        assert "timestamp" not in text.lower(), name
        assert "t_start" not in text, name


def test_placeholders_are_substituted(cfg):
    text = prompts.render(cfg, hooks.PROMPT_NAME, stream_id="s2026",
                          clip_count=3, option_count=5, schema="{}")

    assert "s2026" in text and "3 short-form" in text and "5 hook variants" in text


def test_a_missing_placeholder_is_loud(cfg):
    """Silently leaving `$schema` in the text would send a model an instruction
    referring to an example that is not there."""
    with pytest.raises(sources.LLMError, match="schema"):
        prompts.render(cfg, hooks.PROMPT_NAME, stream_id="s", clip_count=1,
                       option_count=5)


def test_an_unknown_prompt_name_lists_what_the_file_holds(cfg):
    with pytest.raises(sources.LLMError, match=hooks.PROMPT_NAME):
        prompts.render(cfg, "no-such-prompt")


def test_a_missing_prompt_file_names_the_path(tmp_path):
    missing = config.load(
        overrides=[f"llm.prompts.file={(tmp_path / 'gone.yaml').as_posix()}"])

    with pytest.raises(sources.LLMError, match="not found"):
        prompts.load(missing)


def test_the_file_can_be_replaced_without_touching_code(tmp_path):
    """The reason prompt text is config at all: its quality cannot be measured
    here, so it is the part most certain to be rewritten."""
    mine = tmp_path / "mine.yaml"
    mine.write_text("prompts:\n  hook: |\n    ask it $option_count things\n",
                    encoding="utf-8")
    cfg = config.load(overrides=[f"llm.prompts.file={mine.as_posix()}"])

    assert prompts.render(cfg, "hook", option_count=2) == "ask it 2 things\n"


def test_llm_config_is_outside_the_versioned_subtrees():
    """Editing a prompt must never invalidate a candidate or force a re-score.
    Same reasoning that put `render:`, `previews:`, `search:` and `digest:`
    outside."""
    from clipforge.config import VERSIONED_SUBTREES
    assert "llm" not in VERSIONED_SUBTREES


def test_rewriting_a_prompt_does_not_mint_a_new_config_version(tmp_path):
    """The consequence of the line above, asserted rather than assumed. §9.1
    keeps digests forever; §6.1 promises re-scoring is free. A prompt edit must
    cost neither."""
    mine = tmp_path / "mine.yaml"
    mine.write_text("prompts:\n  hook: |\n    entirely different text\n",
                    encoding="utf-8")

    before = config.load().version
    after = config.load(
        overrides=[f"llm.prompts.file={mine.as_posix()}",
                   "llm.anthropic.model=claude-haiku-4-5"]).version

    assert after == before


def test_the_prompt_hash_is_stable_and_moves_with_the_text(tmp_path, cfg):
    """§9.1 keeps digests forever and never regenerates them, so two digests of
    one stream made by two prompts have to be distinguishable in origin."""
    first = prompts.digest_of(cfg, hooks.PROMPT_NAME)

    assert first == prompts.digest_of(cfg, hooks.PROMPT_NAME)

    edited = tmp_path / "edited.yaml"
    edited.write_text(
        "prompts:\n  hook: |\n    " + prompts.get(cfg, hooks.PROMPT_NAME)
        .replace("\n", "\n    ") + "\n    one more line\n", encoding="utf-8")
    other = config.load(overrides=[f"llm.prompts.file={edited.as_posix()}"])

    assert prompts.digest_of(other, hooks.PROMPT_NAME) != first
