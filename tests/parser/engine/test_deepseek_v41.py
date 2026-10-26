# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.parser.engine.replay_harness import (
    DUMMY_TOOLS,
    MockTokenizer,
    _test_request,
    collect_output,
    replay_streaming,
)
from vllm.parser.deepseek_v41 import deepseek_v41_config
from vllm.parser.parser_manager import ParserManager

CALLS = (
    '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="city" string="true">杭州</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="count" string="false">42</｜DSML｜ parameter>\n'
    '</｜DSML｜ invoke>\n<｜DSML｜ invoke name="add">\n'
    '<｜DSML｜ parameter name="x" string="false">1.5</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="y" string="false">2.25</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)

# V4's ``tool_calls`` wrapper name with V4.1's spacing (#62051).
V4_NAMED_CALLS = CALLS.replace("｜DSML｜ calls>", "｜DSML｜ tool_calls>")


def tokenizer_for(text, special_calls):
    vocab = {"<think>": 50, "</think>": 51}
    if special_calls:
        vocab |= {"<｜DSML｜ calls>": 52, "</｜DSML｜ calls>": 53}
    tokens: list[tuple[int, str]] = []
    while text:
        special = next((marker for marker in vocab if text.startswith(marker)), None)
        piece = special or text[0]
        tokens.append((vocab[piece] if special else 100 + len(tokens), piece))
        text = text[len(piece) :]
    return MockTokenizer(vocab, tokens), tokens


def parser_for(tokenizer, controls):
    cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v41",
        reasoning_parser_name="deepseek_v41",
        enable_auto_tools=True,
    )
    return cls(tokenizer, chat_template_kwargs=controls)


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("special_calls", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_registered_adapters_parse_parallel_calls_across_chunks(
    thinking,
    special_calls,
    chunk_size,
):
    text = ("Plan.</think>" if thinking else "") + "Checking." + CALLS
    tokenizer, tokens = tokenizer_for(text, special_calls)
    parser = parser_for(tokenizer, {"thinking": thinking})
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
            prompt_token_ids=[50 if thinking else 51],
        )
    )
    assert output.reasoning == ("Plan." if thinking else "")
    assert output.content.strip() == "Checking."
    assert [call["name"] for call in output.tool_calls] == ["get_weather", "add"]
    assert [json.loads(call["arguments"]) for call in output.tool_calls] == [
        {"city": "杭州", "count": 42},
        {"x": 1.5, "y": 2.25},
    ]


@pytest.mark.parametrize("thinking", [False, True])
def test_registered_adapters_parse_complete_output(thinking):
    text = ("Plan.</think>" if thinking else "") + CALLS
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": thinking})
    reasoning, content, calls = parser.parse(
        text,
        _test_request(tools=DUMMY_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert (reasoning or "") == ("Plan." if thinking else "")
    assert not (content or "").strip()
    assert [call.name for call in calls] == ["get_weather", "add"]
    assert json.loads(calls[0].arguments) == {"city": "杭州", "count": 42}


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_v4_wrapper_name_opens_tool_block(thinking, chunk_size):
    text = ("Plan.</think>" if thinking else "") + "Checking." + V4_NAMED_CALLS
    tokenizer, tokens = tokenizer_for(text, False)
    output = collect_output(
        replay_streaming(
            parser_for(tokenizer, {"thinking": thinking}),
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
            prompt_token_ids=[50 if thinking else 51],
        )
    )
    reasoning, content, calls = parser_for(tokenizer, {"thinking": thinking}).parse(
        text,
        _test_request(tools=DUMMY_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert output.reasoning == (reasoning or "") == ("Plan." if thinking else "")
    assert output.content.strip() == (content or "").strip() == "Checking."
    assert [call["name"] for call in output.tool_calls] == ["get_weather", "add"]
    assert [call.name for call in calls] == ["get_weather", "add"]
    assert (
        [json.loads(call["arguments"]) for call in output.tool_calls]
        == [json.loads(call.arguments) for call in calls]
        == [{"city": "杭州", "count": 42}, {"x": 1.5, "y": 2.25}]
    )


def test_v4_wrapper_name_ends_reasoning():
    text = "Plan." + V4_NAMED_CALLS
    tokenizer, tokens = tokenizer_for(text, False)
    output = collect_output(
        replay_streaming(
            parser_for(tokenizer, {"thinking": True}),
            tokens,
            chunk_size=1,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
            prompt_token_ids=[50],
        )
    )
    assert output.reasoning == "Plan."
    assert not output.content.strip()
    assert [call["name"] for call in output.tool_calls] == ["get_weather", "add"]


@pytest.mark.parametrize(
    ("controls", "text", "expected_reasoning", "reasoning_tokens"),
    [
        ({"thinking": False}, "12", "", 0),
        ({"enable_thinking": False}, "12", "", 0),
        ({"thinking": True, "reasoning_effort": "none"}, "12", "", 0),
        ({}, "</think>12", "", 0),
        ({}, "Plan.</think>12", "Plan.", 5),
    ],
)
def test_reasoning_adapter_controls_and_usage(
    controls,
    text,
    expected_reasoning,
    reasoning_tokens,
):
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, controls)
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=1,
            finished_on_last=True,
        )
    )
    assert output.reasoning == expected_reasoning
    assert output.content == "12"
    assert (
        parser.reasoning_parser.count_reasoning_tokens([tid for tid, _ in tokens])
        == reasoning_tokens
    )
    assert parser.is_reasoning_end([50, 100, 51])
    assert not parser.is_reasoning_end([51, 100, 50])


def test_python_argument_conversion_and_partial_values():
    converter = deepseek_v41_config().arg_converter
    raw = (
        '<｜DSML｜ parameter name="object" string="false">'
        '{"a": [true, null]}</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="bad" string="false">[broken</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="bare">7</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="text" string="true">  a<b'
    )
    assert json.loads(converter(raw, True)) == {
        "object": {"a": [True, None]},
        "bad": "[broken",
        "bare": 7,
        "text": "  a<b",
    }
    assert json.loads(converter(raw, False)) == {
        "object": {"a": [True, None]},
        "bad": "[broken",
        "bare": 7,
        "text": "  a<b",
    }


@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize(
    "cut", ["<", "</", "</｜", "</｜DSML｜", "</｜DSML｜ parameter"]
)
def test_python_argument_conversion_drops_partial_close_marker(cut, partial):
    converter = deepseek_v41_config().arg_converter
    raw = '<｜DSML｜ parameter name="city" string="true">Paris' + cut
    assert json.loads(converter(raw, partial)) == {"city": "Paris"}


@pytest.mark.parametrize("partial", [True, False])
def test_python_argument_conversion_ignores_unclosed_repeated_name(partial):
    converter = deepseek_v41_config().arg_converter
    raw = (
        '<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="city" string="true">Lyon'
    )
    assert json.loads(converter(raw, partial)) == {"city": "Paris"}


@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
@pytest.mark.parametrize(
    ("text", "expected_name", "expected_args"),
    [
        (
            (
                '<｜DSML｜ calls>\n<｜DSML｜ invoke name="WebFetch">\n'
                '<｜DSML｜ parameter name="url" string="true">https://example.com'
            ),
            "WebFetch",
            {"url": "https://example.com"},
        ),
        (
            (
                '<｜DSML｜ calls>\n<｜DSML｜ invoke name="Read">\n'
                '<｜DSML｜ parameter name="file_path" string="true">/tmp/a'
                "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
            ),
            "Read",
            {"file_path": "/tmp/a"},
        ),
    ],
)
def test_truncated_and_unclosed_parameter_streaming_and_non_streaming(
    chunk_size, text, expected_name, expected_args
):
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": False})
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
        )
    )
    assert len(output.tool_calls) == 1
    assert output.tool_calls[0]["name"] == expected_name
    assert json.loads(output.tool_calls[0]["arguments"]) == expected_args

    parser_non_stream = parser_for(tokenizer, {"thinking": False})
    _, content, calls = parser_non_stream.parse(
        text,
        _test_request(tools=DUMMY_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert not (content or "").strip()
    assert calls is not None and len(calls) == 1
    assert calls[0].name == expected_name
    assert json.loads(calls[0].arguments) == expected_args


@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
@pytest.mark.parametrize(
    ("value", "closer"),
    [
        ("Paris", "</｜DSML｜>"),
        ('print("</｜DSML｜>")', "</｜DSML｜ parameter>"),
        ("text </｜DSML｜> literal", "</｜DSML｜ parameter>"),
        ("</｜DSML｜>", "</｜DSML｜ parameter>"),
    ],
)
@pytest.mark.parametrize("string_mode", ["true", "false", None])
@pytest.mark.parametrize("has_schema", [False, True])
def test_parameter_closer_recovery_preserves_argument_text(
    chunk_size, value, closer, string_mode, has_schema
):
    attr = "" if string_mode is None else f' string="{string_mode}"'
    raw_value = (
        value if string_mode == "true" else json.dumps(value, ensure_ascii=False)
    )
    parameters: dict[str, object] = {"type": "object"}
    if has_schema:
        parameters["properties"] = {
            "city": {"type": "string"},
            "count": {"type": "integer"},
        }
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": parameters},
        }
    ]
    text = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="get_weather">'
        f'<｜DSML｜ parameter name="city"{attr}>{raw_value}{closer}\n'
        '<｜DSML｜ parameter name="count" string="false">2</｜DSML｜ parameter>'
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    )
    tokenizer, tokens = tokenizer_for(text, False)
    expected = {"city": value, "count": 2}
    output = collect_output(
        replay_streaming(
            parser_for(tokenizer, {"thinking": False}),
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=tools,
        )
    )
    assert json.loads(output.tool_calls[0]["arguments"]) == expected
    _, _, calls = parser_for(tokenizer, {"thinking": False}).parse(
        text,
        _test_request(tools=tools),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert calls is not None and len(calls) == 1
    assert json.loads(calls[0].arguments) == expected


@pytest.mark.parametrize("string_mode", ["true", "false", None])
@pytest.mark.parametrize("after_closer", [0, 2, 8, 20])
@pytest.mark.parametrize("next_parameter", [False, True])
def test_bare_closer_waits_for_parameter_boundary(
    string_mode, after_closer, next_parameter
):
    attr = "" if string_mode is None else f' string="{string_mode}"'
    text = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="get_weather">'
        f'<｜DSML｜ parameter name="city"{attr}>Paris</｜DSML｜>\n'
    )
    expected: dict[str, object] = {"city": "Paris"}
    if next_parameter:
        text += '<｜DSML｜ parameter name="count" string="false">2</｜DSML｜ parameter>'
        expected["count"] = 2
    text += "</｜DSML｜ invoke></｜DSML｜ calls>"
    tokenizer, tokens = tokenizer_for(text, False)
    split = text.index("</｜DSML｜>") + len("</｜DSML｜>") + after_closer
    output = collect_output(
        replay_streaming(
            parser_for(tokenizer, {"thinking": False}),
            tokens,
            chunk_size=split,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
        )
    )
    assert json.loads(output.tool_calls[0]["arguments"]) == expected
