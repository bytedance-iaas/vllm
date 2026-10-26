# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeVar

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)

# Used internally
_ChatCompletionResponseChoiceT = TypeVar(
    "_ChatCompletionResponseChoiceT",
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)


def maybe_filter_parallel_tool_calls(
    choice: _ChatCompletionResponseChoiceT, request: ChatCompletionRequest
) -> _ChatCompletionResponseChoiceT:
    """Filter to first tool call only when parallel_tool_calls is explicitly False."""
    if request.parallel_tool_calls is not False:
        return choice

    if isinstance(choice, ChatCompletionResponseChoice) and choice.message.tool_calls:
        choice.message.tool_calls = choice.message.tool_calls[:1]
    elif (
        isinstance(choice, ChatCompletionResponseStreamChoice)
        and choice.delta.tool_calls
    ):
        choice.delta.tool_calls = [
            tool_call for tool_call in choice.delta.tool_calls if tool_call.index == 0
        ]

    return choice


def resolve_finish_reason(
    engine_finish_reason: str | None,
    request: ChatCompletionRequest | None,
    tool_calls_made: bool,
) -> str:
    """Map the engine finish reason to the OpenAI chat ``finish_reason``.

    ``tool_calls`` is reported only when the choice actually carries tool
    calls, the engine stopped on its own (a ``length`` cut stays visible so
    clients can tell a truncated call from a complete one) and the caller did
    not force a single named tool, for which OpenAI reports ``stop``.
    """
    finish_reason = engine_finish_reason or "stop"
    forced_named_tool = request is not None and isinstance(
        request.tool_choice, ChatCompletionNamedToolChoiceParam
    )
    if tool_calls_made and finish_reason == "stop" and not forced_named_tool:
        return "tool_calls"
    return finish_reason
