"""Anthropic Messages API passthrough preprocessing helpers.

Ports of the helpers in copilot-api/src/routes/messages/preprocess.ts so that
Anthropic /v1/messages payloads can be forwarded to Copilot's native Messages
endpoint without tripping its stricter validation.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable

from effort_mapping import map_effort_for_model

__all__ = [
    "strip_cache_control_scope",
    "filter_assistant_thinking_placeholders",
    "merge_tool_result_with_reminder",
    "strip_tool_reference_turn_boundary",
    "sanitize_ide_tools",
    "apply_adaptive_thinking",
    "prepare_messages_passthrough_payload",
    "budget_tokens_to_effort",
    "detect_compact_type",
    "COMPACT_REQUEST",
    "COMPACT_AUTO_CONTINUE",
    "TOOL_REFERENCE_TURN_BOUNDARY",
]


IDE_EXECUTE_CODE_TOOL = "mcp__ide__executeCode"
_IDE_EXECUTE_CODE_TOOL_NORMALIZED = IDE_EXECUTE_CODE_TOOL.lower()
IDE_GET_DIAGNOSTICS_TOOL = "mcp__ide__getDiagnostics"
IDE_GET_DIAGNOSTICS_DESCRIPTION = (
    "Get language diagnostics from VS Code. Returns errors, warnings, "
    "information, and hints for files in the workspace."
)

TOOL_REFERENCE_TURN_BOUNDARY = "Tool loaded."

# Compaction detection — ports copilot-api/src/lib/compact.ts.
COMPACT_REQUEST = 1
COMPACT_AUTO_CONTINUE = 2

_COMPACT_SYSTEM_PROMPT_START = (
    "You are a helpful AI assistant tasked with summarizing conversations"
)
_COMPACT_TEXT_ONLY_GUARD = (
    "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools."
)
_COMPACT_SUMMARY_PROMPT_START = (
    "Your task is to create a detailed summary of the conversation so far"
)
_COMPACT_MESSAGE_SECTIONS = ("Pending Tasks:", "Current Work:")
_COMPACT_AUTO_CONTINUE_PROMPT_STARTS = (
    # Claude Code
    "This session is being continued from a previous conversation that ran out of context. "
    "The summary below covers the earlier portion of the conversation.",
    # OpenCode
    "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed.",
)


# ---------------------------------------------------------------------------
# 1. cache_control scope stripping
# ---------------------------------------------------------------------------

def _strip_scope_from_block(block: Any) -> None:
    if not isinstance(block, dict):
        return
    cc = block.get("cache_control")
    if isinstance(cc, dict) and "scope" in cc:
        cc.pop("scope", None)
    # Recurse into nested content arrays (tool_result blocks etc.)
    nested = block.get("content")
    if isinstance(nested, list):
        for child in nested:
            _strip_scope_from_block(child)


def strip_cache_control_scope(body: dict) -> dict:
    """Remove the ``scope`` key from every ``cache_control`` object.

    Walks ``system``, ``messages[*].content[*]`` (recursively) and
    ``tools[*]``. Copilot rejects unknown ``cache_control`` fields like
    ``scope`` even though the Anthropic API accepts them.
    """
    if not isinstance(body, dict):
        return body

    system = body.get("system")
    if isinstance(system, list):
        for blk in system:
            _strip_scope_from_block(blk)
    # system as string => nothing to do

    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for blk in content:
                    _strip_scope_from_block(blk)

    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            _strip_scope_from_block(tool)

    return body


# ---------------------------------------------------------------------------
# 2. Filter placeholder thinking blocks
# ---------------------------------------------------------------------------

def _is_placeholder_thinking(block: dict) -> bool:
    if not isinstance(block, dict) or block.get("type") != "thinking":
        return False
    thinking = block.get("thinking")
    signature = block.get("signature")
    if not isinstance(thinking, str) or not thinking:
        return True
    if thinking == "Thinking...":
        return True
    if not isinstance(signature, str) or not signature:
        return True
    if "@" in signature:
        return True
    return False


def filter_assistant_thinking_placeholders(body: dict) -> dict:
    """Drop assistant thinking blocks that are placeholders.

    Claude Code resends ``{type: "thinking", thinking: "Thinking..."}`` or
    blocks whose ``signature`` starts with ``@``. Forwarding those to the
    upstream causes a 400.
    """
    if not isinstance(body, dict):
        return body
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        msg["content"] = [
            blk for blk in content
            if not (isinstance(blk, dict)
                    and blk.get("type") == "thinking"
                    and _is_placeholder_thinking(blk))
        ]
    return body


# ---------------------------------------------------------------------------
# 3. Merge trailing system-reminder text into the last tool_result
# ---------------------------------------------------------------------------

def _has_tool_reference(block: dict) -> bool:
    content = block.get("content") if isinstance(block, dict) else None
    if not isinstance(content, list):
        return False
    return any(
        isinstance(c, dict) and c.get("type") == "tool_reference" for c in content
    )


def _is_text_block(block: Any) -> bool:
    if not isinstance(block, dict) or block.get("type") != "text":
        return False
    return isinstance(block.get("text"), str)


def _is_attachment_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    return block.get("type") in ("image", "document")


def _merge_content_with_texts(tr: dict, text_blocks: list[dict]) -> None:
    """Fold a list of text blocks into a single tool_result block in place.

    Mirrors TS `mergeContentWithTexts`: if tr.content is a string, concatenate;
    if it's a list, extend (unless the tool_result already has a tool_reference,
    in which case we leave it alone — match TS `hasToolRef` guard).
    """
    tr_content = tr.get("content")
    if isinstance(tr_content, str):
        joined = "\n\n".join(tb.get("text", "") for tb in text_blocks)
        tr["content"] = f"{tr_content}\n\n{joined}"
        return
    if isinstance(tr_content, list):
        if _has_tool_reference(tr):
            return
        tr["content"] = list(tr_content) + [
            {"type": "text", "text": tb.get("text", "")} for tb in text_blocks
        ]
        return
    # No existing content: initialize it with the new text blocks.
    tr["content"] = [
        {"type": "text", "text": tb.get("text", "")} for tb in text_blocks
    ]


def _merge_content_with_attachments(tr: dict, attachments: list[dict]) -> None:
    tr_content = tr.get("content")
    if isinstance(tr_content, str):
        tr["content"] = [{"type": "text", "text": tr_content}] + list(attachments)
        return
    if isinstance(tr_content, list):
        tr["content"] = list(tr_content) + list(attachments)
        return
    tr["content"] = list(attachments)


def _get_mergeable_tool_result_indices(tool_results: list[dict]) -> list[int]:
    """Indices of tool_results that are neither errors nor tool_reference carriers."""
    out: list[int] = []
    for idx, blk in enumerate(tool_results):
        if not isinstance(blk, dict):
            continue
        if blk.get("is_error"):
            continue
        if _has_tool_reference(blk):
            continue
        out.append(idx)
    return out


_PDF_FILE_READ_PREFIX = "PDF file read:"


def _starts_with_pdf_file_read(tr: dict) -> bool:
    content = tr.get("content") if isinstance(tr, dict) else None
    if isinstance(content, str):
        return content.startswith(_PDF_FILE_READ_PREFIX)
    if not isinstance(content, list) or not content:
        return False
    if any(isinstance(b, dict) and b.get("type") == "document" for b in content):
        return False
    first = content[0]
    if not isinstance(first, dict) or first.get("type") != "text":
        return False
    text = first.get("text")
    return isinstance(text, str) and text.startswith(_PDF_FILE_READ_PREFIX)


def _assign_attachments_to_tool_results(
    target: dict[int, list[dict]],
    attachments: list[tuple[int, dict]],
    *,
    tool_result_indices: list[int],
    fallback_indices: list[int] | None = None,
) -> None:
    """Mirrors TS `assignAttachmentsToToolResults`.

    `attachments` is a list of (order, attachment_block) preserving source
    ordering. When the count of candidate tool_results equals the count of
    attachments we pair them 1:1; otherwise we dump all remaining attachments
    into the last fallback index.
    """
    if not attachments:
        return
    fallbacks = fallback_indices if fallback_indices is not None else tool_result_indices

    if tool_result_indices and len(tool_result_indices) == len(attachments):
        for i, tri in enumerate(tool_result_indices):
            target.setdefault(tri, []).append(attachments[i][1])
        return

    if not fallbacks:
        return
    last = fallbacks[-1]
    target.setdefault(last, []).extend(att for _, att in attachments)


def _merge_attachments_into_tool_results(
    tool_results: list[dict],
    attachments_by_tr_index: dict[int, list[dict]],
) -> None:
    if not attachments_by_tr_index:
        return
    for idx, tr in enumerate(tool_results):
        atts = attachments_by_tr_index.get(idx)
        if not atts:
            continue
        # Preserve source ordering already implied by assignment; `order` was
        # monotonic so the list is already sorted.
        _merge_content_with_attachments(tr, atts)


def _merge_user_message_content(content: list) -> list | None:
    """Mirror copilot-api's `mergeUserMessageContent`.

    Folds trailing text blocks into the LAST mergeable tool_result (paired 1:1
    when counts match) and merges any image/document attachments into
    tool_results (with the PDF-read special case). Returns the rewritten
    content list, or None when the content cannot be merged (e.g. contains an
    unknown block type).
    """
    tool_results: list[dict] = []
    text_blocks: list[dict] = []
    attachments: list[tuple[int, dict]] = []

    for order, blk in enumerate(content):
        if not isinstance(blk, dict):
            return None
        t = blk.get("type")
        if t == "tool_result":
            tool_results.append(blk)
        elif t == "text":
            text_blocks.append(blk)
        elif t in ("image", "document"):
            attachments.append((order, blk))
        else:
            # Unknown content type — bail rather than risk data loss.
            return None

    if not tool_results or (not text_blocks and not attachments):
        return None

    # --- Merge text blocks into tool_results -------------------------------
    if text_blocks:
        if len(tool_results) == len(text_blocks):
            for tr, tb in zip(tool_results, text_blocks):
                _merge_content_with_texts(tr, [tb])
        else:
            # Fold all texts into the last tool_result.
            # Match TS: skip tool_results whose content carries a tool_reference
            # (handled inside _merge_content_with_texts); identify last
            # mergeable tool_result.
            last_idx = len(tool_results) - 1
            _merge_content_with_texts(tool_results[last_idx], text_blocks)

    # --- Merge attachments into tool_results -------------------------------
    if attachments:
        mergeable = _get_mergeable_tool_result_indices(tool_results)
        pdf_indices = [i for i in mergeable if _starts_with_pdf_file_read(tool_results[i])]

        documents = [(o, a) for (o, a) in attachments if a.get("type") == "document"]
        by_tr: dict[int, list[dict]] = {}
        remaining = attachments
        count_indices = list(mergeable)

        if documents and pdf_indices:
            pair_count = min(len(pdf_indices), len(documents))
            matched_docs = documents[:pair_count]
            matched_doc_orders = {o for (o, _) in matched_docs}
            matched_pdf_indices = pdf_indices[:pair_count]
            matched_pdf_set = set(matched_pdf_indices)

            _assign_attachments_to_tool_results(
                by_tr,
                matched_docs,
                tool_result_indices=matched_pdf_indices,
            )
            count_indices = [i for i in mergeable if i not in matched_pdf_set]
            remaining = [
                (o, a) for (o, a) in attachments
                if a.get("type") != "document" or o not in matched_doc_orders
            ]

        _assign_attachments_to_tool_results(
            by_tr,
            remaining,
            tool_result_indices=count_indices,
            fallback_indices=mergeable,
        )
        _merge_attachments_into_tool_results(tool_results, by_tr)

    # Rebuild the user content list: only tool_result blocks, in their
    # original relative order.
    return list(tool_results)


def merge_tool_result_with_reminder(body: dict, *, skip_last_message: bool = False) -> dict:
    """Port of copilot-api's ``mergeToolResultForClaude``.

    For every user turn whose content is a list of tool_result / text /
    image / document blocks, fold the text and attachment blocks into the
    tool_result blocks. When ``skip_last_message`` is True, the final message
    in the conversation is left untouched (used during compaction requests
    so that the compact user prompt is not rewritten).
    """
    if not isinstance(body, dict):
        return body
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return body

    last_idx = len(messages) - 1
    for idx, msg in enumerate(messages):
        if skip_last_message and idx == last_idx:
            continue
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        merged = _merge_user_message_content(content)
        if merged is not None:
            msg["content"] = merged

    return body


def strip_tool_reference_turn_boundary(body: dict) -> dict:
    """Drop bare ``Tool loaded.`` text blocks from user turns that contain a
    tool_reference-bearing tool_result.

    Port of copilot-api's ``stripToolReferenceTurnBoundary``.
    """
    if not isinstance(body, dict):
        return body
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        has_tool_ref = any(
            isinstance(blk, dict) and blk.get("type") == "tool_result" and _has_tool_reference(blk)
            for blk in content
        )
        if not has_tool_ref:
            continue
        filtered = []
        for blk in content:
            if (
                isinstance(blk, dict)
                and blk.get("type") == "text"
                and isinstance(blk.get("text"), str)
                and blk["text"].strip() == TOOL_REFERENCE_TURN_BOUNDARY
            ):
                continue
            filtered.append(blk)
        msg["content"] = filtered
    return body


# ---------------------------------------------------------------------------
# 4. Sanitize VS Code IDE tools
# ---------------------------------------------------------------------------

def sanitize_ide_tools(body: dict) -> dict:
    """Drop ``mcp__ide__executeCode`` unconditionally and
    rewrite ``mcp__ide__getDiagnostics`` description to match the VS Code
    Claude agent expectations.
    """
    if not isinstance(body, dict):
        return body
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return body

    new_tools: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            new_tools.append(tool)
            continue
        name = tool.get("name")
        if isinstance(name, str) and name.strip().lower() == _IDE_EXECUTE_CODE_TOOL_NORMALIZED:
            continue
        if name == IDE_GET_DIAGNOSTICS_TOOL:
            patched = dict(tool)
            patched["description"] = IDE_GET_DIAGNOSTICS_DESCRIPTION
            new_tools.append(patched)
            continue
        new_tools.append(tool)
    body["tools"] = new_tools
    return body


# ---------------------------------------------------------------------------
# 5. Adaptive thinking
# ---------------------------------------------------------------------------

def budget_tokens_to_effort(budget_tokens: int | None) -> str:
    """Map an Anthropic ``thinking.budget_tokens`` value to a Copilot effort."""
    if not isinstance(budget_tokens, int) or budget_tokens <= 0:
        return "low"
    if budget_tokens <= 1024:
        return "low"
    if budget_tokens <= 8192:
        return "medium"
    return "high"


def _tool_choice_blocks_thinking(tool_choice: Any) -> bool:
    if not isinstance(tool_choice, dict):
        return False
    return tool_choice.get("type") in ("any", "tool")


def apply_adaptive_thinking(
    body: dict,
    supports_adaptive: bool,
    *,
    reasoning_efforts: Iterable[str] | None = None,
) -> dict:
    """Apply adaptive thinking config when the model supports it.

    Mirrors copilot-api's ``prepareMessagesApiPayload``: when the model
    supports adaptive thinking and the request is not forcing tool use,
    set ``thinking = {type: adaptive}`` and a derived
    ``output_config.effort``. ``display: summarized`` is added only when
    the original request did not already specify a ``thinking`` field, so
    explicit client preferences are preserved.

    Effort precedence:
      1. ``thinking.budget_tokens`` when provided (legacy Anthropic hint)
      2. An incoming ``output_config.effort`` already set on the body
         (e.g. by the Responses-to-Messages translator from
         ``reasoning.effort``). This preserves the caller's reasoning
         preference instead of clobbering it with the default.
      3. The per-model default, which mirrors the TS reference's
         ``getReasoningEffortForModel`` fallback of ``"high"``.

    The resulting effort is then clamped to the model's supported
    ``reasoning_efforts`` whitelist (if provided) and run through
    :func:`map_effort_for_model` for any per-model remapping.
    """
    if not isinstance(body, dict) or not supports_adaptive:
        return body

    # No-op for completely empty payloads (no model and no messages).
    if not body.get("model") and not body.get("messages"):
        return body

    if _tool_choice_blocks_thinking(body.get("tool_choice")):
        return body

    incoming_thinking = body.get("thinking")
    had_thinking = isinstance(incoming_thinking, dict)

    # Derive an effort hint from a legacy budget_tokens value when present.
    budget = None
    if had_thinking:
        candidate = incoming_thinking.get("budget_tokens")
        if isinstance(candidate, int):
            budget = candidate

    # Preserve any effort already placed on the body by an upstream
    # translator (e.g. responses_request_to_anthropic_messages maps
    # reasoning.effort to output_config.effort BEFORE preprocess runs).
    output_config = body.get("output_config")
    incoming_effort: str | None = None
    if isinstance(output_config, dict):
        oc_effort = output_config.get("effort")
        if isinstance(oc_effort, str) and oc_effort:
            incoming_effort = oc_effort.strip().lower() or None

    new_thinking: dict = {"type": "adaptive"}
    if not had_thinking:
        new_thinking["display"] = "summarized"
    # Always summarize for opus-4.7 to match the upstream reference behavior.
    model_name = body.get("model")
    if isinstance(model_name, str) and model_name.lower().endswith("claude-opus-4.7"):
        new_thinking["display"] = "summarized"
    body["thinking"] = new_thinking

    # Effort resolution: budget -> existing output_config -> default "high"
    if budget is not None:
        effort = budget_tokens_to_effort(budget)
    elif incoming_effort:
        effort = incoming_effort
    else:
        effort = "high"

    # Map "none"/"minimal" effort levels to "low" to match TS reference.
    if effort in ("none", "minimal"):
        effort = "low"
    mapped = map_effort_for_model(model_name, effort) or effort
    if mapped in ("none", "minimal"):
        mapped = "low"

    # Clamp to the per-model supported reasoning_efforts whitelist if known.
    if reasoning_efforts:
        allowed = [str(e).strip().lower() for e in reasoning_efforts if isinstance(e, str) and e.strip()]
        if allowed and mapped not in allowed:
            mapped = allowed[-1]

    if isinstance(output_config, dict):
        output_config["effort"] = mapped
    else:
        body["output_config"] = {"effort": mapped}
    return body


# ---------------------------------------------------------------------------
# 6. Compaction detection (port of copilot-api/lib/compact.ts + getCompactType)
# ---------------------------------------------------------------------------

def _compact_candidate_text(message: Any) -> str:
    if not isinstance(message, dict) or message.get("role") != "user":
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for blk in content:
        if not isinstance(blk, dict) or blk.get("type") != "text":
            continue
        text = blk.get("text")
        if not isinstance(text, str):
            continue
        if text.startswith("<system-reminder>"):
            continue
        parts.append(text)
    return "\n\n".join(parts)


def _is_compact_summary_request(last_message: Any) -> bool:
    text = _compact_candidate_text(last_message)
    if not text:
        return False
    return (
        _COMPACT_TEXT_ONLY_GUARD in text
        and _COMPACT_SUMMARY_PROMPT_START in text
        and any(section in text for section in _COMPACT_MESSAGE_SECTIONS)
    )


def _is_compact_auto_continue(last_message: Any) -> bool:
    text = _compact_candidate_text(last_message)
    if not text:
        return False
    return any(text.startswith(p) for p in _COMPACT_AUTO_CONTINUE_PROMPT_STARTS)


def detect_compact_type(body: Any) -> int:
    """Return ``COMPACT_REQUEST``, ``COMPACT_AUTO_CONTINUE``, or 0.

    Port of copilot-api's ``getCompactType``.
    """
    if not isinstance(body, dict):
        return 0
    messages = body.get("messages")
    last_message = None
    if isinstance(messages, list) and messages:
        last_message = messages[-1]

    if last_message is not None and _is_compact_summary_request(last_message):
        return COMPACT_REQUEST
    if last_message is not None and _is_compact_auto_continue(last_message):
        return COMPACT_AUTO_CONTINUE

    system = body.get("system")
    if isinstance(system, str):
        return COMPACT_REQUEST if system.startswith(_COMPACT_SYSTEM_PROMPT_START) else 0
    if isinstance(system, list):
        for blk in system:
            if not isinstance(blk, dict):
                continue
            text = blk.get("text")
            if isinstance(text, str) and text.startswith(_COMPACT_SYSTEM_PROMPT_START):
                return COMPACT_REQUEST
    return 0


# ---------------------------------------------------------------------------
# 7. Orchestrator
# ---------------------------------------------------------------------------

def prepare_messages_passthrough_payload(
    body: dict,
    *,
    model_supports_adaptive: bool,
    is_compact: bool = False,
    reasoning_efforts: Iterable[str] | None = None,
) -> dict:
    """Run the full passthrough preprocessing pipeline on a deep copy.

    Pipeline order mirrors the TS handler + prepareMessagesApiPayload:
      sanitizeIdeTools -> stripToolReferenceTurnBoundary ->
      mergeToolResultForClaude(skipLastMessage=is_compact==COMPACT_REQUEST) ->
      stripCacheControlScope -> filterAssistantThinkingBlocks ->
      applyAdaptiveThinking -> clampTemperature.
    """
    cleaned = copy.deepcopy(body) if isinstance(body, dict) else body
    sanitize_ide_tools(cleaned)
    strip_tool_reference_turn_boundary(cleaned)
    merge_tool_result_with_reminder(cleaned, skip_last_message=bool(is_compact))
    strip_cache_control_scope(cleaned)
    filter_assistant_thinking_placeholders(cleaned)
    apply_adaptive_thinking(
        cleaned,
        model_supports_adaptive,
        reasoning_efforts=reasoning_efforts,
    )
    clamp_temperature_for_claude(cleaned)
    return cleaned


def clamp_temperature_for_claude(body: dict) -> None:
    """Clamp ``temperature`` to [0, 1]; Anthropic models reject 1.5/2.0."""
    if not isinstance(body, dict):
        return
    temp = body.get("temperature")
    if isinstance(temp, (int, float)):
        if temp < 0:
            body["temperature"] = 0
        elif temp > 1:
            body["temperature"] = 1
