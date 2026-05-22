import json
import re
import traceback
from typing import Any

from dataset.core.config import build_conversation_system_prompt


def answer_extraction(response: str) -> str:
    try:
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant", 1)[1]

        if "</think>" in response:
            think_end = response.find("</think>") + len("</think>")
            response = response[think_end:]

        response = response.strip()
        if response.startswith("```"):
            lines = response.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines).strip()

        return response
    except Exception:
        print("Error in answer_extraction")
        print(traceback.format_exc())
        return response


def coerce_json_value(text: str) -> Any:
    candidate = text.strip()
    if not candidate:
        raise ValueError("Empty model output")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start_positions = [idx for idx in (candidate.find("["), candidate.find("{")) if idx != -1]
    if not start_positions:
        raise ValueError("No JSON object or array found in model output")

    start = min(start_positions)
    stack: list[str] = []
    closing_map = {"{": "}", "[": "]"}

    for index in range(start, len(candidate)):
        char = candidate[index]
        if char in closing_map:
            stack.append(closing_map[char])
        elif char in ("}", "]"):
            if not stack or char != stack[-1]:
                continue
            stack.pop()
            if not stack:
                snippet = candidate[start:index + 1]
                return json.loads(snippet)

    raise ValueError("Could not extract a valid JSON payload from model output")


def normalize_record_list(
    payload: Any,
    required_top_level_key: str | None = None,
) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if required_top_level_key and required_top_level_key in payload:
            nested = payload[required_top_level_key]
            if not isinstance(nested, list):
                raise ValueError(f"'{required_top_level_key}' must be a list")
            records = nested
        else:
            records = [payload]
    else:
        raise ValueError("Parsed payload must be a dict or list")

    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each record must be a JSON object")
        normalized.append(record)
    return normalized


def parse_json_records(
    text: str,
    required_top_level_key: str | None = None,
) -> list[dict[str, Any]]:
    payload = coerce_json_value(text)
    return normalize_record_list(payload, required_top_level_key=required_top_level_key)


def parse_sft_plaintext_records(
    text: str,
    expected_first_user_text: str,
    user_level: str = "B",
) -> list[dict[str, Any]]:
    candidate = text.strip()
    if not candidate:
        raise ValueError("Empty model output")

    blocks = re.split(r"(?im)^###\s*conversation\s*\d+\s*$", candidate)
    conversations = [block.strip() for block in blocks if block.strip()]
    if not conversations:
        raise ValueError("No conversation blocks found in model output")

    normalized_first_user_text = expected_first_user_text.strip()
    records: list[dict[str, Any]] = []

    for conversation in conversations:
        lines = [line.strip() for line in conversation.splitlines() if line.strip()]
        messages: list[dict[str, str]] = [
            {"role": "system", "content": build_conversation_system_prompt(user_level)}
        ]

        for line in lines:
            speaker_match = re.match(
                r"^(user|assistant|assistants|assistents)\s*:\s*(.*)$",
                line,
                flags=re.IGNORECASE,
            )
            if speaker_match:
                speaker = speaker_match.group(1).lower()
                content = speaker_match.group(2).strip()
                if not content:
                    continue
                role = "user" if speaker == "user" else "assistant"
                messages.append({"role": role, "content": content})
                continue
            raise ValueError(f"Unexpected line in conversation block: {line}")

        if len(messages) < 5:
            raise ValueError("Conversation must contain at least two user/assistant exchanges")
        if messages[1]["role"] != "user":
            raise ValueError("Conversation must begin with a user turn")
        messages[1]["content"] = normalized_first_user_text

        previous_role = "system"
        for message in messages[1:]:
            if message["role"] == previous_role:
                raise ValueError("Consecutive turns from the same role are not allowed")
            previous_role = message["role"]

        records.append({"messages": messages})

    return records


def parse_sft_continuation_turns(
    text: str,
    expected_roles: list[str],
) -> list[dict[str, str]]:
    candidate = text.strip()
    if not candidate:
        raise ValueError("Empty continuation output")

    lines = [line.strip() for line in candidate.splitlines() if line.strip()]
    parsed_messages: list[dict[str, str]] = []

    for line in lines:
        speaker_match = re.match(
            r"^(user|assistant|assistants|assistents)\s*:\s*(.*)$",
            line,
            flags=re.IGNORECASE,
        )
        if not speaker_match:
            continue

        speaker = speaker_match.group(1).lower()
        content = speaker_match.group(2).strip()
        if not content:
            continue

        role = "user" if speaker == "user" else "assistant"
        parsed_messages.append({"role": role, "content": content})
        if len(parsed_messages) == len(expected_roles):
            break

    if len(parsed_messages) != len(expected_roles):
        raise ValueError("Did not find the expected number of continuation turns")

    for parsed_message, expected_role in zip(parsed_messages, expected_roles, strict=True):
        if parsed_message["role"] != expected_role:
            raise ValueError("Continuation output has the wrong speaker order")

    return parsed_messages
