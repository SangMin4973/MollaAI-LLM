import json

from dataset.core.categories import category_guidance
from dataset.core.config import (
    DATASET_SYSTEM_PROMPT,
    build_conversation_system_prompt,
    DPOGenerationOptions,
    SFTGenerationOptions,
)


COMMON_ASSISTANT_RULES = """Assistant behavior:
- Friendly spoken English conversation partner.
- Prioritize conversation flow over grammar correction.
- Sound natural for TTS and spoken dialogue.
- No textbook or teacher tone unless the user asks for correction.
- If the user's English is understandable, respond to the meaning first.
- If correction helps, keep it brief and gentle in one sentence.
- Do not invent background information.
- Keep replies short, usually 1-3 sentences.
- Use simple punctuation and at most one follow-up question."""


def format_seed_constraints(options: SFTGenerationOptions) -> str:
    return (
        "Seed-specific goal:\n"
        f"- Behavior: {options.target_behavior}\n"
        f"- Tone: {options.assistant_tone}\n"
        f"- Correction: {options.should_correct}\n"
        f"- Notes: {options.notes or 'None'}"
    )


def build_dataset_messages(user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DATASET_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_sft_prompt(options: SFTGenerationOptions) -> list[dict[str, str]]:
    conversation_guide = category_guidance(options.conversation_type)
    final_system_prompt = build_conversation_system_prompt(options.user_level)
    user_prompt = f"""You are creating training data for a spoken English conversation assistant.

Goal:
Create natural spoken English conversations between a user and an assistant.

{COMMON_ASSISTANT_RULES}

User profile:
- English learner at level {options.user_level}.
- User may make mistakes, hesitate, self-correct, mix Korean and English, or speak in short fragments.

Conversation type:
{conversation_guide}

{format_seed_constraints(options)}

First user message:
"{options.seed_text.strip()}"

Task:
- Create {options.variation_count} different conversations.
- Each conversation should have {options.min_turns}-{options.max_turns} turns total.
- Vary wording and follow-up style.
- Treat the user message as spoken input, not polished text.
- Some user turns can stay short or slightly ungrammatical if natural.
- Follow the requested behavior, tone, and correction policy.
- Return plain text only.
- Do not return JSON.
- Write exactly {options.variation_count} conversations.
- Start each conversation with: ### Conversation N
- Then write one turn per line using only:
  User: ...
  Assistant: ...
- Do not include a System line in the output.
- The first line after each header must be: User: {options.seed_text.strip()}
- Keep valid speaker order.
- Separate conversations with one blank line.
- Final sample system prompt:
  {final_system_prompt}
"""
    return build_dataset_messages(user_prompt)


def build_sft_continuation_prompt(
    options: SFTGenerationOptions,
    transcript_lines: list[str],
    turns_to_generate: int,
) -> list[dict[str, str]]:
    conversation_guide = category_guidance(options.conversation_type)
    final_system_prompt = build_conversation_system_prompt(options.user_level)
    transcript = "\n".join(transcript_lines)
    expected_format = "Assistant: ...\nUser: ..." if turns_to_generate == 2 else "Assistant: ..."

    user_prompt = f"""You are creating training data for a spoken English conversation assistant.

Goal:
Continue one natural spoken English conversation between a user and an assistant.

{COMMON_ASSISTANT_RULES}

User profile:
- English learner at level {options.user_level}.
- User may make mistakes, hesitate, self-correct, mix Korean and English, or speak in short fragments.

Conversation type:
{conversation_guide}

{format_seed_constraints(options)}

Final training system prompt:
{final_system_prompt}

Current conversation transcript:
{transcript}

Task:
- Continue this exact conversation.
- Generate exactly {turns_to_generate} next turn(s).
- First line must be Assistant: ...
- Alternate speakers correctly.
- Use only User: and Assistant:
- Return plain text only.
- Do not restart or repeat the transcript.
- Output format:
{expected_format}
"""
    return build_dataset_messages(user_prompt)


def build_dpo_prompt(options: DPOGenerationOptions) -> list[dict[str, str]]:
    system_prompt = options.system_prompt or build_conversation_system_prompt(options.user_level)
    prompt_payload = [{"role": "system", "content": system_prompt}]
    if options.context_messages:
        prompt_payload.extend(options.context_messages)
    prompt_payload.append({"role": "user", "content": options.user_message.strip()})
    correction_policy = str(options.should_correct)

    user_prompt = (
        "Create one DPO training example for a spoken English conversation assistant.\n\n"
        "Requirements:\n"
        "- Chosen: natural, concise, spoken, level-appropriate.\n"
        f"- Chosen behavior: {options.target_behavior}.\n"
        f"- Chosen tone: {options.assistant_tone}.\n"
        f"- Correction policy: {correction_policy}.\n"
        f"- Notes: {options.notes or 'None'}.\n"
        f"- User level: {options.user_level}.\n"
        "- Rejected: plausible but clearly worse, such as over-correction, textbook tone, verbosity, weak context use, or invented assumptions.\n"
        "- Return one JSON object only.\n"
        "- Use this schema exactly:\n"
        "{\n"
        '  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],\n'
        '  "chosen": [{"role": "assistant", "content": "..."}],\n'
        '  "rejected": [{"role": "assistant", "content": "..."}]\n'
        "}\n\n"
        f"Prompt messages:\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )
    return build_dataset_messages(user_prompt)
