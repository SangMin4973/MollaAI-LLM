from dataclasses import dataclass

MODEL_NAME = "Qwen/Qwen3-8B"

DATASET_SYSTEM_PROMPT = (
    "Create training data for a spoken English conversation assistant. "
    "Follow the requested format exactly. No explanations or extra text."
)

BASE_CONVERSATION_SYSTEM_PROMPT = (
    "You are a friendly spoken English conversation partner. "
    "Prioritize natural conversation, short speakable replies, and gentle correction only when useful."
)

LEVEL_SYSTEM_GUIDANCE = {
    "A": (
        "Adapt to an A-level learner. Use very simple vocabulary, short sentences, and easy follow-up "
        "questions. Keep corrections brief and direct."
    ),
    "B": (
        "Adapt to a B-level learner. Use clear everyday English, natural spoken phrasing, and short "
        "follow-up questions. Keep corrections brief and natural."
    ),
    "C": (
        "Adapt to a C-level learner. Use more flexible and natural phrasing, but stay concise and "
        "conversational. Correct only when it clearly improves naturalness."
    ),
}


def build_conversation_system_prompt(user_level: str = "B") -> str:
    normalized_level = user_level.upper()
    level_guidance = LEVEL_SYSTEM_GUIDANCE.get(normalized_level, LEVEL_SYSTEM_GUIDANCE["B"])
    return f"{BASE_CONVERSATION_SYSTEM_PROMPT} {level_guidance}"


DEFAULT_DPO_SYSTEM_PROMPT = build_conversation_system_prompt()
DEFAULT_SFT_CONVERSATION_SYSTEM_PROMPT = DEFAULT_DPO_SYSTEM_PROMPT


@dataclass(slots=True)
class SFTGenerationOptions:
    seed_text: str
    conversation_type: str
    user_level: str = "B"
    variation_count: int = 5
    min_turns: int = 4
    max_turns: int = 8
    target_behavior: str = "natural_conversation"
    assistant_tone: str = "casual"
    should_correct: bool | str = "optional"
    notes: str = ""


@dataclass(slots=True)
class DPOGenerationOptions:
    user_message: str
    system_prompt: str | None = None
    context_messages: list[dict[str, str]] | None = None
    user_level: str = "B"
    target_behavior: str = "natural_conversation"
    assistant_tone: str = "casual"
    should_correct: bool | str = "optional"
    notes: str = ""
