from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConversationCategory:
    key: str
    label: str
    description: str


CORE_CATEGORIES = [
    "awkward_english",
    "small_talk",
    "topic_request",
    "spoken_short_reply",
    "emotion_checkin",
    "conversation_recovery",
    "correction_request",
    "confidence_issue",
]


CONVERSATION_CATEGORIES: dict[str, ConversationCategory] = {
    "small_talk": ConversationCategory(
        key="small_talk",
        label="Small talk",
        description="Natural everyday spoken topics like weather, food, coffee, plans, and routine.",
    ),
    "emotion_checkin": ConversationCategory(
        key="emotion_checkin",
        label="Emotion / status sharing",
        description="Empathetic spoken responses when the user shares feelings, mood, or energy.",
    ),
    "opinion_chat": ConversationCategory(
        key="opinion_chat",
        label="Opinion conversation",
        description="Back-and-forth spoken conversation about opinions, tradeoffs, and everyday views.",
    ),
    "awkward_english": ConversationCategory(
        key="awkward_english",
        label="Awkward English input",
        description="Understands non-native phrasing and keeps the conversation moving without over-correcting.",
    ),
    "spoken_short_reply": ConversationCategory(
        key="spoken_short_reply",
        label="Short spoken reply",
        description="Recovers naturally from very short spoken replies like 'maybe', 'not really', or 'I guess'.",
    ),
    "long_context": ConversationCategory(
        key="long_context",
        label="Long conversation continuity",
        description="Maintains context over longer multi-turn spoken conversations.",
    ),
    "humor_banter": ConversationCategory(
        key="humor_banter",
        label="Humor / light banter",
        description="Light, casual, playful spoken tone without becoming over-the-top.",
    ),
    "correction_request": ConversationCategory(
        key="correction_request",
        label="English correction request",
        description="Gives short, natural corrections only when the user explicitly asks.",
    ),
    "hesitation_repair": ConversationCategory(
        key="hesitation_repair",
        label="Hesitation / self-repair",
        description="Handles hesitations, self-corrections, and broken spoken phrasing naturally.",
    ),
    "mixed_language": ConversationCategory(
        key="mixed_language",
        label="Mixed Korean and English speech",
        description="Handles mixed Korean and English input without sounding confused or overly formal.",
    ),
    "safety_boundary": ConversationCategory(
        key="safety_boundary",
        label="Safety / boundary handling",
        description="Handles risky or sensitive requests with clear but calm boundaries.",
    ),
    "topic_request": ConversationCategory(
        key="topic_request",
        label="Topic request",
        description="Offers easy conversation topics and restarts stalled interactions smoothly.",
    ),
    "confidence_issue": ConversationCategory(
        key="confidence_issue",
        label="Confidence issue",
        description="Encourages nervous learners without sounding like a lecturer or therapist.",
    ),
    "social_english": ConversationCategory(
        key="social_english",
        label="Social English",
        description="Helps with casual expressions for greetings, invitations, and everyday social situations.",
    ),
    "daily_routine": ConversationCategory(
        key="daily_routine",
        label="Daily routine",
        description="Talks naturally about habits, schedules, chores, workdays, and ordinary routines.",
    ),
    "preference_chat": ConversationCategory(
        key="preference_chat",
        label="Preference chat",
        description="Explores likes, dislikes, and simple choices in a casual spoken style.",
    ),
    "clarification_request": ConversationCategory(
        key="clarification_request",
        label="Clarification request",
        description="Asks for or gives clarification when the user's meaning is incomplete or vague.",
    ),
    "conversation_recovery": ConversationCategory(
        key="conversation_recovery",
        label="Conversation recovery",
        description="Repairs awkward pauses, uncertainty, or stalled conversation and keeps things moving.",
    ),
    "casual_slang": ConversationCategory(
        key="casual_slang",
        label="Casual slang",
        description="Explains or uses light casual slang naturally without becoming exaggerated or confusing.",
    ),
    "roleplay_conversation": ConversationCategory(
        key="roleplay_conversation",
        label="Roleplay conversation",
        description="Starts light roleplay practice for realistic spoken situations like cafes or travel.",
    ),
    "natural_expression": ConversationCategory(
        key="natural_expression",
        label="Natural expression",
        description="Suggests more natural spoken wording while keeping explanations short and usable.",
    ),
    "repair_my_sentence": ConversationCategory(
        key="repair_my_sentence",
        label="Repair my sentence",
        description="Fixes the user's sentence directly and simply when they want help rewriting it.",
    ),
    "learning_preference": ConversationCategory(
        key="learning_preference",
        label="Learning preference",
        description="Adapts to how the user prefers to practice or receive feedback.",
    ),
    "culture_chat": ConversationCategory(
        key="culture_chat",
        label="Culture chat",
        description="Handles casual culture topics, habits, and differences without over-explaining.",
    ),
    "message_reply": ConversationCategory(
        key="message_reply",
        label="Message reply help",
        description="Helps the user write natural replies for everyday messages in a casual tone.",
    ),
    "pronunciation_talk": ConversationCategory(
        key="pronunciation_talk",
        label="Pronunciation talk",
        description="Gives simple spoken-friendly help around pronunciation difficulties.",
    ),
}


CATEGORY_GROUPS = {
    "general_conversation": [
        "small_talk",
        "daily_routine",
        "preference_chat",
        "opinion_chat",
        "humor_banter",
        "culture_chat",
    ],
    "learner_support": [
        "awkward_english",
        "correction_request",
        "repair_my_sentence",
        "natural_expression",
        "casual_slang",
        "pronunciation_talk",
    ],
    "conversation_management": [
        "topic_request",
        "spoken_short_reply",
        "hesitation_repair",
        "conversation_recovery",
        "clarification_request",
        "long_context",
    ],
    "emotional_support": [
        "emotion_checkin",
        "confidence_issue",
    ],
    "practical_english": [
        "social_english",
        "message_reply",
        "roleplay_conversation",
        "learning_preference",
        "mixed_language",
    ],
    "safety": [
        "safety_boundary",
    ],
}


def list_category_keys() -> list[str]:
    return list(CONVERSATION_CATEGORIES.keys())


def resolve_conversation_type(value: str) -> str:
    normalized = value.strip()
    if normalized in CONVERSATION_CATEGORIES:
        return CONVERSATION_CATEGORIES[normalized].label
    return normalized


def category_guidance(value: str) -> str:
    normalized = value.strip()
    if normalized in CONVERSATION_CATEGORIES:
        category = CONVERSATION_CATEGORIES[normalized]
        return f"{category.label}: {category.description}"
    return normalized
