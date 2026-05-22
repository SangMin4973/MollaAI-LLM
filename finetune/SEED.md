# Seed 설계 요구사항 문서

## 1. 목적

이 문서는 **영어로 자연스럽게 대화하는 챗봇**을 파인튜닝하기 위한 seed 데이터를 설계하는 기준을 정의한다.

Seed는 최종 학습 데이터가 아니라, LLM으로 대화 데이터를 생성하기 위한 **출발점**이다.

좋은 seed는 다음을 가능하게 해야 한다.

- 다양한 영어 대화 상황 생성
- 비원어민 사용자의 실제 입력 반영
- 자연스러운 영어 대화 흐름 학습
- 과도한 문법 교정 방지
- 대화 끊김, 모호한 입력, 어색한 영어 처리 능력 강화

---

## 2. Seed의 기본 정의

Seed는 하나의 사용자 발화와 그 발화의 메타데이터로 구성된다.

예시:

```python
SeedExample(
    id=1,
    category="awkward_english",
    level="B1",
    user_text="I am boring today.",
    target_behavior="brief_correction_then_continue",
    tone="casual",
    should_correct=True
)
```

---

## 3. 필수 필드

### 3.1 `id`

Seed를 구분하기 위한 고유 번호.

```python
id=1
```

요구사항:

- 정수형 사용
- 중복 금지
- 생성 순서와 추적이 가능해야 함

---

### 3.2 `category`

Seed가 속한 대화 유형.

```python
category="awkward_english"
```

요구사항:

- 사전에 정의된 카테고리만 사용
- 하나의 seed에는 하나의 주요 카테고리만 부여
- 너무 넓거나 모호한 카테고리는 피함

예시 카테고리:

```python
[
    "small_talk",
    "awkward_english",
    "topic_request",
    "spoken_short_reply",
    "emotion_checkin",
    "conversation_recovery",
    "correction_request",
    "confidence_issue",
    "mixed_language",
    "natural_expression"
]
```

---

### 3.3 `level`

사용자의 영어 수준.

```python
level="B1"
```

권장 레벨:

| Level | 설명 |
|---|---|
| A2 | 짧고 단순한 문장, 오류 많음 |
| B1 | 기본 대화 가능, 문법 오류 있음 |
| B2 | 비교적 자연스러운 대화 가능 |
| C1 | 복잡한 주제 가능 |
| native_like | 자연스러운 구어체, slang 가능 |

권장 비율:

| Level | 비율 |
|---|---:|
| A2 | 20% |
| B1 | 35% |
| B2 | 30% |
| C1 | 10% |
| native_like | 5% |

---

### 3.4 `user_text`

사용자의 첫 발화.

```python
user_text="I am boring today."
```

요구사항:

- 실제 사용자가 입력할 법한 문장이어야 함
- 완벽한 영어일 필요 없음
- 짧은 발화, 어색한 영어, 혼합 언어도 포함
- 너무 인위적이거나 교과서적인 문장은 피함

좋은 예시:

```text
I am boring today.
I don't know what to talk about.
Maybe.
오늘 너무 tired 해서 아무것도 하기 싫어.
Can you give me a topic?
I feel nervous when I speak English.
```

나쁜 예시:

```text
I would like to engage in a sophisticated English conversation.
Please initiate a grammatically correct dialogue with me.
Today, I experienced a moderate level of emotional fatigue.
```

---

## 4. 권장 필드

### 4.1 `target_behavior`

assistant가 이 seed에서 보여야 하는 핵심 행동.

```python
target_behavior="brief_correction_then_continue"
```

예시 값:

```python
[
    "natural_conversation",
    "brief_correction_then_continue",
    "ask_simple_followup",
    "offer_topic",
    "repair_conversation",
    "encourage_confidence",
    "explain_naturally",
    "roleplay_start",
    "soften_expression",
    "respond_with_empathy"
]
```

예시:

| Seed | target_behavior |
|---|---|
| `I am boring today.` | `brief_correction_then_continue` |
| `Can you give me a topic?` | `offer_topic` |
| `Maybe.` | `ask_simple_followup` |
| `I feel nervous when I speak English.` | `encourage_confidence` |
| `Wait, that's not what I meant.` | `repair_conversation` |

---

### 4.2 `tone`

assistant가 사용할 응답 톤.

```python
tone="casual"
```

권장 값:

```python
[
    "casual",
    "warm",
    "playful",
    "supportive",
    "simple",
    "neutral",
    "encouraging"
]
```

카테고리별 권장 톤:

| Category | 권장 tone |
|---|---|
| `small_talk` | `casual`, `playful` |
| `emotion_checkin` | `warm`, `supportive` |
| `awkward_english` | `casual`, `encouraging` |
| `correction_request` | `simple`, `neutral` |
| `confidence_issue` | `supportive`, `encouraging` |
| `humor_banter` | `playful` |
| `conversation_recovery` | `casual`, `warm` |

---

### 4.3 `should_correct`

assistant가 사용자의 영어를 교정해야 하는지 여부.

```python
should_correct=True
```

권장 값:

| 값 | 의미 |
|---|---|
| `True` | 짧게 교정해야 함 |
| `False` | 교정하지 않고 대화 우선 |
| `"optional"` | 필요하면 부드럽게 교정 |

판단 기준:

| 상황 | should_correct |
|---|---|
| 사용자가 직접 교정을 요청함 | `True` |
| 의미가 크게 어색함 | `True` |
| 의미는 이해 가능하고 대화 흐름이 중요함 | `"optional"` |
| 감정 대화 중 사소한 오류 | `False` |
| 짧은 잡담 또는 유머 | `False` |

예시:

```python
SeedExample(
    id=1,
    category="awkward_english",
    level="B1",
    user_text="I am boring today.",
    target_behavior="brief_correction_then_continue",
    tone="casual",
    should_correct=True
)
```

```python
SeedExample(
    id=2,
    category="emotion_checkin",
    level="B1",
    user_text="I am tired but I cannot sleep well.",
    target_behavior="respond_with_empathy",
    tone="warm",
    should_correct=False
)
```

---

## 5. 선택 필드

### 5.1 `variation_count`

해당 seed에서 생성할 대화 후보 개수.

```python
variation_count=6
```

권장값:

| Seed 중요도 | variation_count |
|---|---:|
| 핵심 카테고리 | 6~8 |
| 일반 카테고리 | 4~6 |
| 보조 카테고리 | 2~4 |

---

### 5.2 `notes`

데이터 생성 시 주의할 점.

```python
notes="Do not over-correct. Keep the conversation natural."
```

예시:

```python
notes="The assistant should not explain grammar too much."
notes="The assistant should offer an easy topic."
notes="The assistant should respond in simple English."
notes="The assistant may briefly explain in Korean if needed."
```

---

## 6. 권장 Seed 클래스 구조

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class SeedExample:
    id: int
    category: str
    level: str
    user_text: str
    target_behavior: str
    tone: str
    should_correct: bool | str
    variation_count: int = 6
    notes: str = ""
```

예시:

```python
SEED_EXAMPLES: list[SeedExample] = [
    SeedExample(
        id=1,
        category="awkward_english",
        level="B1",
        user_text="I am boring today.",
        target_behavior="brief_correction_then_continue",
        tone="casual",
        should_correct=True,
        variation_count=6,
        notes="Correct briefly, then continue the conversation."
    ),
    SeedExample(
        id=2,
        category="topic_request",
        level="A2",
        user_text="Can you give me a topic?",
        target_behavior="offer_topic",
        tone="simple",
        should_correct=False,
        variation_count=6,
        notes="Offer one easy topic and a simple question."
    )
]
```

---

## 7. 핵심 카테고리

초기 seed bank에서 반드시 포함해야 하는 핵심 카테고리는 다음과 같다.

```python
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
```

---

## 8. 전체 권장 카테고리

```python
CATEGORIES = [
    "small_talk",
    "emotion_checkin",
    "opinion_chat",
    "awkward_english",
    "spoken_short_reply",
    "long_context",
    "humor_banter",
    "correction_request",
    "hesitation_repair",
    "mixed_language",
    "safety_boundary",

    "topic_request",
    "confidence_issue",
    "social_english",
    "daily_routine",
    "preference_chat",
    "clarification_request",
    "conversation_recovery",
    "casual_slang",
    "roleplay_conversation",
    "natural_expression",
    "repair_my_sentence",
    "learning_preference",
    "culture_chat",
    "message_reply",
    "pronunciation_talk",
]
```

---

## 9. 카테고리 그룹 구조

```python
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
```

---

## 10. Seed 수량 요구사항

### 단계별 권장 규모

| 단계 | Seed 개수 | 목적 |
|---|---:|---|
| 테스트 | 20~50개 | 생성 파이프라인 확인 |
| PoC | 200~300개 | 카테고리별 품질 확인 |
| MVP | 600개 이상 | SFT 3,000개 이상 생성 |
| 고도화 | 1,000개 이상 | 다양성 확대 및 실패 케이스 보강 |

초기 MVP 기준으로는 **600개 seed**를 권장한다.

---

## 11. Seed 카테고리별 권장 비율

600개 seed 기준:

| 카테고리 그룹 | 비율 | 개수 |
|---|---:|---:|
| learner_support | 30% | 180 |
| general_conversation | 25% | 150 |
| conversation_management | 20% | 120 |
| emotional_support | 10% | 60 |
| practical_english | 10% | 60 |
| safety | 5% | 30 |

---

## 12. 좋은 Seed 기준

좋은 seed는 다음 조건을 만족해야 한다.

```text
1. 실제 사용자가 입력할 법하다.
2. 너무 완벽한 영어만 포함하지 않는다.
3. 하나의 명확한 카테고리에 속한다.
4. assistant의 목표 행동이 분명하다.
5. 여러 변형 대화를 만들 수 있다.
6. 영어 학습자의 실제 어려움을 반영한다.
7. 대화가 자연스럽게 이어질 여지가 있다.
```

좋은 예시:

```python
SeedExample(
    id=101,
    category="confidence_issue",
    level="B1",
    user_text="I understand English, but I can't speak well.",
    target_behavior="encourage_confidence",
    tone="supportive",
    should_correct=False
)
```

---

## 13. 나쁜 Seed 기준

다음과 같은 seed는 피한다.

```text
1. 너무 인위적인 문장
2. 너무 긴 설명형 입력
3. 카테고리가 불명확한 입력
4. assistant가 할 행동이 모호한 입력
5. 실제 사용 빈도가 낮은 표현
6. 학습 목표와 관련 없는 일반 지식 질문
7. 위험하거나 민감하지만 안전 설계가 없는 입력
```

나쁜 예시:

```python
SeedExample(
    id=999,
    category="small_talk",
    level="B2",
    user_text="Please generate a comprehensive dialogue concerning the philosophical implications of weather.",
    target_behavior="natural_conversation",
    tone="casual",
    should_correct=False
)
```

---

## 14. Seed 작성 시 주의사항

### 14.1 완벽한 영어만 넣지 않는다

실제 사용자는 다음처럼 말할 수 있다.

```text
I am boring.
I don't know what should I say.
I want speak English better.
My condition is not good.
```

이런 입력은 반드시 포함해야 한다.

---

### 14.2 교정 요청과 일반 대화를 분리한다

```text
Can you correct my English?
```

위 문장은 `correction_request`.

```text
I am boring today.
```

위 문장은 `awkward_english`.

두 카테고리를 섞지 않는다.

---

### 14.3 mixed language를 포함한다

한국어 사용자는 영어와 한국어를 섞어 말할 수 있다.

```text
오늘 너무 tired 해서 아무것도 하기 싫어.
I want to speak more naturally, but 자꾸 말이 끊겨.
```

이런 seed는 실제 서비스 품질에 중요하다.

---

### 14.4 짧고 모호한 입력을 충분히 넣는다

```text
Maybe.
Not really.
idk.
lol.
I guess.
Never mind.
```

이 카테고리는 대화 복구 능력을 높인다.

---

### 14.5 안전 관련 seed는 별도 관리한다

`safety_boundary` 카테고리는 반드시 별도 검수한다.

예시:

```text
How do I insult someone in English?
How can I threaten someone politely?
I want to say something really hurtful.
```

이런 seed는 모델이 위험하거나 공격적인 표현을 도와주지 않도록 설계해야 한다.

---

## 15. 최종 예시 Seed 리스트

```python
SEED_EXAMPLES: list[SeedExample] = [
    SeedExample(
        id=1,
        category="small_talk",
        level="B1",
        user_text="I had coffee three times today.",
        target_behavior="natural_conversation",
        tone="playful",
        should_correct=False,
        variation_count=6,
        notes="Keep it casual and fun."
    ),
    SeedExample(
        id=2,
        category="awkward_english",
        level="B1",
        user_text="I am boring today.",
        target_behavior="brief_correction_then_continue",
        tone="casual",
        should_correct=True,
        variation_count=6,
        notes="Briefly correct 'bored' and continue the conversation."
    ),
    SeedExample(
        id=3,
        category="topic_request",
        level="A2",
        user_text="Can you give me a topic?",
        target_behavior="offer_topic",
        tone="simple",
        should_correct=False,
        variation_count=6,
        notes="Offer one easy topic, not a long list."
    ),
    SeedExample(
        id=4,
        category="confidence_issue",
        level="B1",
        user_text="I feel nervous when I speak English.",
        target_behavior="encourage_confidence",
        tone="supportive",
        should_correct=False,
        variation_count=6,
        notes="Encourage the user without sounding like a therapist."
    ),
    SeedExample(
        id=5,
        category="conversation_recovery",
        level="B1",
        user_text="Sorry, I lost my train of thought.",
        target_behavior="repair_conversation",
        tone="casual",
        should_correct=False,
        variation_count=6,
        notes="Help reset the conversation naturally."
    )
]
```

---

## 16. 최종 요구사항 요약

```text
Seed 설계의 핵심 목표:
영어 학습자가 실제로 입력할 법한 다양한 발화를 기반으로,
자연스러운 영어 대화 데이터를 안정적으로 생성하는 것.

필수 요구사항:
- category를 명확히 지정한다.
- user level을 포함한다.
- user_text는 실제적이어야 한다.
- target_behavior를 지정한다.
- correction 여부를 명시한다.
- 너무 완벽한 영어만 넣지 않는다.
- 짧고 모호한 발화, 어색한 영어, mixed language를 포함한다.
- safety seed는 별도로 검수한다.

권장 초기 규모:
- 테스트: 20~50개
- PoC: 200~300개
- MVP: 600개 이상
```