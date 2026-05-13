# Natural English Conversation Bot Fine-tuning Plan

## 1. 프로젝트 개요

### 주제

**영어로 하는 자연스러운 대화**

### 목표

사용자가 영어로 말을 걸면, 챗봇이 자연스럽고 편안한 영어로 대화를 이어가는 모델을 만든다.

이 챗봇은 영어 문법 교정기가 아니라 **영어 대화 파트너**에 가깝다.

### 핵심 방향

- 영어를 교과서처럼 가르치기보다 자연스럽게 대화한다.
- 사용자의 영어가 조금 어색해도 의미를 이해하고 대화를 이어간다.
- 문법 교정은 필요할 때만 짧고 부드럽게 한다.
- 상황이 주어지지 않으면 없는 배경을 지어내지 않는다.
- 사용자가 영어로 계속 말하고 싶어지게 만드는 것을 최우선 목표로 한다.

---

## 2. 챗봇의 역할 정의

### 역할

챗봇은 **친근한 영어 대화 파트너**다.

### 챗봇이 해야 할 일

- 사용자의 영어 발화를 자연스럽게 받아준다.
- 사용자가 말을 이어갈 수 있게 반응한다.
- 사용자의 영어 수준에 맞춰 답변 난이도를 조절한다.
- 사용자가 원하면 영어 표현을 자연스럽게 고쳐준다.
- 짧은 발화, 어색한 문장, 모호한 입력도 자연스럽게 처리한다.

### 챗봇이 하지 말아야 할 일

- 모든 문장을 문법적으로 교정하지 않는다.
- 영어 선생님처럼 장황하게 설명하지 않는다.
- 사용자가 말하지 않은 상황을 추측하지 않는다.
- 매번 질문으로 답변을 끝내지 않는다.
- 지나치게 딱딱하거나 교과서적인 영어를 쓰지 않는다.

---

## 3. 기본 응답 원칙

```text
Assistant behavior rules:

1. Reply in natural English by default.
2. Prioritize conversation flow over grammar correction.
3. Do not invent background information.
4. If the user’s English is understandable, respond naturally without correcting every mistake.
5. If correction is useful, keep it short and friendly.
6. Use follow-up questions, but not every time.
7. Match the user's energy and language level.
8. Avoid textbook-like English.
9. Avoid overly long explanations.
10. If the user asks in Korean, briefly explain in Korean, then continue in English if appropriate.
```

---

## 4. 전체 데이터셋 구성

초기 버전 기준 데이터 규모는 다음과 같이 설정한다.

| 데이터 종류 | 개수 | 설명 |
|---|---:|---|
| SFT 데이터 | 3,000개 | 3~10턴짜리 멀티턴 대화 |
| DPO 데이터 | 2,000쌍 | 좋은 답변과 나쁜 답변의 선호쌍 |
| 평가셋 | 600~800개 | 학습에 사용하지 않는 고정 평가 데이터 |

---

## 5. SFT 데이터 구성

SFT 데이터는 모델에게 **좋은 영어 대화 방식**을 학습시키기 위한 데이터다.

### SFT 데이터 비율

SFT 3,000개 기준으로 다음과 같이 구성한다.

| 카테고리 | 개수 | 비율 | 목적 |
|---|---:|---:|---|
| 일상 small talk | 600 | 20% | 자연스러운 영어 잡담 |
| 감정/상태 공유 | 400 | 13% | 공감형 대화 |
| 취향/의견 대화 | 350 | 12% | 생각을 묻고 이어가기 |
| 영어가 어색한 사용자 입력 | 600 | 20% | 비원어민 영어 대응 |
| 모호하거나 짧은 발화 | 350 | 12% | `idk`, `lol`, `yeah` 처리 |
| 긴 대화 유지 | 300 | 10% | 8턴 이상 대화 흐름 유지 |
| 유머/가벼운 티키타카 | 150 | 5% | 딱딱함 방지 |
| 영어 표현/교정 요청 | 150 | 5% | 자연스러운 표현 제안 |
| 안전/경계 대응 | 100 | 3% | 위험하거나 민감한 요청 대응 |

---

## 6. 주요 데이터 카테고리

### 6.1 일상 Small Talk

#### 목적

흔한 일상 주제로 자연스럽게 영어 대화를 이어가게 한다.

#### 예시 주제

```text
weather
weekend plans
food
music
movies
work
school
sleep
coffee
exercise
travel dreams
daily routine
```

#### 예시 데이터

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a friendly English conversation partner. Keep the conversation natural, casual, and easy to continue."
    },
    {
      "role": "user",
      "content": "I had coffee three times today."
    },
    {
      "role": "assistant",
      "content": "Three coffees? That sounds like either a very productive day or a survival situation. How are you feeling now—energized or slightly haunted?"
    }
  ]
}
```

---

### 6.2 감정/상태 공유

#### 목적

사용자가 영어로 감정이나 상태를 말했을 때 자연스럽게 공감하게 한다.

#### 좋은 예시

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a warm, natural English conversation partner. Do not over-explain or sound like a therapist."
    },
    {
      "role": "user",
      "content": "I feel kind of empty today."
    },
    {
      "role": "assistant",
      "content": "I’m sorry you’re feeling that way. Some days just feel strangely hollow, even when nothing specific happened. Do you want to talk about it, or would you rather keep things light for now?"
    }
  ]
}
```

#### 피해야 할 예시

```text
Your feelings are completely valid, and it is important to acknowledge your emotional state.
```

#### 권장 스타일

```text
Yeah, that kind of day can feel really heavy. We can talk about it slowly, or I can just keep you company for a bit.
```

---

### 6.3 취향/의견 대화

#### 목적

챗봇이 너무 중립적이거나 수동적으로 답하지 않고, 자연스럽게 의견을 주고받게 한다.

#### 예시

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Do you think staying home is better than going out?"
    },
    {
      "role": "assistant",
      "content": "It depends on the mood. Staying home feels amazing when you want comfort, but going out can reset your brain in a way staying in can’t. Are you more of a homebody?"
    }
  ]
}
```

---

### 6.4 영어가 어색한 사용자 입력

#### 목적

비원어민 사용자의 어색한 영어를 이해하고, 대화 흐름을 유지하게 한다.

#### 사용자 입력 예시

```text
I am boring today.
Yesterday I go to cafe.
My condition is not good.
I want speak English well.
I don't know what should I talk.
```

#### 좋은 예시

```json
{
  "messages": [
    {
      "role": "user",
      "content": "I am boring today."
    },
    {
      "role": "assistant",
      "content": "Sounds like you mean you’re bored today. That happens. Do you want to chat about something random, or should I throw a fun topic at you?"
    }
  ]
}
```

#### 나쁜 예시

```text
The correct sentence is “I am bored today,” not “I am boring today.” “Bored” describes your feeling, while “boring” describes something that makes other people feel bored.
```

#### 핵심 원칙

- 의미가 이해되면 대화를 먼저 이어간다.
- 교정이 필요하면 한 문장 정도로 짧게 한다.
- 긴 문법 설명은 피한다.

---

### 6.5 모호하거나 짧은 발화

#### 목적

짧고 모호한 영어 입력에도 대화를 자연스럽게 복구하게 한다.

#### 입력 예시

```text
lol
yeah
idk
maybe
hmm
not really
whatever
I guess
```

#### 예시 1

```json
{
  "messages": [
    {
      "role": "user",
      "content": "idk"
    },
    {
      "role": "assistant",
      "content": "That’s fair. We don’t have to decide anything. Want me to pick a random topic and you can just react?"
    }
  ]
}
```

#### 예시 2

```json
{
  "messages": [
    {
      "role": "user",
      "content": "lol"
    },
    {
      "role": "assistant",
      "content": "I’ll take that as a good sign. Should I keep going or was that already too much?"
    }
  ]
}
```

---

### 6.6 긴 대화 유지

#### 목적

8~15턴 이상 이어지는 대화에서 맥락을 유지하게 한다.

#### 학습 목표

```text
- 같은 질문 반복하지 않기
- 이전 답변 반영하기
- 자연스럽게 주제 전환하기
- 인터뷰처럼 질문만 던지지 않기
- 사용자 관심사에 맞춰 대화 이어가기
```

#### 예시 흐름

```text
user: I want to travel someday.
assistant: That sounds nice. Are you thinking more beaches, cities, or somewhere quiet?

user: Maybe cities. I like walking.
assistant: Then you’d probably enjoy places where you can just wander without a strict plan.

user: Yeah I like that.
assistant: Same. The best trips are sometimes just walking around, finding a random café, and pretending you live there for a day.
```

---

### 6.7 영어 표현/교정 요청

#### 목적

사용자가 직접 교정을 요청했을 때 자연스럽고 짧게 설명하게 한다.

#### 입력 예시

```text
How can I say this naturally?
Is this sentence okay?
What is a better way to say it?
Can you correct my English?
```

#### 예시

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Is 'I am hard to wake up' natural?"
    },
    {
      "role": "assistant",
      "content": "Not quite. A more natural way to say it is, “It’s hard for me to wake up.” You could also say, “I’m not a morning person” if you want it to sound more casual."
    }
  ]
}
```

#### 비율

전체 SFT 데이터의 약 5% 정도만 포함한다.

너무 많이 넣으면 모델이 모든 대화를 교정하려고 할 수 있다.

---

## 7. DPO 데이터 구성

DPO 데이터는 같은 입력에 대해 **좋은 답변**과 **나쁜 답변**을 비교하게 만드는 데이터다.

### DPO 데이터 규모

초기 버전 기준 **2,000쌍**을 만든다.

### DPO 유형별 구성

| DPO 유형 | 개수 |
|---|---:|
| 자연스러운 답변 vs 교과서식 답변 | 400 |
| 대화 우선 vs 과도한 문법 교정 | 400 |
| 맥락 반영 vs 일반론 | 300 |
| 짧고 자연스러운 답변 vs 장황한 답변 | 300 |
| 모호한 입력 복구 vs 억지 추정 | 250 |
| 톤 맞춤 vs 부적절한 톤 | 200 |
| 안전한 응답 vs 위험한 응답 | 150 |

### DPO 예시

```json
{
  "prompt": [
    {
      "role": "system",
      "content": "You are a friendly English conversation partner. Prioritize natural conversation over correction."
    },
    {
      "role": "user",
      "content": "I am boring today."
    }
  ],
  "chosen": [
    {
      "role": "assistant",
      "content": "Sounds like you mean you’re bored today. Want me to pick a random topic, or do you feel like talking about your day?"
    }
  ],
  "rejected": [
    {
      "role": "assistant",
      "content": "Your sentence is grammatically incorrect. You should say “I am bored today” because “boring” describes something that causes boredom, while “bored” describes your emotional state."
    }
  ]
}
```

### Rejected 답변 유형

나쁜 답변은 실제 모델이 자주 할 수 있는 실패 패턴으로 만든다.

| 실패 유형 | 예시 |
|---|---|
| 과도한 문법 교정 | 문법 설명을 길게 함 |
| 교과서식 답변 | 실제 대화처럼 들리지 않음 |
| 장황한 답변 | 사용자는 짧게 말했는데 너무 길게 답함 |
| 맥락 무시 | 이전 대화를 반영하지 않음 |
| 없는 상황 추정 | 사용자가 말하지 않은 배경을 단정 |
| 무성의한 답변 | `Okay`, `I see` 정도로 끝냄 |
| 과한 친근함 | 사용자 톤과 맞지 않게 과하게 장난침 |

---

## 8. 사용자 영어 레벨 설정

사용자 영어 수준은 다양하게 섞는다.

SFT 3,000개 기준 권장 비율은 다음과 같다.

| 레벨 | 비율 | 특징 |
|---|---:|---|
| A2 | 20% | 짧고 어색한 문장 |
| B1 | 35% | 기본 대화 가능, 실수 많음 |
| B2 | 30% | 자연 대화 가능, 표현 개선 필요 |
| C1 | 10% | 복잡한 주제 가능 |
| Native-like casual | 5% | 실제 구어체, 농담, slang |

가장 중요한 구간은 **B1~B2**다.

---

## 9. 데이터 생성 계획

### Step 1. 시드 발화 600개 작성

사람이 먼저 user seed를 작성한다.

#### 시드 예시

```text
I feel tired today.
I don't know what to talk about.
I want to speak English better.
Yesterday I meet my friend.
I am boring.
Do you like music?
Can you give me a topic?
I want to travel but I have no money.
I think my English is bad.
What should I say when I meet new people?
```

### 시드 카테고리별 개수

| 카테고리 | 시드 개수 |
|---|---:|
| Small talk | 120 |
| 감정/상태 | 100 |
| 취향/의견 | 80 |
| 어색한 영어 | 140 |
| 짧은/모호한 발화 | 80 |
| 교정 요청 | 40 |
| 유머/가벼운 대화 | 20 |
| 안전/민감한 대화 | 20 |

---

### Step 2. LLM으로 후보 대화 생성

각 시드에서 5~8개의 변형 대화를 만든다.

```text
600 seeds × 6 variations = 3,600 candidate conversations
```

생성 후 검수하여 최종 3,000개 정도를 남긴다.

---

### Step 3. 사람 검수

검수자는 아래 기준으로 데이터를 삭제하거나 수정한다.

```text
삭제할 데이터:

- assistant가 너무 문법 선생님처럼 말함
- 답변이 너무 길고 설명적임
- 사용자가 말하지 않은 상황을 추정함
- assistant가 매번 질문으로 끝냄
- 영어가 너무 교과서적임
- 한국어 번역투 영어가 많음
- 대화가 실제 사람 대화처럼 들리지 않음
```

---

### Step 4. DPO 데이터 생성

SFT 데이터 또는 평가 중 나온 실패 케이스를 기반으로 rejected 답변을 만든다.

```text
좋은 답변 chosen
나쁜 답변 rejected
```

DPO 데이터는 모델에게 다음 차이를 학습시킨다.

```text
- 자연스러운 답변 > 교과서식 답변
- 대화 흐름 유지 > 과도한 문법 교정
- 맥락 반영 > 일반론
- 짧고 자연스러운 응답 > 장황한 설명
- 부드러운 확인 > 없는 상황 추정
```

---

## 10. 데이터 생성 프롬프트 템플릿

LLM으로 후보 데이터를 생성할 때 사용할 수 있는 프롬프트다.

```text
You are creating training data for an English conversation chatbot.

Goal:
Create natural English conversations between a user and an assistant.

Assistant behavior:
- The assistant is a friendly English conversation partner.
- The assistant should prioritize conversation flow over grammar correction.
- The assistant should not sound like a textbook or a teacher unless the user asks for correction.
- If the user's English is awkward but understandable, respond naturally.
- If a brief correction is helpful, include it gently in one sentence.
- Do not invent background information.
- Keep assistant replies usually 1–4 sentences.
- Do not end every response with a question.

User profile:
- The user is an English learner.
- The user's English level is: B1.
- The user may make grammar mistakes or use unnatural expressions.

Conversation type:
Awkward English input

First user message:
"I am boring today."

Create 5 conversations.
Each conversation should have 4–8 turns.
Return the result in JSONL messages format.
```

---

## 11. SFT 데이터 포맷

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a friendly English conversation partner. Keep the conversation natural and prioritize flow over correction."
    },
    {
      "role": "user",
      "content": "I am boring today."
    },
    {
      "role": "assistant",
      "content": "Sounds like you mean you’re bored today. Want me to throw out a random topic, or are you more in the mood to talk about your day?"
    },
    {
      "role": "user",
      "content": "Random topic please."
    },
    {
      "role": "assistant",
      "content": "Okay—if you could instantly become good at one hobby, what would you choose? No practice, no struggle, just suddenly talented."
    }
  ]
}
```

---

## 12. DPO 데이터 포맷

```json
{
  "prompt": [
    {
      "role": "system",
      "content": "You are a friendly English conversation partner. Prioritize natural conversation over correction."
    },
    {
      "role": "user",
      "content": "Yesterday I go to cafe."
    }
  ],
  "chosen": [
    {
      "role": "assistant",
      "content": "Nice. You’d usually say, “Yesterday I went to a café.” Was it a quick coffee stop, or did you stay there for a while?"
    }
  ],
  "rejected": [
    {
      "role": "assistant",
      "content": "Your sentence is incorrect. The past tense of go is went. Also, you need an article before cafe, so the correct sentence is: Yesterday I went to a cafe."
    }
  ]
}
```

---

## 13. 평가셋 구성

평가셋은 학습 데이터와 절대 섞지 않는다.

초기 평가셋은 약 700개로 구성한다.

| 평가 항목 | 개수 |
|---|---:|
| Small talk | 100 |
| 감정 대화 | 100 |
| 어색한 영어 입력 | 150 |
| 짧은/모호한 발화 | 100 |
| 교정 요청 | 100 |
| 긴 대화 | 50 |
| 안전/민감한 요청 | 100 |

---

## 14. 평가 기준

각 답변을 1~5점으로 평가한다.

| 평가 항목 | 질문 |
|---|---|
| 자연스러움 | 실제 영어 대화처럼 들리는가? |
| 대화 지속성 | 다음 말을 하고 싶게 만드는가? |
| 교정 균형 | 너무 많이 고치지 않는가? |
| 영어 수준 적응 | 사용자 레벨에 맞는가? |
| 맥락 반영 | 이전 대화를 잘 반영하는가? |
| 과잉추정 방지 | 없는 상황을 만들지 않았는가? |
| 톤 | 친근하지만 과하지 않은가? |
| 길이 | 답변이 너무 길거나 짧지 않은가? |

---

## 15. 4주 구축 일정

### Week 1 — 설계

#### 작업

```text
- 챗봇 페르소나 정의
- 응답 원칙 작성
- 데이터 카테고리 확정
- 시드 user 발화 600개 작성
- 평가 기준표 작성
```

#### 산출물

```text
persona.md
style_guide.md
category_schema.md
seed_prompts.csv
eval_rubric.md
```

---

### Week 2 — SFT 후보 데이터 생성

#### 작업

```text
- LLM으로 3,600~4,000개 후보 대화 생성
- 카테고리별 균형 확인
- 너무 비슷한 데이터 제거
- 1차 자동 필터링
```

#### 산출물

```text
sft_candidates.jsonl
data_stats.csv
duplicate_report.csv
```

---

### Week 3 — 검수 및 DPO 생성

#### 작업

```text
- SFT 후보에서 3,000개 선별
- rejected 답변 생성
- chosen/rejected 품질 검수
- DPO 2,000쌍 구성
```

#### 산출물

```text
sft_train.jsonl
sft_valid.jsonl
sft_test.jsonl
dpo_train.jsonl
dpo_valid.jsonl
```

---

### Week 4 — 평가셋 구축 및 1차 튜닝

#### 작업

```text
- 평가셋 700개 작성
- SFT 1차 학습
- 고정 평가셋으로 테스트
- 실패 케이스 분류
- DPO 적용 여부 판단
```

#### 산출물

```text
fixed_eval.jsonl
safety_eval.jsonl
failure_analysis.md
v1_model_report.md
```

---

## 16. 권장 폴더 구조

```text
english_conversation_bot/
  docs/
    persona.md
    style_guide.md
    labeling_guide.md
    eval_rubric.md

  seeds/
    user_seeds.csv
    topic_bank.csv
    error_patterns.csv

  data/
    sft/
      sft_train.jsonl
      sft_valid.jsonl
      sft_test.jsonl

    dpo/
      dpo_train.jsonl
      dpo_valid.jsonl

    eval/
      fixed_eval.jsonl
      ambiguity_eval.jsonl
      correction_eval.jsonl
      safety_eval.jsonl

  reports/
    data_stats.csv
    failure_analysis.md
    model_comparison.md
```

---

## 17. 성공 기준

이 프로젝트의 성공 기준은 다음 한 문장으로 정리할 수 있다.

> 사용자가 영어로 계속 말하고 싶어지는가?

### 성공적인 응답 예시

```text
user: I am boring today.

assistant:
Sounds like you mean you’re bored today. Want me to throw out a random topic, or are you more in the mood to talk about your day?
```

### 실패한 응답 예시

```text
user: I am boring today.

assistant:
The correct expression is “I am bored today.” “Boring” is an adjective used to describe something that causes boredom.
```

문법적으로는 맞지만, 자연스러운 대화 파트너로는 실패다.

---

## 18. 최종 요약

```text
Project name:
Natural English Conversation Bot Fine-tuning Project

Primary goal:
Build a chatbot that helps users continue natural English conversations.

Initial dataset size:
- SFT: 3,000 multi-turn conversations
- DPO: 2,000 preference pairs
- Evaluation set: 700 cases

Core data categories:
- Awkward English input: 20%
- Daily small talk: 20%
- Emotion/status sharing: 13%
- Short or ambiguous utterances: 12%
- Long conversation maintenance: 10%

Data generation method:
- Create 600 human-written user seeds
- Generate 3,600 candidate conversations with LLM
- Human-review and select 3,000 SFT conversations
- Create 2,000 DPO pairs using realistic failure patterns
- Build a separate fixed evaluation set

Core principles:
- Conversation flow comes first.
- Grammar correction is secondary.
- English should sound natural and conversational.
- The assistant should adapt to the user’s level.
- The assistant should not invent missing context.
```