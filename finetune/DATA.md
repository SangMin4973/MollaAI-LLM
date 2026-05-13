# DATA Generation Guide

이 문서는 `dataset/` 안의 JSON, JSONL, 계획 문서, 그리고 생성 코드를 기준으로 데이터가 어떻게 만들어지는지 정리한 문서다.

## 기준 문서와 코드

- `DATASET_PLAN.md`: 전체 대화 방향과 seed 설계 원칙
- `SEED.md`: seed 필드 정의와 작성 기준
- `core/categories.py`: 대화 카테고리 정의
- `core/seeds.py`: seed bank
- `core/config.py`: 시스템 프롬프트와 옵션 dataclass
- `generation/prompt_templates.py`: 생성용 프롬프트 템플릿
- `generation/json_utils.py`: 모델 출력 정리와 JSON 파싱
- `generation/llm.py`: 실제 생성기
- `scripts/main.py`: CLI 진입점

## 데이터 생성 목표

목표는 영어 회화용 SFT 데이터를 만드는 것이다.

핵심 방향은 다음과 같다.

- 영어를 교과서처럼 가르치지 않고 자연스럽게 대화한다.
- 사용자가 어색하게 말해도 의미를 먼저 이해한다.
- 문법 교정은 필요한 경우에만 짧고 자연스럽게 한다.
- 짧은 발화, hesitation, mixed language, conversation recovery를 적극적으로 포함한다.

## Seed 구조

각 seed는 단순 문장이 아니라 메타데이터를 함께 가진다.

```python
SeedExample(
    id=1,
    category="awkward_english",
    level="B",
    user_text="I am boring today.",
    target_behavior="brief_correction_then_continue",
    tone="casual",
    should_correct=True,
    variation_count=6,
    notes="..."
)
```

필드 의미:

- `id`: seed 식별자
- `category`: 대화 유형
- `level`: 사용자 영어 수준
- `user_text`: 첫 사용자 발화
- `target_behavior`: assistant의 핵심 행동
- `tone`: 응답 톤
- `should_correct`: 교정 여부
- `variation_count`: 기본 생성 변형 수
- `notes`: 세부 제약 설명

## 생성 파이프라인

### 1. CLI가 seed를 고른다

`scripts/main.py`는 아래 옵션을 지원한다.

- `--mode sft|dpo`
- `--seed-number`
- `--repeat-count`
- `--conversation-type`
- `--user-level`
- `--variation-count`
- `--min-turns`
- `--max-turns`
- `--output-file`

seed 목록과 카테고리 목록은 모델을 로드하지 않고 바로 확인할 수 있다.

```bash
python3 dataset/main.py --list-categories
python3 dataset/main.py --list-seeds
```

### 2. 프롬프트를 만든다

`generation/prompt_templates.py`가 seed 메타데이터를 반영한 prompt를 만든다.

SFT 생성은 두 단계로 돌아간다.

- 첫 응답용 전체 프롬프트
- 이후 대화 연장을 위한 continuation 프롬프트

즉, 한 번에 긴 대화를 대충 뽑는 방식이 아니라, 짧은 continuation을 반복해서 안정적으로 대화를 길게 만든다.

### 3. 모델이 생성한다

`generation/llm.py`의 `DatasetQwenGenerator`가 `Qwen/Qwen3-8B`를 4-bit로 로드한다.

생성 설정:

- `bnb 4-bit`
- `nf4`
- `bf16 compute`
- SFT 생성용 `GenerationConfig`
- DPO 생성용 `GenerationConfig`

### 4. 출력이 정리된다

모델 출력은 바로 쓰지 않고 정리한다.

- `answer_extraction()`: `<think>`나 코드펜스 같은 노이즈를 제거
- `parse_sft_continuation_turns()`: `User:` / `Assistant:` 형식을 파싱
- `parse_json_records()`: DPO JSON을 파싱

SFT continuation은 잘못된 speaker order나 잘못된 턴 수가 나오면 재시도한다.

### 5. 최종 데이터로 저장한다

출력은 상황에 따라 다음 형식으로 저장된다.

- `messages` list JSON
- `sharegpt` JSON
- JSONL text corpus

## 현재 데이터 파일

### `data/raw/sft_all.json`

내부 표준 형식이다.

형식:

```json
{
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

통계:

- 샘플 수: `1014`
- 턴 수: `4 ~ 8`
- 평균 턴 수: `5.97`

### `data/processed/sft_all_sharegpt.json`

LLaMA-Factory 학습용 `sharegpt` 포맷이다.

형식:

```json
{
  "system": "...",
  "conversations": [
    { "from": "user", "value": "..." },
    { "from": "assistant", "value": "..." }
  ]
}
```

통계:

- 샘플 수: `1014`
- 턴 수: `3 ~ 7`
- 평균 턴 수: `4.97`

### `data/processed/sft_all_conversations_lf.json`

`human/gpt` 기반의 LLaMA-Factory 호환 변환본이다.

형식:

```json
{
  "conversations": [
    { "from": "human", "value": "..." },
    { "from": "gpt", "value": "..." }
  ]
}
```

통계:

- 샘플 수: `1014`
- 턴 수: `2 ~ 6`
- 평균 턴 수: `3.97`

### `data/calibration/my_quantize_corpus.jsonl`

GPTQ 캘리브레이션용 텍스트 코퍼스다.

형식:

```json
{"text": "user line\nassistant line\n..."}
```

통계:

- 라인 수: `1014`
- 각 라인은 하나의 대화 샘플을 평문으로 펼친 것

## category 설계

대화 카테고리는 `core/categories.py`에 정의되어 있다.

예:

- `small_talk`
- `awkward_english`
- `spoken_short_reply`
- `emotion_checkin`
- `conversation_recovery`
- `correction_request`
- `confidence_issue`
- `mixed_language`
- `natural_expression`
- `roleplay_conversation`

카테고리는 seed의 첫 발화와 목표 반응을 좁혀서, 생성 결과가 너무 넓게 퍼지지 않도록 도와준다.

## 생성 예시

SFT 후보 생성:

```bash
python3 dataset/main.py --mode sft --seed-number 7
```

DPO 샘플 생성:

```bash
python3 dataset/main.py --mode dpo --seed-number 8
```

전체 seed 반복 생성:

```bash
python3 dataset/main.py --mode sft --repeat-count 1 --output-file sft_all.json
python3 dataset/main.py --mode dpo --repeat-count 1 --output-file dpo_all.json
```

## 생성 규칙 요약

- assistant는 자연스럽고 짧은 spoken English를 사용한다.
- 사용자의 영어가 이해 가능하면 대화 흐름을 우선한다.
- 필요한 경우에만 부드럽게 교정한다.
- 배경 정보를 함부로 지어내지 않는다.
- SFT는 보통 4~8턴의 대화로 만든다.
- DPO는 `prompt / chosen / rejected` 3개 필드의 JSON 객체 1개를 만든다.
