# Dataset Fine-tuning Notes

이 폴더는 영어 회화용 SFT 데이터를 만들고, LLaMA-Factory로 학습하고, 병합/양자화까지 이어가는 전체 흐름을 담고 있다.

## 현재 폴더 구조

```text
dataset/
  core/         # seed, category, config, model config
  generation/   # prompt builder, JSON parser, local generator
  scripts/      # CLI entry points
  data/
    raw/        # 원본 생성 결과
    processed/  # 학습용 변환본
    calibration/# GPTQ/awq 캘리브레이션 코퍼스
  *.md          # 설계 문서와 현재 README
```

호환성을 위해 기존 진입점은 유지한다.

- `dataset/main.py` -> `dataset/scripts/main.py`
- `dataset/test.py` -> `dataset/scripts/test.py`

## 학습한 데이터 구조

실제 학습은 LLaMA-Factory `sharegpt` 포맷으로 진행했다.

`dataset/data/dataset_info.json` 기준으로 등록된 데이터셋은 아래와 같다.

- dataset key: `sft_all_conversations`
- source file: `data/processed/sft_all_sharegpt.json`
- formatting: `sharegpt`

핵심 샘플 수는 `1014`개다.

### `data/processed/sft_all_sharegpt.json`

학습에 사용한 메인 데이터다.

예시 구조:

```json
{
  "system": "You are a friendly spoken English conversation partner ...",
  "conversations": [
    { "from": "user", "value": "I am boring today." },
    { "from": "assistant", "value": "You mean you're bored today? Let's do something fun!" }
  ]
}
```

### `data/raw/sft_all.json`

내부 생성 파이프라인에서 사용한 원본 구조다.

```json
{
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

이 구조는 생성/검증 단계에서 다루기 쉽다. 이후 LLaMA-Factory용으로 `sharegpt` 포맷으로 변환했다.

### `data/processed/sft_all_conversations_lf.json`

LLaMA-Factory 호환용 변환본이다.

```json
{
  "conversations": [
    { "from": "human", "value": "..." },
    { "from": "gpt", "value": "..." }
  ]
}
```

이 파일은 legacy 스타일 변환을 위해 남겨둔 중간 산출물이다.

### `data/calibration/my_quantize_corpus.jsonl`

GPTQ 캘리브레이션용 텍스트 코퍼스다.

한 줄에 한 샘플이 들어가며 형식은 아래와 같다.

```json
{"text": "user line\nassistant line\nuser line\nassistant line"}
```

총 `1014`줄이며, 학습 대화 내용을 평문으로 펼쳐서 만들었다.

## 학습 설정 요약

기준 파일:

- `LLaMA-Factory/examples/train_lora/my_qwen3_lora_sft_sft_all_sharegpt.yaml`

핵심 설정:

- base model: `Qwen/Qwen3-8B`
- quantization: `bnb 4-bit`
- finetuning: `LoRA`
- lora rank: `8`
- lora target: `all`
- cutoff length: `768`
- batch size: `1`
- gradient accumulation: `8`
- learning rate: `5e-5`
- epochs: `5`
- warmup ratio: `0.1`
- scheduler: `cosine`
- eval split: `0.05`
- eval interval: `100`
- save interval: `100`
- bf16: `true`

## 병합과 최종 양자화

### 1. LoRA 학습

`Qwen/Qwen3-8B`를 4-bit QLoRA로 로드한 뒤 `sft_all_conversations` 데이터셋으로 SFT를 수행했다.

### 2. LoRA 병합

`LLaMA-Factory/examples/merge_lora/my_qwen3_lora_sft_merge.yaml`로 어댑터를 베이스 모델에 병합했다.

중요한 점:

- 병합 단계에서는 quantized model이나 `quantization_bit`를 사용하지 않는다.
- 즉, 병합 입력은 이미 학습된 adapter와 base model checkpoint다.

### 3. GPTQ export

`LLaMA-Factory/examples/merge_lora/my_qwen3_gptq_export.yaml`로 병합된 모델을 4-bit GPTQ로 다시 내보냈다.

캘리브레이션 설정:

- quantization bit: `4`
- calibration dataset: `dataset/data/calibration/my_quantize_corpus.jsonl`
- calibration samples: `32`
- max length: `64`

## 참고 문서

- `DATASET_PLAN.md`: 원래 데이터 설계 문서
- `SEED.md`: seed 설계 기준
- `Fintune1.md`: 기존 학습 전략 메모
