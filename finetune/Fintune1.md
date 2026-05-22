# Fine-tune Strategy 1

## 목표

- 베이스 모델: `Qwen/Qwen3-8B`
- 학습 방식: LLaMA-Factory 기반 SFT
- 데이터셋 원본: `dataset/data/raw/sft_all.json`
- 학습용 데이터셋: `dataset/data/processed/sft_all_sharegpt.json`
- GPU 환경: RTX 3060 12GB

## 현재 데이터 특성

- 총 샘플 수: `1014`
- 대화 턴 수 평균: `4.97`
- 응답 길이 평균: `114.29`

## 데이터 포맷

- `dataset/data/raw/sft_all.json`: 내부 생성용 `messages` 포맷
- `dataset/data/processed/sft_all_sharegpt.json`: LLaMA-Factory `sharegpt` 포맷
- `dataset/data/processed/sft_all_conversations_lf.json`: 레거시 `human/gpt` 변환본

`dataset/data/dataset_info.json`는 아래처럼 등록한다.

```json
{
  "sft_all_conversations": {
    "file_name": "processed/sft_all_sharegpt.json",
    "formatting": "sharegpt"
  }
}
```

## 현재 학습 설정

기준 파일:

- `LLaMA-Factory/examples/train_lora/my_qwen3_lora_sft_sft_all_sharegpt.yaml`

핵심 설정:

```yaml
model_name_or_path: Qwen/Qwen3-8B
quantization_bit: 4
quantization_method: bnb
finetuning_type: lora
lora_rank: 8
lora_target: all
dataset_dir: /home/dltkd/molla/MollaAI-AI/demo/dataset/data
dataset: sft_all_conversations
template: qwen3_nothink
cutoff_len: 768
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
num_train_epochs: 5.0
val_size: 0.05
eval_strategy: steps
eval_steps: 100
bf16: true
```

## 병합과 양자화

- LoRA 학습 후 병합 단계에서는 quantized model을 사용하지 않는다.
- `LLaMA-Factory/examples/merge_lora/my_qwen3_lora_sft_merge.yaml`로 병합한다.
- `LLaMA-Factory/examples/merge_lora/my_qwen3_gptq_export.yaml`로 4-bit GPTQ export를 수행한다.
- 캘리브레이션 코퍼스는 `dataset/data/calibration/my_quantize_corpus.jsonl`을 사용한다.

