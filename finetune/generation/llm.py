import copy
import json
import sys
import warnings

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

from dataset.core.config import (
    DPOGenerationOptions,
    MODEL_NAME,
    SFTGenerationOptions,
    build_conversation_system_prompt,
)
from dataset.generation.json_utils import answer_extraction, parse_json_records, parse_sft_continuation_turns
from dataset.core.model_config import (
    dataset_gen_config,
    dataset_pair_gen_config,
    quantization_config,
)
from dataset.generation.prompt_templates import build_dpo_prompt, build_sft_continuation_prompt, build_sft_prompt

warnings.filterwarnings("ignore")


class DatasetQwenGenerator:
    def __init__(self, model_name: str = MODEL_NAME, cache_dir: str = "./"):
        print("Dataset LLM model loading...", file=sys.stderr)
        # Some checkpoints serialize `extra_special_tokens` as a list, which newer
        # transformers versions reject. Override it with a dict to keep loading stable.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, extra_special_tokens={})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
        )
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        print("Dataset LLM ready", file=sys.stderr)

    def generate_from_messages(
        self,
        messages: list[dict[str, str]],
        generation_config,
    ) -> str:
        _, generated_text = self._generate_raw_text_from_messages(messages, generation_config)
        return answer_extraction(generated_text).strip()

    def _generate_raw_text_from_messages(
        self,
        messages: list[dict[str, str]],
        generation_config,
    ) -> tuple[str, str]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw = self.pipe(prompt, generation_config=generation_config)
        generated_text = raw[0]["generated_text"]
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].lstrip()
        return prompt, generated_text

    @staticmethod
    def _build_sft_continuation_generation_config():
        continuation_config = copy.deepcopy(dataset_gen_config)
        continuation_config.max_new_tokens = 180
        return continuation_config

    def build_sft_prompt(
        self,
        seed_text: str,
        conversation_type: str,
        user_level: str = "B",
        variation_count: int = 5,
        min_turns: int = 4,
        max_turns: int = 8,
        target_behavior: str = "natural_conversation",
        assistant_tone: str = "casual",
        should_correct: bool | str = "optional",
        notes: str = "",
    ) -> list[dict[str, str]]:
        return build_sft_prompt(
            SFTGenerationOptions(
                seed_text=seed_text,
                conversation_type=conversation_type,
                user_level=user_level,
                variation_count=variation_count,
                min_turns=min_turns,
                max_turns=max_turns,
                target_behavior=target_behavior,
                assistant_tone=assistant_tone,
                should_correct=should_correct,
                notes=notes,
            )
        )

    @staticmethod
    def _select_target_turn_count(variation_index: int, min_turns: int, max_turns: int) -> int:
        normalized_min = max(4, min_turns)
        normalized_max = max(normalized_min, max_turns)
        even_turns = [turn_count for turn_count in range(normalized_min, normalized_max + 1) if turn_count % 2 == 0]
        if not even_turns:
            fallback_turn_count = normalized_max if normalized_max % 2 == 0 else normalized_max + 1
            even_turns = [fallback_turn_count]
        return even_turns[variation_index % len(even_turns)]

    @staticmethod
    def _messages_to_transcript_lines(messages: list[dict[str, str]]) -> list[str]:
        transcript_lines: list[str] = []
        for message in messages:
            if message["role"] == "user":
                transcript_lines.append(f"User: {message['content']}")
            elif message["role"] == "assistant":
                transcript_lines.append(f"Assistant: {message['content']}")
        return transcript_lines

    def _build_sft_continuation_messages(
        self,
        seed_text: str,
        conversation_type: str,
        user_level: str,
        min_turns: int,
        max_turns: int,
        target_behavior: str,
        assistant_tone: str,
        should_correct: bool | str,
        notes: str,
        transcript_lines: list[str],
        turns_to_generate: int,
    ) -> list[dict[str, str]]:
        return build_sft_continuation_prompt(
            SFTGenerationOptions(
                seed_text=seed_text,
                conversation_type=conversation_type,
                user_level=user_level,
                variation_count=1,
                min_turns=min_turns,
                max_turns=max_turns,
                target_behavior=target_behavior,
                assistant_tone=assistant_tone,
                should_correct=should_correct,
                notes=notes,
            ),
            transcript_lines=transcript_lines,
            turns_to_generate=turns_to_generate,
        )

    def generate_sft_candidates(
        self,
        seed_text: str,
        conversation_type: str,
        user_level: str = "B",
        variation_count: int = 5,
        min_turns: int = 4,
        max_turns: int = 8,
        target_behavior: str = "natural_conversation",
        assistant_tone: str = "casual",
        should_correct: bool | str = "optional",
        notes: str = "",
        verbose: bool = False,
    ):
        for variation_index in range(variation_count):
            target_turn_count = self._select_target_turn_count(
                variation_index=variation_index,
                min_turns=min_turns,
                max_turns=max_turns,
            )
            conversation_messages: list[dict[str, str]] = [
                {"role": "system", "content": build_conversation_system_prompt(user_level)},
                {"role": "user", "content": seed_text.strip()},
            ]

            while len(conversation_messages) < target_turn_count:
                remaining_turns = target_turn_count - len(conversation_messages)
                turns_to_generate = 2 if remaining_turns >= 2 else 1
                continuation_prompt_messages = self._build_sft_continuation_messages(
                    seed_text=seed_text,
                    conversation_type=conversation_type,
                    user_level=user_level,
                    min_turns=min_turns,
                    max_turns=max_turns,
                    target_behavior=target_behavior,
                    assistant_tone=assistant_tone,
                    should_correct=should_correct,
                    notes=notes,
                    transcript_lines=self._messages_to_transcript_lines(conversation_messages),
                    turns_to_generate=turns_to_generate,
                )

                expected_roles = ["assistant", "user"] if turns_to_generate == 2 else ["assistant"]
                last_error: Exception | None = None
                continuation_messages: list[dict[str, str]] | None = None

                for attempt in range(3):
                    _, generated_text = self._generate_raw_text_from_messages(
                        continuation_prompt_messages,
                        generation_config=self._build_sft_continuation_generation_config(),
                    )
                    extracted_text = answer_extraction(generated_text).strip()
                    try:
                        continuation_messages = parse_sft_continuation_turns(
                            extracted_text,
                            expected_roles=expected_roles,
                        )
                        if verbose:
                            print(f"=== Variation {variation_index + 1} Step ===", file=sys.stderr)
                            print("=== Generated Output ===", file=sys.stderr)
                            print(extracted_text, file=sys.stderr)
                            print(file=sys.stderr)
                            print("=== Parsed Output ===", file=sys.stderr)
                            print(json.dumps(continuation_messages, ensure_ascii=False, indent=2), file=sys.stderr)
                            print(file=sys.stderr)
                        break
                    except ValueError as exc:
                        last_error = exc
                        print(
                            "SFT continuation parse failed on attempt "
                            f"{attempt + 1}/3 for seed_text={seed_text!r}, variation={variation_index + 1}",
                            file=sys.stderr,
                        )
                        print(extracted_text[:1000], file=sys.stderr)

                if continuation_messages is None:
                    raise ValueError(f"Failed to parse SFT continuation after 3 attempts: {last_error}")

                conversation_messages.extend(continuation_messages)

            yield {"messages": conversation_messages}

    def build_dpo_prompt(
        self,
        user_message: str,
        system_prompt: str | None = None,
        context_messages: list[dict[str, str]] | None = None,
        user_level: str = "B",
        target_behavior: str = "natural_conversation",
        assistant_tone: str = "casual",
        should_correct: bool | str = "optional",
        notes: str = "",
    ) -> list[dict[str, str]]:
        return build_dpo_prompt(
            DPOGenerationOptions(
                user_message=user_message,
                system_prompt=system_prompt,
                context_messages=context_messages,
                user_level=user_level,
                target_behavior=target_behavior,
                assistant_tone=assistant_tone,
                should_correct=should_correct,
                notes=notes,
            )
        )

    def generate_dpo_pair(
        self,
        user_message: str,
        system_prompt: str | None = None,
        context_messages: list[dict[str, str]] | None = None,
        user_level: str = "B",
        target_behavior: str = "natural_conversation",
        assistant_tone: str = "casual",
        should_correct: bool | str = "optional",
        notes: str = "",
    ) -> dict[str, object]:
        messages = self.build_dpo_prompt(
            user_message=user_message,
            system_prompt=system_prompt,
            context_messages=context_messages,
            user_level=user_level,
            target_behavior=target_behavior,
            assistant_tone=assistant_tone,
            should_correct=should_correct,
            notes=notes,
        )
        last_error: Exception | None = None
        for attempt in range(3):
            raw = self.generate_from_messages(messages, generation_config=dataset_pair_gen_config)
            try:
                records = parse_json_records(raw)
                if len(records) != 1:
                    raise ValueError("DPO generation must return exactly one JSON object")
                return records[0]
            except ValueError as exc:
                last_error = exc
                print(f"DPO JSON parse failed on attempt {attempt + 1}/3 for user_message={user_message!r}")
                print(raw[:1000])
        raise ValueError(f"Failed to parse DPO JSON after 3 attempts: {last_error}")

    @staticmethod
    def to_jsonl(records: list[dict[str, object]]) -> str:
        return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
