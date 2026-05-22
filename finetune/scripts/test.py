import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dataset.core.config import build_conversation_system_prompt
from dataset.generation.json_utils import answer_extraction, parse_sft_continuation_turns
from dataset.core.seeds import get_seed_by_id


if TYPE_CHECKING:
    from dataset.generation.llm import DatasetQwenGenerator


def build_prompt(generator: "DatasetQwenGenerator", messages: list[dict[str, str]]) -> str:
    return generator.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one seed's full SFT generation cycle")
    parser.add_argument("--seed-number", type=int, default=1)
    parser.add_argument("--target-turns", type=int, default=6)
    parser.add_argument("--cache-dir", default="./")
    args = parser.parse_args()

    from dataset.generation.llm import DatasetQwenGenerator

    seed = get_seed_by_id(args.seed_number)
    generator = DatasetQwenGenerator(cache_dir=args.cache_dir)

    conversation_messages: list[dict[str, str]] = [
        {"role": "system", "content": build_conversation_system_prompt(seed.level)},
        {"role": "user", "content": seed.text},
    ]

    print("=== Seed ===")
    print(f"id={seed.id} level={seed.level} category={seed.category}")
    print(seed.text)
    print()

    step = 1
    while len(conversation_messages) < args.target_turns:
        remaining_turns = args.target_turns - len(conversation_messages)
        turns_to_generate = 2 if remaining_turns >= 2 else 1
        messages = generator._build_sft_continuation_messages(
            seed_text=seed.text,
            conversation_type=seed.category_key,
            user_level=seed.level,
            min_turns=4,
            max_turns=8,
            target_behavior=seed.target_behavior,
            assistant_tone=seed.tone,
            should_correct=seed.should_correct,
            notes=seed.notes,
            transcript_lines=generator._messages_to_transcript_lines(conversation_messages),
            turns_to_generate=turns_to_generate,
        )
        prompt = build_prompt(generator, messages)
        raw = generator.pipe(
            prompt,
            generation_config=generator._build_sft_continuation_generation_config(),
        )
        generated_text = raw[0]["generated_text"]
        extracted_text = answer_extraction(generated_text).strip()
        expected_roles = ["assistant", "user"] if turns_to_generate == 2 else ["assistant"]
        parsed = parse_sft_continuation_turns(extracted_text, expected_roles=expected_roles)
        current_transcript = "\n".join(generator._messages_to_transcript_lines(conversation_messages))
        conversation_messages.extend(parsed)

        print(f"=== Step {step} Current Transcript ===")
        print(current_transcript)
        print()
        print(f"=== Step {step} Raw Output ===")
        print(generated_text)
        print()
        print(f"=== Step {step} Extracted Output ===")
        print(extracted_text)
        print()
        print(f"=== Step {step} Parsed Output ===")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        print()

        step += 1

    print("=== Final Conversation Messages ===")
    print(json.dumps({"messages": conversation_messages}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
