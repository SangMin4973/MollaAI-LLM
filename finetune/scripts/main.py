import argparse
import json
import sys
import traceback
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dataset.core.categories import list_category_keys
from dataset.core.seeds import SEED_EXAMPLES, get_seed_by_id, list_seed_examples


class StdoutArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)

    def exit(self, status=0, message=None):
        if message:
            raise ValueError(message)
        raise SystemExit(status)


def emit_jsonl(records: list[dict[str, object]]) -> None:
    for record in records:
        print(json.dumps(record, ensure_ascii=False), flush=True)


class JsonArrayWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")
        self.handle.write("[\n")
        self.first_record = True

    def write_record(self, record: dict[str, object]) -> None:
        if not self.first_record:
            self.handle.write(",\n")
        self.handle.write(json.dumps(record, ensure_ascii=False, indent=2))
        self.handle.flush()
        self.first_record = False

    def close(self) -> None:
        self.handle.write("\n]\n")
        self.handle.close()


def resolve_user_level(cli_value: str | None, seed_level: str) -> str:
    return cli_value if cli_value is not None else seed_level


def resolve_variation_count(cli_value: int | None, seed_variation_count: int) -> int:
    return cli_value if cli_value is not None else seed_variation_count


def main() -> None:
    parser = StdoutArgumentParser(description="Dataset LLM generator")
    parser.add_argument("--mode", choices=["sft", "dpo"], help="Generation mode")
    # 문자열 대신 시드 번호를 받아서 미리 정의된 예문을 선택합니다.
    parser.add_argument("--seed-number", type=int, help="Seed example number")
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=0,
        help="Repeat generation this many times across the full seed set",
    )
    parser.add_argument("--conversation-type", default=None)
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--list-seeds", action="store_true")
    parser.add_argument("--user-level", default=None)
    parser.add_argument("--variation-count", type=int, default=None)
    parser.add_argument("--min-turns", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--cache-dir", default="./")
    parser.add_argument("--output-file", default=None, help="Write output records to this file as a JSON array")
    args = parser.parse_args()

    if args.list_categories:
        print(json.dumps(list_category_keys(), ensure_ascii=False, indent=2))
        return

    if args.list_seeds:
        print(json.dumps(list_seed_examples(), ensure_ascii=False, indent=2))
        return

    if not args.mode:
        raise ValueError("--mode is required unless --list-categories or --list-seeds is used")
    if args.repeat_count < 0:
        raise ValueError("--repeat-count must be 0 or greater")
    if args.repeat_count > 0 and args.seed_number is not None:
        raise ValueError("--seed-number cannot be combined with --repeat-count")
    if args.repeat_count == 0 and args.seed_number is None:
        raise ValueError(
            "--seed-number is required unless --repeat-count, --list-categories, or --list-seeds is used"
        )

    from dataset.generation.llm import DatasetQwenGenerator

    generator = DatasetQwenGenerator(cache_dir=args.cache_dir)

    output_writer = None
    if args.output_file:
        output_path = Path(args.output_file)
        output_writer = JsonArrayWriter(output_path)

    try:
        if args.repeat_count > 0:
            if args.mode == "sft":
                for _ in range(args.repeat_count):
                    for selected_seed in SEED_EXAMPLES:
                        conversation_type = args.conversation_type or selected_seed.category_key
                        for record in generator.generate_sft_candidates(
                            seed_text=selected_seed.text,
                            conversation_type=conversation_type,
                            user_level=resolve_user_level(args.user_level, selected_seed.level),
                            variation_count=resolve_variation_count(
                                args.variation_count, selected_seed.variation_count
                            ),
                            min_turns=args.min_turns,
                            max_turns=args.max_turns,
                            target_behavior=selected_seed.target_behavior,
                            assistant_tone=selected_seed.tone,
                            should_correct=selected_seed.should_correct,
                            notes=selected_seed.notes,
                            verbose=True,
                        ):
                            if output_writer is not None:
                                output_writer.write_record(record)
                            else:
                                print(json.dumps(record, ensure_ascii=False), flush=True)
                return

            for _ in range(args.repeat_count):
                for selected_seed in SEED_EXAMPLES:
                    record = generator.generate_dpo_pair(
                        user_message=selected_seed.text,
                        user_level=resolve_user_level(args.user_level, selected_seed.level),
                        target_behavior=selected_seed.target_behavior,
                        assistant_tone=selected_seed.tone,
                        should_correct=selected_seed.should_correct,
                        notes=selected_seed.notes,
                    )
                    if output_writer is not None:
                        output_writer.write_record(record)
                    else:
                        print(json.dumps(record, ensure_ascii=False), flush=True)
            return

        selected_seed = get_seed_by_id(args.seed_number)

        if args.mode == "sft":
            # conversation-type을 별도로 주면 강제하고, 없으면 시드의 카테고리를 기본값으로 씁니다.
            conversation_type = args.conversation_type or selected_seed.category_key
            for record in generator.generate_sft_candidates(
                seed_text=selected_seed.text,
                conversation_type=conversation_type,
                user_level=resolve_user_level(args.user_level, selected_seed.level),
                variation_count=resolve_variation_count(
                    args.variation_count, selected_seed.variation_count
                ),
                min_turns=args.min_turns,
                max_turns=args.max_turns,
                target_behavior=selected_seed.target_behavior,
                assistant_tone=selected_seed.tone,
                should_correct=selected_seed.should_correct,
                notes=selected_seed.notes,
                verbose=True,
            ):
                if output_writer is not None:
                    output_writer.write_record(record)
                else:
                    print(json.dumps(record, ensure_ascii=False), flush=True)
            return

        record = generator.generate_dpo_pair(
            user_message=selected_seed.text,
            user_level=resolve_user_level(args.user_level, selected_seed.level),
            target_behavior=selected_seed.target_behavior,
            assistant_tone=selected_seed.tone,
            should_correct=selected_seed.should_correct,
            notes=selected_seed.notes,
        )
        if output_writer is not None:
            output_writer.write_record(record)
        else:
            print(json.dumps(record, ensure_ascii=False, indent=2))
    finally:
        if output_writer is not None:
            output_writer.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Error in main")
        print(traceback.format_exc())
        sys.exit(1)
