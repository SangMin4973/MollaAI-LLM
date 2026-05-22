import os
import logging
import traceback
import warnings
import time
from threading import Thread
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger("molla.llm")

TOK_MODEL_NAME =  "Qwen/Qwen3-8B"
MODEL_NAME = "ssaann/eng_conversation_sft"

gen_config = GenerationConfig(
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
)

class QwenChat:
    def __init__(self, model_name=MODEL_NAME, tok_model=TOK_MODEL_NAME, cache_dir=None):
        print("🔧 LLM 모델 로딩 중...")

        cache_dir = cache_dir or os.getenv("HF_HOME")

        self.tokenizer = AutoTokenizer.from_pretrained(tok_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
        )

        print("✅ LLM 준비 완료")

    def get_prompt(self, query: str):
        full_prompt = f"""
You are an English conversation teacher speaking with a student on a live phone call.

Follow these rules strictly:
- Speak only as the teacher.
- Never write dialogue for the student.
- Never include role labels such as "Teacher:", "Student:", "User:", or "Assistant:".
- Do not create a sample conversation or script.
- Respond with only one short natural reply for the student's latest utterance.
- Use at most 2 short sentences.
- Ask at most 1 short follow-up question only if it helps continue the conversation.
- Do not answer your own question.
- Do not use markdown, bullet points, or stage directions.
- Output plain spoken English only.

<question>
{query}
<answer>
"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an English conversation teacher on a live phone call. "
                    "Reply briefly in plain spoken English, only as the teacher, "
                    "and never generate the student's side of the conversation."
                ),
            },
            {"role": "user", "content": full_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        return full_prompt, prompt

    def _tokenize_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = getattr(self.model, "device", None)
        if model_device is None:
            return inputs
        return {key: value.to(model_device) for key, value in inputs.items()}

    def answer_extraction(self, response: str) -> str:
        try:
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant", 1)[1]

            if "</think>" in response:
                think_end = response.find("</think>") + len("</think>")
                response = response[think_end:]

            return response.strip()

        except Exception:
            print("Error in answer_extraction")
            print(traceback.format_exc())
            return response

    def stream_answer(self, query: str, request_id: str = "-"):
        try:
            started_at = time.perf_counter()
            logger.info(
                "llm_stream_prepare request_id=%s query_len=%s",
                request_id,
                len(query),
            )
            _, prompt = self.get_prompt(query)
            logger.info(
                "llm_prompt_ready request_id=%s elapsed_ms=%s prompt_len=%s",
                request_id,
                int((time.perf_counter() - started_at) * 1000),
                len(prompt),
            )
            inputs = self._tokenize_prompt(prompt)
            input_tokens = 0
            input_ids = inputs.get("input_ids")
            if input_ids is not None and getattr(input_ids, "shape", None) is not None:
                input_tokens = int(input_ids.shape[-1])
            logger.info(
                "llm_inputs_ready request_id=%s elapsed_ms=%s input_tokens=%s",
                request_id,
                int((time.perf_counter() - started_at) * 1000),
                input_tokens,
            )
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = {
                **inputs,
                "generation_config": gen_config,
                "streamer": streamer,
            }

            worker = Thread(
                target=self.model.generate,
                kwargs=generation_kwargs,
                daemon=True,
            )
            worker.start()
            logger.info(
                "llm_generate_started request_id=%s elapsed_ms=%s",
                request_id,
                int((time.perf_counter() - started_at) * 1000),
            )

            chunk_count = 0
            for chunk in streamer:
                if chunk:
                    chunk_count += 1
                    if chunk_count == 1:
                        logger.info(
                            "llm_first_chunk request_id=%s elapsed_ms=%s chunk=%r",
                            request_id,
                            int((time.perf_counter() - started_at) * 1000),
                            chunk[:80],
                        )
                    yield chunk

            worker.join()
            logger.info(
                "llm_stream_done request_id=%s elapsed_ms=%s chunks=%s",
                request_id,
                int((time.perf_counter() - started_at) * 1000),
                chunk_count,
            )
        except Exception:
            logger.exception("llm_stream_failed request_id=%s", request_id)
            yield "오류가 발생했습니다."

    def ask(self, query: str) -> str:
        try:
            answer = self.answer_extraction("".join(self.stream_answer(query)))
            return answer

        except Exception:
            print("Error in ask()")
            print(traceback.format_exc())
            return "오류가 발생했습니다."
