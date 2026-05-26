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

MODEL_NAME = "ssaann/eng_conversation_sft"
RUNTIME_SYSTEM_PROMPT = (
    "You are a friendly spoken English conversation partner. "
    "Prioritize natural conversation, short speakable replies, and gentle correction only when useful. "
    "Use clear everyday English, natural spoken phrasing, and short follow-up questions. "
    "Keep corrections brief and natural."
)

gen_config = GenerationConfig(
    max_new_tokens=200,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    remove_invalid_values=True,
    renormalize_logits=True,
)

class QwenChat:
    def __init__(self, model_name=MODEL_NAME, tok_model=None, cache_dir=None):
        print("🔧 LLM 모델 로딩 중...")

        cache_dir = cache_dir or os.getenv("HF_HOME")
        tok_model = tok_model or model_name

        self.tokenizer = AutoTokenizer.from_pretrained(tok_model, extra_special_tokens={})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_dir,
        )

        print("✅ LLM 준비 완료")

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise
            logger.warning("llm_chat_template_retry_without_enable_thinking")
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def get_prompt(self, query: str):
        full_prompt = query.strip()

        messages = [
            {
                "role": "system",
                "content": RUNTIME_SYSTEM_PROMPT,
            },
            {"role": "user", "content": full_prompt},
        ]

        prompt = self._apply_chat_template(messages)

        return full_prompt, prompt

    def _resolve_input_device(self):
        input_embeddings_getter = getattr(self.model, "get_input_embeddings", None)
        if callable(input_embeddings_getter):
            input_embeddings = input_embeddings_getter()
            weight = getattr(input_embeddings, "weight", None)
            device = getattr(weight, "device", None)
            if device is not None:
                return device
        return getattr(self.model, "device", None)

    def _tokenize_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_device = self._resolve_input_device()
        if input_device is None:
            return inputs
        return {key: value.to(input_device) for key, value in inputs.items()}

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

            worker_error: list[BaseException] = []

            def _generate_in_worker() -> None:
                try:
                    self.model.generate(**generation_kwargs)
                except Exception as exc:
                    worker_error.append(exc)
                    if hasattr(streamer, "on_finalized_text"):
                        streamer.on_finalized_text("", stream_end=True)

            worker = Thread(
                target=_generate_in_worker,
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
            if worker_error:
                raise RuntimeError("LLM generation failed in worker thread") from worker_error[0]
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
