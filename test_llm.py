from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch


torch_stub = types.SimpleNamespace(float16="float16", Tensor=object)
transformers_stub = types.SimpleNamespace(
    AutoModelForCausalLM=object,
    AutoTokenizer=object,
    GenerationConfig=lambda **kwargs: kwargs,
    TextIteratorStreamer=object,
)

sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("transformers", transformers_stub)

import llm


class _EmptyStreamer:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def __iter__(self):
        return iter(())

    def on_finalized_text(self, _text: str, stream_end: bool = False) -> None:
        return None


class LlmStreamAnswerTests(unittest.TestCase):
    def test_init_uses_finetuned_tokenizer_by_default(self) -> None:
        tokenizer_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        model_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        class _TokenizerLoader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                tokenizer_calls.append((args, kwargs))
                return object()

        class _ModelLoader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                model_calls.append((args, kwargs))
                return object()

        with patch.object(llm, "AutoTokenizer", _TokenizerLoader), patch.object(llm, "AutoModelForCausalLM", _ModelLoader):
            llm.QwenChat()

        self.assertEqual(tokenizer_calls[0][0][0], llm.MODEL_NAME)
        self.assertEqual(tokenizer_calls[0][1], {"extra_special_tokens": {}})
        self.assertEqual(model_calls[0][0][0], llm.MODEL_NAME)

    def test_tokenize_prompt_moves_inputs_to_embedding_device_for_auto_device_map(self) -> None:
        chat = object.__new__(llm.QwenChat)

        class _Tensor:
            def __init__(self) -> None:
                self.calls: list[object] = []

            def to(self, device):
                self.calls.append(device)
                return f"moved:{device}"

        class _Tokenizer:
            def __call__(self, _prompt, return_tensors="pt"):
                return {"input_ids": tensor}

        class _Embeddings:
            class _Weight:
                device = "cuda:0"

            weight = _Weight()

        class _Model:
            device = "cuda:0"
            hf_device_map = {"model.embed_tokens": 0}

            def get_input_embeddings(self):
                return _Embeddings()

        tensor = _Tensor()
        chat.tokenizer = _Tokenizer()
        chat.model = _Model()

        inputs = chat._tokenize_prompt("hello")

        self.assertEqual(inputs["input_ids"], "moved:cuda:0")
        self.assertEqual(tensor.calls, ["cuda:0"])

    def test_get_prompt_retries_without_enable_thinking(self) -> None:
        chat = object.__new__(llm.QwenChat)

        class _TokenizerWithoutEnableThinking:
            def apply_chat_template(self, _messages, **kwargs):
                if "enable_thinking" in kwargs:
                    raise TypeError("unexpected keyword argument 'enable_thinking'")
                return "prompt-without-enable-thinking"

        chat.tokenizer = _TokenizerWithoutEnableThinking()

        full_prompt, prompt = chat.get_prompt("hello")

        self.assertIn("hello", full_prompt)
        self.assertEqual(prompt, "prompt-without-enable-thinking")

    def test_returns_fallback_when_generate_thread_fails(self) -> None:
        chat = object.__new__(llm.QwenChat)
        chat.tokenizer = object()

        class _BrokenModel:
            device = None

            def generate(self, **_kwargs) -> None:
                raise RuntimeError("generation failed")

        chat.model = _BrokenModel()
        chat.get_prompt = lambda query: ("prompt", "prompt")
        chat._tokenize_prompt = lambda prompt: {"input_ids": object()}

        with patch.object(llm, "TextIteratorStreamer", _EmptyStreamer):
            chunks = list(chat.stream_answer("hello", request_id="test"))

        self.assertEqual(chunks, ["오류가 발생했습니다."])


if __name__ == "__main__":
    unittest.main()
