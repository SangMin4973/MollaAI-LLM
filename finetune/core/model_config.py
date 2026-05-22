import torch
from transformers import BitsAndBytesConfig, GenerationConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

dataset_gen_config = GenerationConfig(
    max_new_tokens=1400,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.05,
    no_repeat_ngram_size=4,
)

dataset_pair_gen_config = GenerationConfig(
    max_new_tokens=700,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.05,
    no_repeat_ngram_size=4,
)
