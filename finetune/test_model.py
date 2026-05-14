from transformers import AutoTokenizer, AutoModelForCausalLM

repo_id = "ssaann/eng_conversation_sft"

tokenizer = AutoTokenizer.from_pretrained(repo_id, extra_special_tokens={})
model = AutoModelForCausalLM.from_pretrained(
      repo_id,
      device_map="auto",
      torch_dtype="auto",
  )

messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, introduce yourself briefly."},
  ]

inputs = tokenizer.apply_chat_template(
     messages,
      add_generation_prompt=True,
      return_tensors="pt",
  ).to(model.device)

output = model.generate(
      inputs,
      max_new_tokens=128,
      temperature=0.6,
      top_p=0.95,
      top_k=20,
  )

print(tokenizer.decode(output[0], skip_special_tokens=True))
