# %%

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

dtype = torch.bfloat16
device = torch.device("cuda")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it", device_map="auto", torch_dtype=dtype
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

# %%


prompts = [
    "Hello, how are you?",
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
    "What is the capital of Spain?",
]

prompts = [
    [
        {"role": "user", "content": p},
    ]
    for p in prompts
]

prompts = [
    tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
    for p in prompts
]

input_ids = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

# %%

output_ids = model.generate(input_ids["input_ids"], max_new_tokens=10)

# %%

print(input_ids["input_ids"].shape)
print(output_ids.shape)

generated_tokens = output_ids[:, input_ids["input_ids"].shape[1] :]

decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(decoded_output)
# %%
