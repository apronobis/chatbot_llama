from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login

# Replace 'your_token_here' with your actual Hugging Face token
# login(token='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')

model_name = "pretrained/Llama-3.1-8B"
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.bfloat16,
                                         bnb_4bit_use_double_quant=True,
                                         bnb_4bit_quant_type="nf4"
                                         )
tokenizer = AutoTokenizer.from_pretrained(model_name)
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)

input_text = "How can I find the way to make ...   Please continue writing."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = quantized_model.generate(**input_ids, max_new_tokens=10)

print(tokenizer.decode(output[0], skip_special_tokens=False))
