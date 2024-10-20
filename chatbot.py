import transformers
import torch

model_id = "pretrained/LLama-3.2-3B"

from huggingface_hub import login

# Replace 'your_token_here' with your actual Hugging Face token
# login(token='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')

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]


def getAnswer(message):
    # global messages
    # messages.append({ 'role': 'user', 'content': message })
    messages = [
        { 'role': 'user', 'content': message }
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]['generated_text'][-1]['content']
# print(outputs[0]["generated_text"][-1])
