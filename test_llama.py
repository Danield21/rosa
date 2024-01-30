from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

print("Running...")
model_name = 'meta-llama/Llama-2-7b-hf'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
#
# # Put on multiple gpus
