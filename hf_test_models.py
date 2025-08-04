# hf_test_models.py
from huggingface_hub import HfApi
import os

api = HfApi()

# List models that are available for inference (i.e., hosted/warm models)
models = api.list_models(filter="text2text-generation", limit=10)

print("Available text2text-generation models:")
for model in models:
    print(f"- {model.modelId}")
