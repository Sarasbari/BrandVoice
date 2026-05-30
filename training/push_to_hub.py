import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Resolve repository root
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)

# Ensure local training directory takes import priority
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from config import TrainingConfig
except ImportError:
    from training.config import TrainingConfig

def main():
    # 1. Fail loudly if HF_TOKEN is not set
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Set HF_TOKEN env var before running", file=sys.stderr)
        sys.exit(1)

    # Load config
    config = TrainingConfig()
    base_model_id = config.base_model_id
    
    adapter_dir = config.output_dir
    if not os.path.isabs(adapter_dir):
        adapter_dir = os.path.normpath(os.path.join(repo_root, adapter_dir))
        
    merged_model_dir = os.path.normpath(os.path.join(repo_root, "outputs", "merged-model"))
    dataset_path = config.dataset_path
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.normpath(os.path.join(repo_root, dataset_path))

    # Read dataset size
    dataset_size = 0
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset_size = sum(1 for line in f if line.strip())
    else:
        dataset_size = 2995  # Fallback to default scraped dataset size

    print(f"Loading tokenizer for {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # 2. Load base model in 16-bit (NOT 4-bit)
    print(f"Loading base model in 16-bit (float16): {base_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 3. Load LoRA adapter
    print(f"Loading PEFT adapter weights from {adapter_dir}...")
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    except ImportError:
        print("PEFT package not found locally. Ensure 'peft' is installed to run this script.", file=sys.stderr)
        sys.exit(1)

    # 4. Merge adapter weights into base weights
    print("Merging adapter weights into base model (merge_and_unload)...")
    model = model.merge_and_unload()

    # 5. Save merged model locally
    print(f"Saving merged model locally to {merged_model_dir}...")
    os.makedirs(merged_model_dir, exist_ok=True)
    model.save_pretrained(merged_model_dir)
    tokenizer.save_pretrained(merged_model_dir)

    # 6. Auto-generate model card (README.md)
    print("Generating HuggingFace model card (README.md)...")
    model_card_content = f"""---
license: apache-2.0
base_model: {base_model_id}
tags:
- text-generation
- fine-tuned
- qlora
- brand-voice
- notion
---

# BrandVoice — Notion Brand Voice Fine-Tuned Model

This model is a fine-tuned version of [{base_model_id}](https://huggingface.co/{base_model_id}) on the Notion Brand dataset to capture Notion's specific writing style (short, punchy, product-first, action-oriented, and engaging tone).

## Model Details
- **Base Model**: {base_model_id}
- **Method**: QLoRA (4-bit Quantization + LoRA Fine-Tuning)
- **LoRA Hyperparameters**:
  - Rank (r): {config.lora_r}
  - Alpha: {config.lora_alpha}
  - Target Modules: {config.target_modules}
  - Dropout: {config.lora_dropout}
- **Dataset Size**: {dataset_size} instruction-completion pairs
- **Intended Use**: Generating marketing materials, tweets, LinkedIn posts, blog introductions, and changelog updates in the exact Notion brand voice.
"""
    model_card_path = os.path.join(merged_model_dir, "README.md")
    with open(model_card_path, "w", encoding="utf-8") as f:
        f.write(model_card_content)

    # 7. Push to HuggingFace Hub
    print(f"Logging in to HuggingFace Hub using provided HF_TOKEN...")
    try:
        from huggingface_hub import login
        login(token=hf_token)
    except ImportError:
        print("huggingface_hub package not found locally. Ensure it is installed.", file=sys.stderr)
        sys.exit(1)

    print(f"Pushing merged model to HuggingFace Hub repo: {config.hf_repo_id}...")
    model.push_to_hub(config.hf_repo_id)
    tokenizer.push_to_hub(config.hf_repo_id)

    # Print final output path
    print(f"Pushed to https://huggingface.co/{config.hf_repo_id}")

if __name__ == "__main__":
    main()
