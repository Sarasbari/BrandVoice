import os
import sys
import json
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Resolve repository root
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)

# Put training folder in sys.path to load TrainingConfig
sys.path.append(os.path.join(repo_root, "training"))
try:
    from config import TrainingConfig
except ImportError:
    from training.config import TrainingConfig

def clean_output(text: str) -> str:
    """Helper to clean output formatting."""
    return text.replace("<s>", "").replace("</s>", "").strip()

def main():
    # 1. Load config
    config = TrainingConfig()
    base_model_id = config.base_model_id
    
    adapter_dir = config.output_dir
    if not os.path.isabs(adapter_dir):
        adapter_dir = os.path.normpath(os.path.join(repo_root, adapter_dir))
        
    results_json_path = os.path.join(repo_root, "eval", "comparison_results.json")

    # Hardcoded test prompts mapping to the format used in our dataset
    prompts = [
        {
            "instruction": "Write a tweet announcing a new feature",
            "input": "Feature: dark mode. Audience: developers. Tone: excited"
        },
        {
            "instruction": "Write a LinkedIn post about a product update",
            "input": "Update: tables support formulas. Audience: business users. Tone: helpful"
        },
        {
            "instruction": "Write a blog intro about why teams need a single source of truth",
            "input": "Topic: single source of truth. Audience: startup founders"
        },
        {
            "instruction": "Write a changelog entry for a new feature",
            "input": "Feature: drag-to-resize columns in database view. Platform: desktop"
        },
        {
            "instruction": "Write a tweet about productivity",
            "input": "Topic: switching between too many tools. Tone: relatable"
        }
    ]

    # Load tokenizer
    print(f"Loading tokenizer for {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Configure 4-bit quantisation for base model loading
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype, torch.float16)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )

    base_outputs = []

    # 1. Load base model (4-bit) and generate
    print(f"Loading base model: {base_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    print("Generating base model outputs...")
    for idx, prompt_info in enumerate(prompts):
        instruction = prompt_info["instruction"]
        input_context = prompt_info["input"]
        
        # Build Mistral instruct chat template prompt
        prompt_text = f"<s>[INST] {instruction}\n{input_context} [/INST]"
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract and decode generated tokens only
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generation = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        base_outputs.append(generation)
        print(f"Base Output {idx+1}/{len(prompts)} completed.")

    # 2. Unload base model
    print("Unloading base model and clearing CUDA memory...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Load fine-tuned model via PeftModel
    finetuned_outputs = []
    print(f"Reloading base model in 4-bit for adapter merging...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    print(f"Loading PEFT adapter weights from {adapter_dir}...")
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    except ImportError:
        print("PEFT package not found locally. To run this script locally, ensure 'peft' is installed.")
        sys.exit(1)
        
    print("Generating fine-tuned model outputs...")
    for idx, prompt_info in enumerate(prompts):
        instruction = prompt_info["instruction"]
        input_context = prompt_info["input"]
        
        prompt_text = f"<s>[INST] {instruction}\n{input_context} [/INST]"
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generation = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        finetuned_outputs.append(generation)
        print(f"Fine-tuned Output {idx+1}/{len(prompts)} completed.")

    # 4. Save results to comparison_results.json
    comparison_data = []
    for i in range(len(prompts)):
        comparison_data.append({
            "instruction": prompts[i]["instruction"],
            "input": prompts[i]["input"],
            "base_output": base_outputs[i],
            "finetuned_output": finetuned_outputs[i]
        })
        
    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"Comparison results saved successfully to {results_json_path}")

    # 5. Print formatted side-by-side comparison tables
    print("\n" + "=" * 80)
    print("MISTRAL BASE VS BRANDVOICE FINE-TUNED COMPARISON")
    print("=" * 80)
    
    for idx, item in enumerate(comparison_data):
        print(f"\nPROMPT {idx+1}: {item['instruction']}")
        print(f"CONTEXT:  {item['input']}")
        print("-" * 80)
        print(">>> MISTRAL BASE MODEL OUTPUT:")
        print(item["base_output"])
        print("-" * 80)
        print(">>> BRANDVOICE (FINE-TUNED) OUTPUT:")
        print(item["finetuned_output"])
        print("=" * 80)

if __name__ == "__main__":
    main()
