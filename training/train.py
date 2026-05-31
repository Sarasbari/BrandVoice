import os
import sys
import json
import math
import inspect
import torch
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# Ensure local training directory takes import priority to avoid clashes with global config packages
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Resolve repository root
repo_root = os.path.dirname(script_dir)

try:
    from config import TrainingConfig
except ImportError:
    # Fallback to absolute import if run as module
    from training.config import TrainingConfig

def get_timestamp() -> str:
    """Returns the current formatted timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    try:
        # 1. Load config
        print(f"[{get_timestamp()}] Loading configuration...")
        config = TrainingConfig()
        config.print_summary()

        # Resolve relative paths in config to be absolute relative to repo root
        dataset_path = config.dataset_path
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.normpath(os.path.join(repo_root, dataset_path))

        output_dir = config.output_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.normpath(os.path.join(repo_root, output_dir))

        loss_curve_path = "outputs/loss_curve.json"
        if not os.path.isabs(loss_curve_path):
            loss_curve_path = os.path.normpath(os.path.join(repo_root, loss_curve_path))

        # 2. Load tokenizer
        print(f"[{get_timestamp()}] Loading tokenizer for model: {config.base_model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print(f"[{get_timestamp()}] Tokenizer loaded successfully. Pad token: {tokenizer.pad_token}, padding side: {tokenizer.padding_side}")

        # 3. Load base model with 4-bit quantisation
        print(f"[{get_timestamp()}] Configuring 4-bit quantisation...")
        cuda_supports_bf16 = bool(
            torch.cuda.is_available()
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
        use_bf16 = cuda_supports_bf16
        use_fp16 = bool(config.fp16 and not use_bf16)
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        print(
            f"[{get_timestamp()}] Mixed precision: "
            f"bf16={use_bf16}, fp16={use_fp16}, bnb_compute_dtype={compute_dtype}"
        )
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )

        print(f"[{get_timestamp()}] Loading base model (this might take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
        )
        
        # Configure model padding and cache settings
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False  # Required when using gradient checkpointing
        
        print(f"[{get_timestamp()}] Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)

        # 4. Apply LoRA via LoraConfig + get_peft_model()
        print(f"[{get_timestamp()}] Configuring LoRA adapter...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        # Calculate trainable parameters
        trainable_params = 0
        all_params = 0
        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        trainable_percent = 100 * trainable_params / all_params
        print(f"Trainable params: {trainable_params} | All params: {all_params} | Trainable %: {trainable_percent:.4f}")

        # 5. Load and format dataset
        print(f"[{get_timestamp()}] Loading and formatting dataset from {dataset_path}...")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please build the dataset first.")

        formatted_data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                instruction = item.get("instruction", "")
                input_context = item.get("input", "")
                output = item.get("output", "")
                
                # Apply Mistral chat template formatting
                formatted_text = f"<s>[INST] {instruction}\n{input_context} [/INST] {output} </s>"
                formatted_data.append({"text": formatted_text})
        
        train_dataset = Dataset.from_list(formatted_data)
        print(f"[{get_timestamp()}] Loaded {len(train_dataset)} training samples")

        # 6. Configure SFTTrainer
        print(f"[{get_timestamp()}] Setting up SFTConfig and SFTTrainer...")
        steps_per_epoch = math.ceil(len(train_dataset) / (config.per_device_train_batch_size * config.gradient_accumulation_steps))
        total_training_steps = steps_per_epoch * config.num_train_epochs
        warmup_steps = max(1, int(total_training_steps * config.warmup_ratio)) if config.warmup_ratio > 0 else 0
        os.environ.setdefault("TENSORBOARD_LOGGING_DIR", os.path.join(output_dir, "logs"))

        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=warmup_steps,
            lr_scheduler_type=config.lr_scheduler_type,
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            report_to="none",
            save_total_limit=2,
            dataset_text_field="text",
            max_length=config.max_seq_length,
        )

        trainer_kwargs = {
            "model": model,
            "train_dataset": train_dataset,
            "args": sft_config,
        }
        trainer_params = inspect.signature(SFTTrainer.__init__).parameters
        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = tokenizer
        else:
            raise TypeError("This TRL version does not support passing a tokenizer or processing_class to SFTTrainer.")

        trainer = SFTTrainer(**trainer_kwargs)

        # 7. Train
        print(f"[{get_timestamp()}] Starting training loop (SFTTrainer)...")
        trainer.train()

        print(f"[{get_timestamp()}] Saving adapter to {output_dir}...")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[{get_timestamp()}] Adapter saved successfully.")

        # 8. Save loss curve data
        print(f"[{get_timestamp()}] Extracting training loss history...")
        loss_history = []
        for log in trainer.state.log_history:
            if "loss" in log and "step" in log:
                loss_history.append({
                    "step": log["step"],
                    "loss": log["loss"]
                })
        
        os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
        with open(loss_curve_path, "w", encoding="utf-8") as f:
            json.dump(loss_history, f, indent=2)
        print(f"[{get_timestamp()}] Loss curve saved to {loss_curve_path}")

        final_loss = loss_history[-1]["loss"] if loss_history else "N/A"
        print(f"[{get_timestamp()}] Final training loss: {final_loss}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "=" * 80)
            print("CUDA OOM — reduce per_device_train_batch_size in config.py or enable gradient_checkpointing")
            print("Reduce per_device_train_batch_size in config.py")
            print("=" * 80 + "\n")
        raise e

if __name__ == "__main__":
    main()
