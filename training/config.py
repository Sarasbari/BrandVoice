from dataclasses import dataclass, field, asdict
from typing import List

@dataclass
class TrainingConfig:
    # MODEL
    base_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    dataset_path: str = "data/dataset.jsonl"
    output_dir: str = "outputs/voice-forge-notion"
    hf_repo_id: str = "your-hf-username/voice-forge-notion-mistral-7b"
    
    # QLORA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # TRAINING
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = 50
    
    # QUANTISATION
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    use_nested_quant: bool = False

    def to_dict(self) -> dict:
        """Returns all configuration fields as a flat dictionary."""
        return asdict(self)

    def print_summary(self) -> None:
        """Prints a structured, formatted table of all configuration parameters."""
        print("=" * 60)
        print(f"{'Voice Forge QLoRA Training Configuration Summary':^60}")
        print("=" * 60)
        print(f"{'Parameter':<32} | {'Value'}")
        print("-" * 60)
        
        # Group configuration values logically
        groups = {
            "MODEL CONFIGURATION": [
                "base_model_id", "dataset_path", "output_dir", "hf_repo_id"
            ],
            "QLORA CONFIGURATION": [
                "lora_r", "lora_alpha", "lora_dropout", "target_modules"
            ],
            "TRAINING HYPERPARAMETERS": [
                "num_train_epochs", "per_device_train_batch_size", 
                "gradient_accumulation_steps", "learning_rate", 
                "max_seq_length", "warmup_ratio", "lr_scheduler_type", 
                "fp16", "logging_steps", "save_steps"
            ],
            "QUANTISATION CONFIGURATION": [
                "load_in_4bit", "bnb_4bit_quant_type", 
                "bnb_4bit_compute_dtype", "use_nested_quant"
            ]
        }
        
        config_dict = self.to_dict()
        for group_title, field_names in groups.items():
            print(f"\n[{group_title}]")
            for name in field_names:
                val = config_dict.get(name)
                print(f"  {name:<30} : {val}")
        print("=" * 60)
