from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str = "llm-jp/llm-jp-3-13b"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"

@dataclass
class DataConfig:
    dataset_name: str = "DeL-TaiseiOzaki/Tengentoppa-sft-v1.0"
    chunk_length: int = 2048
    max_length: int = 2048

@dataclass
class TrainingConfig:
    output_dir: str = "./llm-jp-ft-output"
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 16 
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 1
    logging_steps: int = 100
    warmup_ratio: float = 0.1
    save_strategy: str = "epoch"
    fp16: bool = False
    bf16: bool = True
    deepspeed_config_path: str = "./configs/deepspeed_config.json"
    gradient_checkpointing: bool = True 
    ddp: bool = False  
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100

@dataclass
class EnvironmentConfig:
    master_addr: str = "localhost"
    master_port: str = "29500"
    rank: str = "0"
    local_rank: str = "0"
    world_size: str = "1"

    def set_environment(self):
        """環境変数を設定"""
        import os
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = self.master_port
        os.environ["RANK"] = self.rank
        os.environ["LOCAL_RANK"] = self.local_rank
        os.environ["WORLD_SIZE"] = self.world_size
        os.environ["TOKENIZERS_PARALLELISM"] = "false"