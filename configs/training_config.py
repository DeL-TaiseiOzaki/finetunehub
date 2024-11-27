from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class ModelConfig:
    model_name: str = "llm-jp/llm-jp-3-13b"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"

@dataclass
class DataConfig:
    dataset_name: str = "DeL-TaiseiOzaki/ichikara_003_all"
    chunk_length: int = 2048
    max_length: int = 2048

@dataclass
class LoRAConfig:
    r: int = 8  # LoRAのランク
    alpha: int = 16  # スケーリング係数
    dropout: float = 0.05
    bias: str = "none"  # none, all, または lora_only
    task_type: str = "CAUSAL_LM"
    target_modules: Optional[list] = None  # Noneの場合、すべての線形層が対象
    modules_to_save: Optional[list] = None  # 完全なファインチューニングを行うモジュール
    lora_path: Optional[str] = None  # 事前学習済みLoRAの重みをロードする場合のパス

@dataclass
class TrainingConfig:
    # 基本設定
    training_mode: Literal["full", "lora"] = "full"  # トレーニングモードの選択
    output_dir: str = "./llm-jp-ft-output"
    
    # トレーニングのハイパーパラメータ
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 32 
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 2
    logging_steps: int = 100
    warmup_ratio: float = 0.1
    
    # 学習の最適化設定
    save_strategy: str = "epoch"
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    
    # 分散学習設定
    deepspeed_config_path: Optional[str] = "./configs/deepspeed_config.json"
    ddp: bool = False

    # LoRA設定（training_modeがloraの場合に使用）
    lora_config: Optional[LoRAConfig] = None

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