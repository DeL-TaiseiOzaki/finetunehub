import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from configs.training_config import ModelConfig, TrainingConfig
from typing import Optional
import logging
from utils.memory_utils import clear_memory

logger = logging.getLogger(__name__)

class LLMJPTrainer:
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.tokenizer = None
        self.model = None
        self.trainer = None

   def setup(self):
        """モデルとトークナイザーの初期化"""
        logger.info("トークナイザーを初期化しています...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("モデルを初期化しています...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            torch_dtype=getattr(torch, self.model_config.torch_dtype),
            device_map="auto",
            use_cache=False,
            load_in_8bit=True
        )

        # LoRAモードの場合の処理
        if self.training_config.training_mode == "lora":
            if self.training_config.lora_config is None:
                raise ValueError("LoRAモードが選択されていますが、LoRA設定が指定されていません。")
            
            logger.info("LoRAの設定を適用しています...")
            lora_config = self.training_config.lora_config

            # デフォルトのターゲットモジュールを設定
            if lora_config.target_modules is None:
                lora_config.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

            # LoRA設定の作成
            peft_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
                bias=lora_config.bias,
                task_type=lora_config.task_type,
                target_modules=lora_config.target_modules,
                modules_to_save=lora_config.modules_to_save,
                inference_mode=False
            )

            # モデルをkbit trainingの準備
            self.model = prepare_model_for_kbit_training(self.model)
            
            # LoRAモデルの作成
            self.model = get_peft_model(self.model, peft_config)

            if lora_config.lora_path:
                logger.info(f"事前学習済みLoRA重みを {lora_config.lora_path} からロードしています...")
                self.model.load_adapter(lora_config.lora_path, "default")

            self.model.print_trainable_parameters()

        # DDPの設定
        if hasattr(self.model, 'is_parallelizable') and hasattr(self.model, 'model_parallel'):
            self.model.is_parallelizable = True
            self.model.model_parallel = True

        # Gradient Checkpointingの有効化
        logger.info("Gradient Checkpointingを有効化しています...")
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        
        # モデルパラメータの最適化設定
        self.model.config.use_cache = False
        torch.cuda.empty_cache()

    def create_trainer(self, train_dataset):
        """Trainerの作成"""
        logger.info("トレーニング引数を設定しています...")
        
        # DeepSpeed設定の調整（LoRAモードでは無効化）
        deepspeed_config = None
        if self.training_config.training_mode == "full" and self.training_config.deepspeed_config_path:
            deepspeed_config = self.training_config.deepspeed_config_path

        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            num_train_epochs=self.training_config.num_train_epochs,
            warmup_ratio=self.training_config.warmup_ratio,
            logging_steps=self.training_config.logging_steps,
            save_strategy=self.training_config.save_strategy,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            deepspeed=deepspeed_config,
            ddp_find_unused_parameters=False if self.training_config.ddp else None,
            remove_unused_columns=False,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            max_grad_norm=self.training_config.max_grad_norm
        )

        logger.info("Trainerを初期化しています...")
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )

    def train(self):
        """トレーニングの実行"""
        if self.trainer is None:
            raise ValueError("Trainerが初期化されていません。create_trainer()を先に実行してください。")
        
        try:
            logger.info("トレーニングを開始します...")
            self.trainer.train()
        except Exception as e:
            logger.error(f"トレーニングエラー: {str(e)}")
            if "out of memory" in str(e):
                logger.error("OOMが検出されました。現在のメモリ使用状況:")
                logger.error(torch.cuda.memory_summary())
            raise

    def save_model(self, output_path: str):
        """モデルの保存"""
        if self.model is None:
            raise ValueError("モデルが初期化されていません。")
        
        logger.info(f"モデルを {output_path} に保存しています...")
        if self.training_config.training_mode == "lora":
            self.model.save_pretrained(output_path)  # LoRAの重みのみを保存
        else:
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)

    def cleanup(self):
        """メモリの解放"""
        logger.info("メモリを解放しています...")
        clear_memory(self.trainer)
        self.trainer = None