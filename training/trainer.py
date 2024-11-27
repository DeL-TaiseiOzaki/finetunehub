import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
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
            device_map=self.model_config.device_map,
            use_cache=not self.training_config.gradient_checkpointing
        )

        if self.training_config.gradient_checkpointing:
            logger.info("Gradient Checkpointingを有効化しています...")
            self.model.gradient_checkpointing_enable()

    def create_trainer(self, train_dataset):
        """Trainerの作成"""
        logger.info("トレーニング引数を設定しています...")
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
            deepspeed=self.training_config.deepspeed_config_path,
            ddp_find_unused_parameters=False if self.training_config.ddp else None,
            remove_unused_columns=False,  # 追加
        )

        logger.info("Trainerを初期化しています...")
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,  # 追加
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
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

    def cleanup(self):
        """メモリの解放"""
        logger.info("メモリを解放しています...")
        clear_memory(self.trainer)
        self.trainer = None