import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from typing import Dict, Any
from ..config.training_config import ModelConfig, TrainingConfig
from ..data.data_processor import DataProcessor
from ..utils.memory_utils import clear_memory

class GemmaTrainer:
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
        # トークナイザーの準備
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # モデルの準備
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            torch_dtype=getattr(torch, self.model_config.torch_dtype),
            device_map=self.model_config.device_map
        )

    def create_trainer(self, train_dataset):
        """Trainerの作成"""
        # データコレーターの準備
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # トレーニング引数の準備
        training_args = TrainingArguments(
            **{k: v for k, v in vars(self.training_config).items() 
               if not k.startswith('_')}
        )

        # Trainerの作成
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

    def train(self):
        """トレーニングの実行"""
        if self.trainer is None:
            raise ValueError("Trainer has not been initialized. Call create_trainer first.")
        
        self.trainer.train()
        
    def save_model(self, output_path: str):
        """モデルの保存"""
        if self.model is None:
            raise ValueError("Model has not been initialized.")
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
    def cleanup(self):
        """メモリの解放"""
        clear_memory(self.trainer)
        self.trainer = None