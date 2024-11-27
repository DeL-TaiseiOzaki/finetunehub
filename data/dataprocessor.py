from typing import Dict, List, Any
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from functools import partial
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer, 
                 chunk_length: int = 2048,
                 max_length: int = 2048):
        self.tokenizer = tokenizer
        self.chunk_length = chunk_length
        self.max_length = max_length

    def format_prompt(self, sample: Dict[str, str]) -> str:
        """プロンプトをLLM-JPの形式に整形"""
        instruction = sample['instruction']
        input_text = sample.get('input', '')
        output = sample['output']
        
        if input_text:
            prompt = f"\n{instruction}\n\n{input_text}\n\n{output}"
        else:
            prompt = f"{instruction}\n\n{output}"
            
        return prompt

    def template_dataset(self, sample: Dict[str, str]) -> Dict[str, str]:
        """データセットにテンプレートを適用"""
        sample["text"] = f"{self.format_prompt(sample)}{self.tokenizer.eos_token}"
        return sample

    def prepare_dataset(self, dataset_name: str):
        """データセットの準備"""
        logger.info(f"データセット '{dataset_name}' をロードしています...")
        dataset = load_dataset(dataset_name, split="train")
        
        logger.info("プロンプトテンプレートを適用しています...")
        dataset = dataset.map(
            self.template_dataset,
            remove_columns=dataset.column_names,
            desc="Applying templates"
        )
        
        logger.info("データセットをトークン化しています...")
        def tokenize_function(self, examples):
            # トークン化
            model_inputs = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_token_type_ids=False
            )
            
            # labelsの追加（input_idsと同じ値を使用）
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            
            return model_inputs
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        logger.info(f"データセットの準備が完了しました。サイズ: {len(tokenized_dataset)}")
        return tokenized_dataset