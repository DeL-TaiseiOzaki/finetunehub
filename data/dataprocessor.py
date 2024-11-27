from typing import Dict, List, Any
from datasets import load_dataset
from transformers import PreTrainedTokenizer
import numpy as np
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

    def format_prompt(self, instruction: str, input_text: str = '', output: str = '') -> str:
        """プロンプトをLLM-JPの形式に整形"""
        if input_text:
            return f"\n{instruction}\n\n{input_text}\n\n{output}"
        return f"{instruction}\n\n{output}"

    def template_dataset(self, examples: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """データセットにテンプレートを適用（バッチ処理対応）"""
        texts = []
        for i in range(len(examples['instruction'])):
            prompt = self.format_prompt(
                instruction=examples['instruction'][i],
                input_text=examples.get('input', [''] * len(examples['instruction']))[i],
                output=examples['output'][i]
            )
            texts.append(f"{prompt}{self.tokenizer.eos_token}")
        
        return {
            "text": np.array(texts, dtype=np.string_)
        }

    def prepare_dataset(self, dataset_name: str):
        """データセットの準備"""
        logger.info(f"データセット '{dataset_name}' をロードしています...")
        dataset = load_dataset(dataset_name, split="train")
        
        logger.info("プロンプトテンプレートを適用しています...")
        dataset = dataset.map(
            self.template_dataset,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Applying templates"
        )
        
        def tokenize_function(examples):
            model_inputs = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_token_type_ids=False
            )
            
            # labelsの設定
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            
            return model_inputs
        
        logger.info("データセットをトークン化しています...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        logger.info(f"データセットの準備が完了しました。サイズ: {len(tokenized_dataset)}")
        return tokenized_dataset