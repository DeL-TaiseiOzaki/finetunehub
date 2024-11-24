from typing import Dict, List, Any
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from functools import partial

class DataProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, chunk_length: int = 2048):
        self.tokenizer = tokenizer
        self.chunk_length = chunk_length

    def format_prompt(self, sample: Dict[str, str]) -> str:
        """プロンプトをGemmaの形式に整形"""
        instruction = f"<start_of_turn>user\n{sample['instruction']}<end_of_turn>\n"
        input_text = (f"<start_of_turn>user\n{sample['input']}<end_of_turn>\n" 
                     if sample.get("input") else "")
        output = f"<start_of_turn>assistant\n{sample['output']}<end_of_turn>"
        return instruction + input_text + output

    def template_dataset(self, sample: Dict[str, str]) -> Dict[str, str]:
        """データセットにテンプレートを適用"""
        sample["text"] = f"{self.format_prompt(sample)}{self.tokenizer.eos_token}"
        return sample

    def chunk_data(self, samples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """データを指定された長さのチャンクに分割"""
        concatenated = {k: sum(samples[k], []) for k in samples.keys()}
        total_length = len(concatenated[list(samples.keys())[0]])
        result = {k: [] for k in samples.keys()}
        
        for i in range(0, total_length, self.chunk_length):
            for k in samples.keys():
                result[k].append(concatenated[k][i:i + self.chunk_length])
                if len(result[k][-1]) < self.chunk_length:
                    remainder = self.chunk_length - len(result[k][-1])
                    result[k][-1].extend([self.tokenizer.pad_token_id] * remainder)
        return result

    def prepare_dataset(self, dataset_name: str):
        """データセットの準備"""
        # データセットのロード
        dataset = load_dataset(dataset_name, split="train")
        
        # テンプレートの適用
        dataset = dataset.map(
            self.template_dataset,
            remove_columns=list(dataset.features)
        )
        
        # トークン化
        tokenized_dataset = dataset.map(
            lambda x: self.tokenizer(x["text"]),
            batched=True,
            remove_columns=list(dataset.features)
        )
        
        # チャンク化
        chunked_dataset = tokenized_dataset.map(
            self.chunk_data,
            batched=True,
        )
        
        return chunked_dataset