import gc
import torch
import deepspeed
from typing import Optional
from transformers import Trainer

def clear_memory(trainer: Optional[Trainer] = None):
    """メモリの解放を行う"""
    if trainer is not None:
        del trainer
    
    gc.collect()
    deepspeed.runtime.utils.empty_cache()
    torch.cuda.empty_cache()