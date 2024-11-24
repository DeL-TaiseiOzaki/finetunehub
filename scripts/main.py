import logging
import os
import torch
from data_processing import load_and_prepare_datasets
from model_utils import load_and_prepare_model
from trainer import train_model
import yaml
import wandb
import gc

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
def setup_logging(logging_dir):
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logging_dir, 'training.log'),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    return logger

def main():
    # GPUメモリをクリア
    clear_gpu_memory()
    
    # 設定ファイルの読み込み
    with open("configs/config_gemma.yaml", 'r') as f:
        config = yaml.safe_load(f)

    logger = setup_logging(config['logging']['logging_dir'])
    logger.info("ファインチューニングプロセスを開始します")

    #wandbの使用設定を取得
    use_wandb = config['logging'].get('use_wandb', False)

    #wandbの初期化
    if use_wandb:
        wandb.init(project="gemma-full-sft")  

    # デバイスとDDPの設定
    ddp = config['training']['ddp']
    if ddp:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = int(config['training']['seed'])
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # モデルの読み込みと準備
    model, tokenizer = load_and_prepare_model(config, device, logger)

    # データセットの読み込みと準備
    train_dataset = load_and_prepare_datasets(config, tokenizer, logger)

    # モデルのトレーニング
    train_model(config, model, tokenizer, train_dataset, device, logger)

    # wandbの終了
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
