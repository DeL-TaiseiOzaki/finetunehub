import argparse
from configs.training_config import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    EnvironmentConfig,
    LoRAConfig
)
from data.dataprocessor import DataProcessor
from training.trainer import LLMJPTrainer
from utils.memory_utils import clear_memory
import logging

# ロギングの設定
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='LLM-JP ファインチューニング')
    
    # モデル設定
    parser.add_argument('--model_name', type=str, default="llm-jp/llm-jp-3-13b",
                      help='モデル名')
    parser.add_argument('--torch_dtype', type=str, default="bfloat16",
                      help='torch dtype')
    
    # データ設定
    parser.add_argument('--dataset_name', type=str, required=True,
                      help='データセット名')
    parser.add_argument('--chunk_length', type=int, default=2048,
                      help='チャンク長')
    parser.add_argument('--max_length', type=int, default=2048,
                      help='最大長')
    
    # トレーニング基本設定
    parser.add_argument('--training_mode', type=str, choices=['full', 'lora'], default='full',
                      help='トレーニングモード (full または lora)')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='出力ディレクトリ')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='バッチサイズ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32,
                      help='勾配累積ステップ数')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                      help='学習率')
    parser.add_argument('--num_epochs', type=int, default=2,
                      help='エポック数')
    
    # 最適化設定
    parser.add_argument('--fp16', action='store_true',
                      help='FP16を使用')
    parser.add_argument('--bf16', action='store_true',
                      help='BF16を使用')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                      help='勾配チェックポイントを使用')
    parser.add_argument('--deepspeed_config', type=str,
                      help='DeepSpeed設定ファイルのパス')
    
    # LoRA設定
    parser.add_argument('--lora_r', type=int, default=32,
                      help='LoRAのランク')
    parser.add_argument('--lora_alpha', type=int, default=32,
                      help='LoRAのアルファ')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                      help='LoRAのドロップアウト')
    parser.add_argument('--lora_target_modules', type=str, nargs='+',
                      help='LoRAのターゲットモジュール（スペース区切りで複数指定可能）') #"q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj"
    parser.add_argument('--lora_path', type=str,
                      help='事前学習済みLoRA重みのパス')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        # 環境設定
        logger.info("環境を設定しています...")
        env_config = EnvironmentConfig()
        env_config.set_environment()
        
        # 設定の初期化
        logger.info("設定を初期化しています...")
        model_config = ModelConfig(
            model_name=args.model_name,
            torch_dtype=args.torch_dtype
        )
        
        data_config = DataConfig(
            dataset_name=args.dataset_name,
            chunk_length=args.chunk_length,
            max_length=args.max_length
        )
        
        # LoRA設定の初期化（LoRAモードの場合）
        lora_config = None
        if args.training_mode == "lora":
            lora_config = LoRAConfig(
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                lora_path=args.lora_path
            )
        
        training_config = TrainingConfig(
            training_mode=args.training_mode,
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            deepspeed_config_path=args.deepspeed_config,
            lora_config=lora_config
        )
        
        # トレーナーの初期化
        logger.info("トレーナーを初期化しています...")
        trainer = LLMJPTrainer(
            model_config=model_config,
            training_config=training_config
        )
        trainer.setup()
        
        # データの準備
        logger.info("データセットを準備しています...")
        data_processor = DataProcessor(
            tokenizer=trainer.tokenizer,
            chunk_length=data_config.chunk_length,
            max_length=data_config.max_length
        )
        dataset = data_processor.prepare_dataset(data_config.dataset_name)
        
        # トレーニングの実行
        logger.info("トレーニングを開始します...")
        trainer.create_trainer(dataset)
        trainer.train()
        
        # モデルの保存
        logger.info("モデルを保存しています...")
        trainer.save_model(f"{training_config.output_dir}/final_model")
        
        # メモリの解放
        logger.info("メモリを解放しています...")
        clear_memory(trainer.trainer)
        trainer.cleanup()
        
        logger.info("トレーニングが正常に完了しました！")
        
    except Exception as e:
        logger.error(f"トレーニング中にエラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    main()