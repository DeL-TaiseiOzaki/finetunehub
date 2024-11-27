from configs.training_config import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    EnvironmentConfig
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

def main():
    try:
        # 環境設定
        logger.info("環境を設定しています...")
        env_config = EnvironmentConfig()
        env_config.set_environment()
        
        # 設定の初期化
        logger.info("設定を初期化しています...")
        model_config = ModelConfig()
        data_config = DataConfig()
        training_config = TrainingConfig()
        
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
        print(len(dataset))
        
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