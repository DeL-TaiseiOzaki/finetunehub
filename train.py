from configs.training_config import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    EnvironmentConfig
)
from data.dataprocessor import DataProcessor
from training.trainer import GemmaTrainer
from utils.memory_utils import clear_memory

def main():
    # 環境設定
    env_config = EnvironmentConfig()
    env_config.set_environment()
    
    # 設定の初期化
    model_config = ModelConfig()
    data_config = DataConfig()
    training_config = TrainingConfig()
    
    # トレーナーの初期化
    trainer = GemmaTrainer(model_config, training_config)
    trainer.setup()
    
    # データの準備
    data_processor = DataProcessor(
        tokenizer=trainer.tokenizer,
        chunk_length=data_config.chunk_length
    )
    dataset = data_processor.prepare_dataset(data_config.dataset_name)
    
    # トレーナーの作成と学習実行
    trainer.create_trainer(dataset)
    trainer.train()
    
    # モデルの保存
    trainer.save_model(f"{training_config.output_dir}/final_model")
    
    # メモリの解放
    trainer.cleanup()

if __name__ == "__main__":
    main()