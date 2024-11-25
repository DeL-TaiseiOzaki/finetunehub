# LLM-JP Fine-tuning

このリポジトリには、LLM-JP-3-13Bモデルのフルパラメータファインチューニングのためのコードが含まれています。DeepSpeedを使用して効率的なトレーニングとメモリ最適化を実現しています。

## 機能

- LLM-JP-3-13Bのフルパラメータファインチューニング
- DeepSpeed ZeRO Stage-3の最適化
- A100 GPU向けの効率的なメモリ管理
- モジュール化された拡張可能なコードベース
- カスタムデータセットとトレーニング設定のサポート

## 必要条件

- A100 80GB GPU (推奨8枚)
- Python 3.8以上
- CUDA 11.8以上

## プロジェクト構造

```
finetunehub/
├── __init__.py
├── configs/                 # 設定ファイル
│   ├── __init__.py
│   ├── training_config.py   # トレーニング設定クラス
│   └── deepspeed_config.json # DeepSpeed設定
├── data/                    # データ処理
│   ├── __init__.py
│   └── dataprocessor.py     # データセット処理クラス
├── training/               # トレーニングロジック
│   ├── __init__.py
│   └── trainer.py          # トレーナークラス
├── utils/                  # ユーティリティ関数
│   ├── __init__.py
│   └── memory_utils.py     # メモリ管理ユーティリティ
├── requirements.txt        # 依存パッケージ
└── train.py               # メインスクリプト
```

## インストール

1. リポジトリのクローン：
```bash
git clone https://github.com/yourusername/finetunehub.git
cd finetunehub
```

2. 仮想環境の作成と有効化（推奨）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
.\venv\Scripts\activate  # Windows
```

3. 依存パッケージのインストール：
```bash
pip install -r requirements.txt
```

## 設定

### モデル設定

`configs/training_config.py`のModelConfigクラスで設定を変更できます：

```python
@dataclass
class ModelConfig:
    model_name: str = "llm-jp/llm-jp-3-13b"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
```

### トレーニング設定

`configs/training_config.py`のTrainingConfigクラスでトレーニングパラメータを調整できます：

```python
@dataclass
class TrainingConfig:
    output_dir: str = "./llm-jp-ft-output"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
    # ... その他のパラメータ
```

### DeepSpeed設定

`configs/deepspeed_config.json`でDeepSpeedの設定を変更できます：

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
    // ... その他の設定
}
```

## 使用方法

1. トレーニングの開始：
```bash
deepspeed --num_gpus=8 train.py
```

2. トレーニングの監視：
- トレーニングログは指定された出力ディレクトリに保存されます
- DeepSpeedのログはメモリ使用量とトレーニングの進行状況を表示します

3. 学習済みモデルの取得：
- 最終的なモデルは`{output_dir}/final_model`に保存されます

## カスタマイズ

### カスタムデータセットの使用

1. `configs/training_config.py`のDataConfigを修正：
```python
@dataclass
class DataConfig:
    dataset_name: str = "your_dataset_name"
    chunk_length: int = 2048
    max_length: int = 2048
```

2. 必要に応じて`data/dataprocessor.py`のDataProcessorクラスを拡張して、データセットのフォーマットに対応します。

### メモリ最適化

GPUに応じて以下のパラメータを調整します：

1. TrainingConfigのバッチサイズとgradient accumulation
2. DeepSpeed設定のZeROステージとオフロード設定
3. DataConfigのチャンク長とmax_length

## トラブルシューティング

一般的な問題と解決策：

1. メモリ不足（OOM）：
- バッチサイズを減らす
- gradient accumulation stepsを増やす
- DeepSpeed設定でCPUオフロードを有効化

2. トレーニングが遅い：
- メモリに余裕があればバッチサイズを増やす
- DeepSpeedのバケットサイズを調整
- CPUオフロード設定を確認

