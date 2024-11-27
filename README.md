# LLM-JP Finetuning Framework

LLM-JP（Japanese Large Language Model）のファインチューニングを行うためのフレームワークです。フルパラメータファインチューニングとLoRAの両方に対応しています。

## 特徴

- フルパラメータファインチューニングとLoRAの柔軟な切り替え
- DeepSpeedによる分散学習のサポート
- メモリ効率を考慮した実装（Gradient Checkpointing等）
- 詳細なログ出力
- コマンドラインからの柔軟な設定

## 必要要件

- Python 3.8以上
- PyTorch 2.0以上
- transformers
- datasets
- peft（LoRA使用時）
- deepspeed（分散学習時）

## インストール

```bash
git clone [repository-url]
cd llm-jp-finetuning
pip install -r requirements.txt
```

## 使用方法

### 基本的な使用方法

1. フルパラメータファインチューニング：

```bash
python train.py \
    --model_name "llm-jp/llm-jp-3-13b" \
    --dataset_name "your-dataset" \
    --output_dir "./full-ft-output" \
    --training_mode "full" \
    --batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --num_epochs 2 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed_config "./configs/deepspeed_config.json"
```

2. LoRAファインチューニング：

```bash
python train.py \
    --model_name "llm-jp/llm-jp-3-13b" \
    --dataset_name "your-dataset" \
    --output_dir "./lora-ft-output" \
    --training_mode "lora" \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --num_epochs 3 \
    --bf16 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj"
```

### コマンドライン引数

#### 基本設定
- `--model_name`: モデル名（デフォルト: "llm-jp/llm-jp-3-13b"）
- `--dataset_name`: データセット名（必須）
- `--output_dir`: 出力ディレクトリ（必須）
- `--training_mode`: トレーニングモード（"full" または "lora"、デフォルト: "full"）

#### データ設定
- `--chunk_length`: チャンク長（デフォルト: 2048）
- `--max_length`: 最大長（デフォルト: 2048）

#### トレーニング設定
- `--batch_size`: バッチサイズ（デフォルト: 2）
- `--gradient_accumulation_steps`: 勾配累積ステップ数（デフォルト: 32）
- `--learning_rate`: 学習率（デフォルト: 1e-5）
- `--num_epochs`: エポック数（デフォルト: 2）

#### 最適化設定
- `--fp16`: FP16を使用
- `--bf16`: BF16を使用
- `--gradient_checkpointing`: 勾配チェックポイントを使用
- `--deepspeed_config`: DeepSpeed設定ファイルのパス

#### LoRA設定
- `--lora_r`: LoRAのランク（デフォルト: 8）
- `--lora_alpha`: LoRAのアルファ（デフォルト: 16）
- `--lora_dropout`: LoRAのドロップアウト（デフォルト: 0.05）
- `--lora_target_modules`: LoRAのターゲットモジュール（スペース区切りで複数指定可能）
- `--lora_path`: 事前学習済みLoRA重みのパス

## プロジェクト構造

```
llm-jp-finetuning/
├── configs/
│   ├── __init__.py
│   ├── training_config.py
│   └── deepspeed_config.json
├── data/
│   ├── __init__.py
│   └── dataprocessor.py
├── training/
│   ├── __init__.py
│   └── trainer.py
├── utils/
│   ├── __init__.py
│   └── memory_utils.py
├── train.py
├── requirements.txt
└── README.md
```

## デバッグとトラブルシューティング

1. メモリエラー
   - Gradient Checkpointingを有効化（`--gradient_checkpointing`）
   - バッチサイズを減らす
   - LoRAを使用する

2. DeepSpeedエラー
   - DeepSpeed設定ファイルのパスが正しいか確認
   - LoRAモードではDeepSpeedが無効化されることに注意

3. トークン化エラー
   - `max_length`パラメータを調整
   - データセットの前処理を確認

## ライセンス

MIT License

## 謝辞

このプロジェクトは以下のライブラリを使用しています：
- Transformers by Hugging Face
- PEFT by Hugging Face
- DeepSpeed by Microsoft
- PyTorch by Facebook AI Research

## 注意事項

- 学習には大量のGPUメモリが必要です
- フルパラメータファインチューニングには高性能なGPUが必須です
- LoRAを使用することで、より少ないリソースでのファインチューニングが可能です

## 貢献

バグ報告や機能追加の提案は、Issuesで受け付けています。プルリクエストも歓迎します。