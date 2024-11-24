# 8. `README.md`

使用方法を更新します。

## Fine-tuning Project

このプロジェクトは、HuggingFaceのモデルをLoRAまたはフルパラメータでファインチューニングするためのスクリプトを提供します。コードは理解しやすいようにモジュール化され、設定は `config.yaml` で管理されています。

### ディレクトリ構成

```
fine_tuning_project/
├── configs/
│   └── config.yaml
├── logs/
│   └── (ログファイル)
├── scripts/
│   ├── data/
│   │   └── data_processing.py
│   ├── main.py
│   ├── models/
│   │   └── model_utils.py
│   ├── run_fine_tuning.sh
│   └── training/
│       └── trainer.py
├── README.md
└── requirements.txt
```

### セットアップ

必要なパッケージをインストールします。

```bash
pip install -r requirements.txt
```

### 使用方法

#### 設定ファイルの編集

`configs/config.yaml` を開き、必要な設定を編集します。

- **モデルとデータセットの指定**:
  - `model.name_or_path`: 使用するモデルの名前またはパス。
  - `dataset.name`: 使用するデータセットの名前。

- **出力ディレクトリの指定**:
  - `training.output_dir`: ファインチューニング済みモデルの保存先ディレクトリ。

- **トレーニングパラメータの設定**:
  - `training.num_train_epochs`: エポック数。
  - `training.per_device_train_batch_size`: バッチサイズ。
  - `training.block_size`: 最大シーケンス長。
  - `training.learning_rate`: 学習率。
  - `training.seed`: ランダムシード。

- **精度とトレーニング戦略**:
  - `training.fp16`: FP16精度を使用する場合は `true`。
  - `training.bf16`: BF16精度を使用する場合は `true`。
  - `training.gradient_checkpointing`: 勾配チェックポイントを有効にする場合は `true`。

- **LoRAの設定**:
  - `lora.use_lora`: LoRAを使用する場合は `true`。
  - `lora.r`: LoRAのランク。
  - `lora.alpha`: LoRAのアルファ値。
  - `lora.dropout`: LoRAのドロップアウト率。
  - `lora.target_modules`: LoRAを適用するモジュール。

- **DDPの設定**:
  - `training.ddp`: マルチGPUトレーニングを使用する場合は `true`。

#### トレーニングの実行

シェルスクリプトに実行権限を与え、実行します。

```bash
cd scripts
chmod +x run_fine_tuning.sh
./run_fine_tuning.sh
```

### 注意事項

- **設定の一元管理**: 設定はすべて `configs/config.yaml` で管理されます。これにより、設定の変更が容易になり、再現性が高まります。

- **LoRAとフルパラメータの切り替え**: `config.yaml` 内の `lora.use_lora` を `true` または `false` に設定することで切り替えが可能です。

- **マルチGPUトレーニング**: `training.ddp` を `true` に設定すると、マルチGPUトレーニングが可能です。

- **メモリ使用量**: 大規模なモデルやデータセットを使用する場合、メモリ不足に注意してください。

## ライセンス

このプロジェクトはApache-2.0ライセンスの下で公開されています。
