import os
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

def train_model(config, model, tokenizer, train_dataset, device, logger):
    # データコレクタの設定
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 数値型のパラメータを適切な型に変換
    num_train_epochs = int(config['training']['num_train_epochs'])
    per_device_train_batch_size = int(config['training']['per_device_train_batch_size'])
    learning_rate = float(config['training']['learning_rate'])
    logging_steps = int(config['logging']['logging_steps'])
    seed = int(config['training']['seed'])

    # wandbの使用設定を取得
    use_wandb = config['logging'].get('use_wandb', False)

    # report_toの設定
    if use_wandb:
        report_to = ["wandb"]
    else:
        report_to = ["none"]

    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=100,
        save_total_limit=2,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        logging_dir=config['logging']['logging_dir'],
        seed=seed,
        optim=config['training']['optim'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        dataloader_pin_memory=False,
        torch_compile=False,
        ddp_find_unused_parameters=False if config['training']['ddp'] else None,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)) if config['training']['ddp'] else -1,
        report_to=report_to
    )

    # トレーナーの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # トレーニングの開始
    logger.info("トレーニングを開始します")
    trainer.train()

    # ファインチューニング済みモデルの保存
    logger.info("ファインチューニング済みモデルを保存しています")
    trainer.save_model(config['training']['output_dir'])
