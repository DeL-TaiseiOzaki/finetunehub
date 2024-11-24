import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_and_prepare_model(config, device, logger):
    model_name_or_path = config['model']['name_or_path']
    tokenizer_name = config['model']['tokenizer_name'] or model_name_or_path
    cache_dir = config['model']['cache_dir']
    use_fp16 = config['training']['fp16']
    use_bf16 = config['training']['bf16']
    gradient_checkpointing = config['training']['gradient_checkpointing']

    logger.info(f"{tokenizer_name} のトークナイザを読み込んでいます")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_lora = config['lora']['use_lora']
    use_qlora = config['lora'].get('use_qlora', False)

    logger.info(f"{model_name_or_path} のモデルを読み込んでいます")
    if use_qlora:
        logger.info("QLoRA設定でモデルを読み込んでいます")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map='auto',
            quantization_config=bnb_config,
            cache_dir=cache_dir,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map='auto' if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if use_fp16 else torch.bfloat16 if use_bf16 else torch.float32,
            cache_dir=cache_dir,
        )

    # トークン埋め込みのサイズを調整
    if tokenizer.vocab_size > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # LoRAまたはフルファインチューニングの設定
    if use_lora or use_qlora:
        logger.info("LoRAの設定を行っています")
        if gradient_checkpointing:
            model.enable_input_require_grads()
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
        lora_config = LoraConfig(
            r=config['lora']['r'],
            lora_alpha=config['lora']['alpha'],
            target_modules=config['lora']['target_modules'],
            lora_dropout=config['lora']['dropout'],
            bias='none',
            task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.to(device)

    return model, tokenizer
