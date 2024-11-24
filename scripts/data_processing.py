from datasets import load_dataset
import json

def get_default_chat_template():
    """デフォルトのチャットテンプレート"""
    return """<bos><start_of_turn>user
{%- for message in messages %}
{%- if message['role'] == 'user' %}
{{message['content']}}{%- elif message['role'] == 'assistant' %}
<end_of_turn>
<start_of_turn>model
{{message['content']}}{%- elif message['role'] == 'system' %}
{{message['content']}}{%- endif %}
{%- endfor %}
<end_of_turn>
<start_of_turn>model"""

def load_and_prepare_datasets(config, tokenizer, logger):
    load_from = config['dataset'].get('load_from', 'huggingface')
    cache_dir = config['dataset']['cache_dir']
    data_files = config['dataset'].get('data_files', None)
    dataset_name = config['dataset'].get('name', None)

    logger.info(f"データセットを読み込んでいます (source: {load_from})")

    if load_from == 'huggingface':
        if dataset_name is None:
            logger.error("Hugging Face データセット名が指定されていません")
            raise ValueError("Hugging Face データセット名が必要です")
        datasets = load_dataset(
            dataset_name,
            cache_dir=cache_dir
        )
        if 'train' in datasets:
            train_dataset = datasets['train']
        else:
            train_dataset = datasets
    elif load_from == 'local':
        if data_files is None or 'train' not in data_files:
            logger.error("ローカルデータファイルが指定されていません")
            raise ValueError("ローカルデータファイルが必要です")
        datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=cache_dir,
            split='train'
        )
        train_dataset = datasets
    else:
        logger.error(f"未知のデータセットソース: {load_from}")
        raise ValueError(f"未知のデータセットソース: {load_from}")

    # 必要なカラムが存在するか確認
    if not all(col in train_dataset.column_names for col in ["instruction","input", 'output']):
        logger.error("データセットに必要なカラム instruction,'input' または 'output' が見つかりません。")
        raise ValueError("データセットに必要なカラムがありません。")

    block_size = config['training']['block_size']

    use_custom_prompt = config['prompts'].get('use_custom_prompt', False)
    default_system_prompt = config['prompts'].get('default_system_prompt', '')
    user_prompt = config['prompts'].get('user_prompt', '')
    use_chat_template = config['prompts'].get('use_chat_template', False)

    # チャットテンプレートの設定
    if use_chat_template:
        # モデル固有のチャットテンプレートを確認
        has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
        if not has_chat_template:
            logger.info("デフォルトのチャットテンプレートを設定します")
            tokenizer.chat_template = get_default_chat_template()

    logger.info("データセットをトークナイズしています")

    def generate_prompt(examples):
        instructions = examples['instruction']
        inputs = examples['input']
        outputs = examples['output']

        prompts = []
        for ins, inp, out in zip(instructions, inputs, outputs):
            # プロンプトのフォーマット
            messages = []
            if use_custom_prompt and default_system_prompt:
                messages.append({'role': 'system', 'content': default_system_prompt})
            
            # ユーザーメッセージの作成
            if use_custom_prompt:
                if inp is None or inp == "":
                    user_content = f"###指示\n{ins}"
                else:
                    user_content = f"###指示\n{ins}\n\n###入力\n{inp}"
            else:
                user_content = str(ins) if inp is None or inp == "" else f"{ins}\n{inp}"
            
            messages.append({'role': 'user', 'content': user_content})
            messages.append({'role': 'assistant', 'content': str(out)})

            if use_chat_template:
                try:
                    formatted_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    logger.warning(f"チャットテンプレートの適用に失敗しました: {e}")
                    # フォールバック: 基本フォーマットを使用
                    formatted_text = ""
                    for msg in messages:
                        role = msg['role']
                        content = msg['content']
                        if role == 'system':
                            formatted_text += f"<s>[システム]\n{content}\n"
                        elif role == 'user':
                            formatted_text += f"[ユーザー]\n{content}\n"
                        elif role == 'assistant':
                            formatted_text += f"[アシスタント]\n{content}</s>\n"
            else:
                # デフォルトのフォーマット
                formatted_text = ""
                for msg in messages:
                    role = msg['role']
                    content = msg['content']
                    if role == 'system':
                        formatted_text += f"<s>[システム]\n{content}\n"
                    elif role == 'user':
                        formatted_text += f"[ユーザー]\n{content}\n"
                    elif role == 'assistant':
                        formatted_text += f"[アシスタント]\n{content}</s>\n"
                    
            prompts.append(formatted_text)
        return {'text': prompts}

    # プロンプトを生成
    train_dataset = train_dataset.map(
        generate_prompt,
        batched=True,
        num_proc=4,
        desc="Generating prompts"
    )

    # 不要なカラムを削除
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col != 'text'])

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            max_length=block_size,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        input_ids = tokenized['input_ids']
        labels = input_ids.clone()

        # アシスタントの返信部分を特定
        assistant_markers = [
            "<start_of_turn>model",  # Gemma形式
            "<start_of_turn>assistant",  # デフォルトテンプレート
            "[アシスタント]",  # 基本フォーマット
        ]

        for i, text in enumerate(examples['text']):
            assistant_start = -1
            for marker in assistant_markers:
                pos = text.rfind(marker)
                if pos != -1:
                    assistant_start = pos
                    break

            if assistant_start != -1:
                assistant_tokens = tokenizer.encode(text[assistant_start:], add_special_tokens=False)
                labels[i, :-len(assistant_tokens)] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': tokenized['attention_mask'],
            'labels': labels,
        }

    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        desc="Tokenizing dataset"
    )

    return tokenized_dataset