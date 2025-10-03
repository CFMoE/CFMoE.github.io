import pandas as pd
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import sys, os, json
from tqdm import tqdm
import torch

def prepare_example(input_text, output_text, tokenizer, max_words_len):
    tokenized_input = tokenizer(input_text, add_special_tokens=False)
    tokenized_output = tokenizer(output_text, add_special_tokens=False)

    input_ids = torch.tensor(tokenized_input['input_ids'] + tokenized_output['input_ids'] + [tokenizer.eos_token_id])
    attention_mask = torch.tensor(tokenized_input['attention_mask'] + tokenized_output['attention_mask'] + [1])
    labels = torch.tensor([-100] * len(tokenized_input['input_ids']) + tokenized_output['input_ids'] + [tokenizer.eos_token_id])

    if len(input_ids) > max_words_len:
        input_ids = input_ids[:max_words_len]
        attention_mask = attention_mask[:max_words_len]
        labels = labels[:max_words_len]
    else:
        padding_length = max_words_len - len(input_ids)
        input_ids = torch.cat([input_ids, torch.full((padding_length,), tokenizer.pad_token_id, dtype=torch.long)])
        attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
        labels = torch.cat([labels, torch.full((padding_length,), -100, dtype=torch.long)])

    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }

def process_alpaca_example(example, prompt_dict, tokenizer, max_words_len):
    if example['input'] == "":
        input_text = prompt_dict["prompt_wo_input"].format_map(example)
    else:
        input_text = prompt_dict["prompt_with_input"].format_map(example)
    output_text = example["output"]
    
    return prepare_example(input_text, output_text, tokenizer, max_words_len)

def process_aqua_example(example, prompt_dict, tokenizer, max_words_len):
    input_text = prompt_dict["prompt"].format_map(example)
    output_text = f"{example['rationale']}\n#### {example['answer']}"

    return prepare_example(input_text, output_text, tokenizer, max_words_len)

def process_boolq_example(example, prompt_dict, tokenizer, max_words_len):
    input_text = prompt_dict["prompt"].format_map(example)
    output_text = example['answer']

    return prepare_example(input_text, output_text, tokenizer, max_words_len)

def process_gsm8k_example(example, prompt_dict, tokenizer, max_words_len):
    input_text = prompt_dict["prompt"].format_map(example)
    output_text = example['answer']

    return prepare_example(input_text, output_text, tokenizer, max_words_len)

def process_mathqa_example(example, prompt_dict, tokenizer, max_words_len):
    input_text = prompt_dict["prompt"].format_map(example)
    output_text = f"{example['rationale']}\n#### {example['answer']}"

    return prepare_example(input_text, output_text, tokenizer, max_words_len)

def process_mawps_example(example, prompt_dict, tokenizer, max_words_len):
    input_text = prompt_dict["prompt"].format_map(example)
    output_text = f"Equation is {example['equation'].lower()}.\nThe solution is x={example['ans']}.\n#### {example['ans']}"

    return prepare_example(input_text, output_text, tokenizer, max_words_len)

def process_mmlu_example(example, prompt_dict, tokenizer, max_words_len):
    input_text = prompt_dict["prompt"].format_map(example)
    output_text = str(example["answer"])
    subject = example["subject"]

    return prepare_example(input_text, output_text, tokenizer, max_words_len)

def process_obqa_example(example, prompt_dict, tokenizer, max_words_len):
    input_text = prompt_dict["prompt"].format_map(example)
    output_text = example["answer"]

    return prepare_example(input_text, output_text, tokenizer, max_words_len)

def process_piqa_example(example, prompt_dict, tokenizer, max_words_len):
    input_text = prompt_dict["prompt"].format_map(example)
    output_text = str(example["answer"])

    return prepare_example(input_text, output_text, tokenizer, max_words_len)

def process_samsum_example(example, prompt_dict, tokenizer, max_words_len):
    input_text = prompt_dict["prompt"].format_map(example)
    output_text = str(example["summary"])

    return prepare_example(input_text, output_text, tokenizer, max_words_len)

def process_siqa_example(example, prompt_dict, tokenizer, max_words_len):
    input_text = prompt_dict["prompt"].format_map(example)
    output_text = str(example["answer"])

    return prepare_example(input_text, output_text, tokenizer, max_words_len)

def process_wikitext_example(example, prompt_dict, tokenizer, max_words_len):
    input_text = prompt_dict["prompt"].format_map(example)
    output_text = example["content"]

    return prepare_example(input_text, output_text, tokenizer, max_words_len)

def process_common_v1_example(example, prompt_dict, tokenizer, max_words_len):
    if example.get("input"):
        prompt = "prompt_with_input"
    else:
        prompt = "prompt_wo_input"
    input_text = prompt_dict[prompt].format_map(example)
    output_text = example["output"]

    return prepare_example(input_text, output_text, tokenizer, max_words_len)

def process_common_v2_example(example, prompt_dict, tokenizer, max_words_len):
    input_text = prompt_dict["prompt"].format_map(example)
    output_text = example["output"]

    return prepare_example(input_text, output_text, tokenizer, max_words_len)

def process_csl_example(example, prompt_dict, tokenizer, max_words_len):
    for k in example:
        example[k] = example[k].strip('\n')
    input_text = prompt_dict["prompt"].format_map(example)
    output_text = example["output"].strip('\n')

    return prepare_example(input_text, output_text, tokenizer, max_words_len)

def process_belle_example(example, prompt_dict, tokenizer, max_words_len):
    for k in example:
        example[k] = example[k].strip('\n')
    input_text = prompt_dict["prompt"].format_map(example)
    output_text = example["output"].strip('\n')

    return prepare_example(input_text, output_text, tokenizer, max_words_len)

process_handlers = {
    'alpaca':   process_alpaca_example,
    'aqua':     process_aqua_example,
    'boolq':    process_boolq_example,
    'gsm8k':    process_gsm8k_example,
    'mathqa':   process_mathqa_example,
    'mawps':    process_mawps_example,
    'mmlu':     process_mmlu_example,
    'obqa':     process_obqa_example,
    'piqa':     process_piqa_example,
    'samsum':   process_samsum_example,
    'siqa':     process_siqa_example,
    'wikitext': process_wikitext_example,
    'common-v1': process_common_v1_example,
    'common-v2': process_common_v2_example,
    'csl': process_csl_example,
    'belle': process_belle_example,
}

class DataPreparer:
    """Data preparation class for CFMoE framework."""
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def load_data(self, dataset_path, dataset_name, dataset_type):
        """Load and tokenize JSON dataset."""
        dataset_name = dataset_name.lower()
        dataset_path = os.path.join(dataset_path, dataset_name)

        prompt_dict = json.load(open(os.path.join(dataset_path, "prompt.json")))

        if dataset_type == "train":
            data_file = os.path.join(dataset_path, "train.json")
        elif dataset_type == "eval":
            data_file = os.path.join(dataset_path, "test.json")
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        print(f"Loading dataset == {dataset_name} == ....")
        dataset = load_dataset("json", data_files=data_file)["train"]

        if dataset_name in process_handlers:
            handler = process_handlers[dataset_name]
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        tokenized_dataset = dataset.map(
            lambda example: handler(example, prompt_dict, self.tokenizer, self.config.max_words_len),
            remove_columns=dataset.column_names
        )

        return tokenized_dataset

    def get_dataloader(self, dataset, shuffle=True):
        """Create data loader."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=2
        )
