import os
import json
from datasets import Dataset
from typing import Optional
import pickle as pkl
from dataclasses import dataclass, field, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from knowledge_propagation.utils import vars, io


def prepare_sft_text(args, dataset: list, tokenizer):
    # assert len(tokenizer.additional_special_tokens) == 1

    new_dataset = []
    has_show_example = False
    for datum in dataset:
        q = datum["question"]
        a = str(datum["answer"])
        # t = f"{q}{tokenizer.additional_special_tokens[0]}{a}" if a[0] == " " else f"{q}{tokenizer.additional_special_tokens[0]} {a}"
        t = f"{q}{a}" if a[0] == " " else f"{q} {a}"
        t = (tokenizer.bos_token or "") + t + tokenizer.eos_token
        if not has_show_example:
            print(f"Example: -> {t}")
            has_show_example = True
        datum[args.dataset_text_field] = t
        new_dataset.append(datum)
    return new_dataset


parser = HfArgumentParser((SFTConfig,))
(args,) = parser.parse_args_into_dataclasses()
model_name_or_path = f"{os.environ['SHARE_RES_DIR']}/models/qwen/Qwen3-1.7B-Base"
# model_name_or_path = "/u/zliu/datastor1/mend/models/Llama-3.2-1B-eos-sft"


model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_eos_token=True, use_fast=False)
tokenizer.padding_side = "right"
original_vocab_size = len(tokenizer)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
# tokenizer.add_special_tokens({'additional_special_tokens': ['<sft_token_1>']}, replace_additional_special_tokens=False)
model.resize_token_embeddings(len(tokenizer))

tokenizer.sep_token = tokenizer.cls_token = tokenizer.mask_token = tokenizer.pad_token
model.config.pad_token_id = tokenizer.pad_token_id

assert tokenizer.eos_token != tokenizer.pad_token
assert tokenizer.eos_token_id != tokenizer.pad_token_id

train_dataset = prepare_sft_text(args, io.load_jsonlines(f"{vars.DATA_DIR}/trivia_qa_wiki_sft/train.jsonl"), tokenizer)
valid_dataset = prepare_sft_text(args, io.load_jsonlines(f"{vars.DATA_DIR}/trivia_qa_wiki_sft/valid.jsonl"), tokenizer)

# train_dataset = prepare_sft_text(
#     args, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/common_date_data/train.jsonl"), tokenizer
# )
# valid_dataset = prepare_sft_text(
#     args, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/common_date_data/valid.jsonl"), tokenizer
# )

# train_dataset = prepare_sft_text(
#     args, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/syn_data_neurips/model_prep/light_weight_sft_content_curated_v1_sample=10.jsonl"), tokenizer
# )
# train_dataset = prepare_sft_text(
#     args, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/common_country_data/train.jsonl"), tokenizer
# )
# valid_dataset = prepare_sft_text(
#     args, io.load_jsonlines(f"{vars.DATA_DIR}/debug_meta_train/common_country_data/valid.jsonl"), tokenizer
# )
# import pdb; pdb.set_trace()
train_dataset = Dataset.from_list(train_dataset)
# valid_dataset = Dataset.from_list(valid_dataset)

response_template = "?"  # tokenizer.additional_special_tokens[0] # "?" alternative "Ġ?"

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,  # type: ignore
    # eval_dataset=valid_dataset,  # type: ignore
    args=args,
    data_collator=collator,
)

trainer.train()

trainer.model.config.pad_token_id = None
trainer.model.resize_token_embeddings(original_vocab_size)
trainer.model.save_pretrained(save_directory=args.output_dir)

trainer.accelerator.wait_for_everyone()

