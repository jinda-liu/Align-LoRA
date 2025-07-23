import torch
import os
# os.environ["WANDB_DISABLED"] = "true"
import random
from functools import partial
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, get_dataset_config_names, concatenate_datasets
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from data_load import *
from custom import *

lamda = 0.2
model_base = AutoModelForCausalLM.from_pretrained('/path/to/Qwen2.5-7B',
                                              device_map="auto",
                                              torch_dtype=torch.bfloat16,
                                              output_hidden_states=True
                                              )
tokenizer = AutoTokenizer.from_pretrained('/path/to/Qwen2.5-7B',
                                          use_fast=False, trust_remote_code=True)

# print(model_base)
model_base.enable_input_require_grads() # 
# print(model_base.dtype)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["down_proj", "up_proj", "gate_proj"],
    inference_mode=False, 
    r=8, # rank
    lora_alpha=32, 
    lora_dropout=0.1# Dropout rate
)
model = get_peft_model(model_base, config)
model.print_trainable_parameters()

exp_name = f'7_lora_kl_mt5_flan_'
training_args = TrainingArguments(
    output_dir=f"./output/{exp_name}",
    logging_dir="./logs/{exp_name}",
    per_device_train_batch_size=5,
    gradient_accumulation_steps=10,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=100,
    save_total_limit=1,
    learning_rate=2e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    # remove_unused_columns=False,
    # report_to="wandb",
    run_name=exp_name,
)

# save_tokenized_dir = f"/root/autodl-tmp/data/flanv2_tokenized/"

# flanv2_train_ds = load_from_disk(f"{save_tokenized_dir}")
# train_datasets = flanv2_train_ds

train_ds = {}

# please load your datasets here and organize them into a dictionary

for name, dataset in train_ds.items():
    print(name)
    print(dataset)
    # print(tokenizer.decode(dataset[0]['input_ids'], skip_special_tokens=True))

train_datasets = [dataset for dataset in train_ds.values()]
train_datasets = concatenate_datasets(train_datasets)
# concat_dataset = train_datasets.shuffle()
# --------------------------------------------------------------------------------
# 
train_datasets = [CustomDataset(dataset) for dataset in train_ds.values()]
# train_datasets = [dataset for dataset in train_ds.values()]

# sample ratios
ratios = [1.0 / len(train_datasets)] * len(train_datasets)
print(ratios)

#  Sampler
sampler = MultiDatasetSampler(train_datasets, training_args.per_device_train_batch_size, ratios)
print(sampler)
print(len(sampler))


concat_dataset = ConcatDataset(train_datasets)
print(concat_dataset)
print(len(concat_dataset))


train_dataloader = DataLoader(
    concat_dataset,
    batch_size=training_args.per_device_train_batch_size,
    sampler=sampler,
    collate_fn=custom_collate_fn,
    drop_last=True
)

# # 验证实现
# print(len(train_dataloader))

for batch in train_dataloader:
    print(batch.keys())
    print(batch['input_ids'].shape)
    print(batch['attention_mask'].shape)
    print(batch['labels'].shape)

    # decode input_ids to text
    input_texts = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
    # label_texts = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
    print("decoded input:")
    for i, text in enumerate(input_texts):
        print(f"{i + 1}: {text}")

    # print(batch['labels'].tolist())
    break  # check the first batch

# --------------------------------------------------------------------------------

# custom Trainer
trainer = KL_Trainer(
    model=model,
    args=training_args,
    train_dataset=concat_dataset,
    num_tasks=5, 
    use_kl=True,
    lamda=lamda,
    grad_step=training_args.gradient_accumulation_steps
)


print(trainer)
print(lamda)
trainer.train()