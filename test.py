import torch
import random
import sys
import os
from datasets import Dataset, load_dataset, get_dataset_config_names, concatenate_datasets, interleave_datasets
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from data_load import *
from test_fc import *
from collections import Counter
from torch.utils.data import DataLoader

# torch_dtype = torch.float32
torch_dtype = torch.bfloat16
base_model = AutoModelForCausalLM.from_pretrained('path/to/model',
                                              device_map="auto",
                                              torch_dtype=torch_dtype,
                                              output_hidden_states=True
                                              )

# for name, param in model.named_parameters():
#     print(f"{name}: {param.dtype}")

tokenizer = AutoTokenizer.from_pretrained('path/to/model', 
                                          use_fast=False, trust_remote_code=True,
                                          padding_side='left')

test_ds = {}
# load test datasets and organize them to test_ds

# # 检查所有数据集的标签分布
for name, dataset in test_ds.items():
    print(name)
    print(dataset)

lora_path = f'path/to/your/lora_model'  # Replace with your LoRA model path
model = PeftModel.from_pretrained(base_model, model_id=lora_path)
experiment_params = os.path.basename(os.path.dirname(lora_path))
# experiment_params = experiment_params.split('_seed')[0]
filename = f"result/{experiment_params}.txt"


device = model.device
# model.to(device)
model.eval()
# print(model)
print('Model loaded')


batch_size = 64
batch_test_mt(model, test_ds, tokenizer, device, batch_size, filename)
# test_math(model, test_ds, tokenizer, device, batch_size, filename)
