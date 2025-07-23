import torch
import random
from torch.utils.data import DataLoader, Sampler, DataLoader, ConcatDataset
from datasets import Dataset
from functools import partial
from transformers import Trainer
from data_load import *
from itertools import combinations

class MultiDatasetSampler(Sampler):
    def __init__(self, datasets, batch_size, ratios):
        self.datasets = datasets
        self.batch_size = batch_size
        self.ratios = ratios
        self.sizes = [len(dataset) for dataset in datasets]
        self.total_size = sum(self.sizes)
        self.num_batches = self.total_size // self.batch_size

        # 计算每个数据集的索引偏移量
        self.offsets = [0] * len(datasets)
        for i in range(1, len(datasets)):
            self.offsets[i] = self.offsets[i - 1] + self.sizes[i - 1]

        # 预先生成所有批次的索引
        self.batch_indices = self._generate_batch_indices()
    def _generate_batch_indices(self):
        batch_indices = []
        for _ in range(self.num_batches):
            indices = []
            for i, dataset in enumerate(self.datasets):
                # 根据比例采样
                n_samples = int(self.batch_size * self.ratios[i])
                dataset_indices = random.sample(range(self.sizes[i]), n_samples)
                # 将数据集索引转换为全局索引
                global_indices = [idx + self.offsets[i] for idx in dataset_indices]
                indices.extend(global_indices)
            batch_indices.extend(indices)
        return batch_indices

    def __iter__(self):
        return iter(self.batch_indices)

    def __len__(self):
        return self.num_batches * self.batch_size

def custom_collate_fn(batch):
    # ... existing code ...
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset  # 封装 Hugging Face 的 Dataset 对象

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),  # 确保转换为张量
            'attention_mask': torch.tensor(item['attention_mask']),  # 确保转换为张量
            'labels': torch.tensor(item['labels'])  # 确保转换为张量
        }


class CustomTrainer_data(Trainer):
    def __init__(self, train_dataloader=None, eval_dataloader=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

    def get_train_dataloader(self):
        if self.train_dataloader is not None:
            return self.train_dataloader
        else:
            return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if self.eval_dataloader is not None:
            return self.eval_dataloader
        else:
            return super().get_eval_dataloader(eval_dataset)

class MMD_Trainer(Trainer):
    def __init__(self, num_tasks, use_mmd, grad_step, train_dataloader=None, lamda=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks=num_tasks
        self.data_loader = train_dataloader
        self.a_outputs_dict = {}  # 移动到类属性中
        self.use_mmd = use_mmd
        self.lamda = lamda
        self.grad_step = grad_step
        self.global_step = 0
        if self.use_mmd:
            self.sigmas = torch.tensor(
                [
                    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
                    1e3, 1e4, 1e5, 1e6
                ],
                dtype=torch.bfloat16,
                device=self.model.device
            )
            self.beta = 1. / (2 * self.sigmas.view(-1, 1, 1, 1))
            self._register_lora_hooks()  # 初始化时注册钩子
            
   
    def hook_fc(self, module, input, output):
        parent_name = module._parent_name
        if parent_name not in self.a_outputs_dict:
            self.a_outputs_dict[parent_name] = output
        else:
            self.a_outputs_dict[parent_name] = torch.cat(
            [self.a_outputs_dict[parent_name], output], 
            dim=0
        )
        
    def _register_lora_hooks(self):
        L_lora = len('.lora_A.default')
        for name, layer in self.model.named_modules():
            if 'lora_A.default' in name:
                parent_name = name[:-L_lora]
                layer._parent_name = parent_name
                layer.register_forward_hook(self.hook_fc)
                
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        inputs['labels'] = inputs['labels'].squeeze(0)

        
        outputs = model(**inputs)
        # for k,v in self.a_outputs_dict.items():
        #     print(v.shape)
        #     break
        # print(self.a_outputs_dict)
        total_loss = outputs.loss
        self.global_step += 1
        if self.use_mmd and self.global_step== self.grad_step:
            rep_loss = self.compute_representation_loss()
            total_loss += self.lamda * rep_loss     
            self.a_outputs_dict.clear()
            self.global_step = 0
            self.log({"Rep_loss": rep_loss.item(), "Total_loss": total_loss.item()})
            
            
        # print('logit loss: ', logits_loss)
        # print('rep loss: ', representation_loss)
        # total_loss = logits_loss + 0.01 * rep_loss
        # self.log({"Logit_loss": logits_loss.item(), "Rep_loss": rep_loss.item(), "Total_loss": total_loss.item()})
        return (total_loss, outputs) if return_outputs else total_loss
    
    def pairwise_distance_3d(self, x, y):
        """计算三维张量之间的成对距离"""
        # input: x (Nx, S, r), y (Ny, S, r)
        # output: (Nx, Ny, S)
        return torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=-1)

    def gaussian_kernel_matrix(self, x, y):
        """计算三维张量的高斯核矩阵"""
        # input: x (Nx, S, r), y (Ny, S, r)
        # output: (Nx, Ny, S)
        dist = self.pairwise_distance_3d(x, y)
        expanded_dist = dist.unsqueeze(0)  # 形状变为 [1, Nx, Ny, S]
        # beta = 1. / (2 * self.sigmas.view(-1, 1, 1, 1))  # [Sigs, 1, 1, 1]
        # s = -beta * expanded_dist  # [Sigs, Nx, Ny, S]
        s = -self.beta * expanded_dist
        kernel_matrix = torch.exp(s).sum(0)  # [Nx, Ny, S]
        return kernel_matrix

    def maximum_mean_discrepancy(self, x, y):
        kernel = partial(self.gaussian_kernel_matrix)
        xx = kernel(x, x).mean()
        yy = kernel(y, y).mean()
        xy = kernel(x, y).mean()
        return xx + yy - 2 * xy

    def mmd_loss(self, source_features, target_features):
        assert len(source_features.shape) == 3 and len(target_features.shape) == 3, \
            f"Input must be 3D tensors (got {source_features.shape} and {target_features.shape})"
        return self.maximum_mean_discrepancy(source_features, target_features)
        
    def compute_representation_loss(self):
        total_loss = torch.tensor(0.0, device=self.model.device)
        num_layers = 0
        # for k,v in self.a_outputs_dict.items():
        #     print(k, v.shape)
            # break
        for layer_name, features in self.a_outputs_dict.items():
            steps = self.grad_step
            step_chunks = torch.split(features, steps, dim=0)  

           
            groups = [torch.cat([torch.chunk(step, self.num_tasks, dim=0)[task_idx] for step in step_chunks], dim=0)
                    for task_idx in range(self.num_tasks)]
            # print(groups[0].shape)
            # groups = torch.chunk(features, self.num_tasks, dim=0)
            layer_total = torch.tensor(0.0, device=self.model.device)
            count = 0
            for group1, group2 in combinations(groups, 2):
                layer_total += self.mmd_loss(group1, group2)
                count += 1
            if count > 0:
                total_loss += layer_total / count
                num_layers += 1
        return total_loss / num_layers    


class KL_Trainer(Trainer):
    def __init__(self, num_tasks, use_kl, grad_step, train_dataloader=None, lamda=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.data_loader = train_dataloader
        self.a_outputs_dict = {}  
        self.use_kl = use_kl      
        self.lamda = lamda        
        self.grad_step = grad_step 
        self.global_step = 0
        self.kl_loss_values = []  
        self.original_loss_values = []  
        self.total_loss_values = []  
        if self.use_kl:
            self._register_lora_hooks()

    # hook_fc 和 _register_lora_hooks 保持不变
    def hook_fc(self, module, input, output):
        """
        get the output of LoRA_A layer and store it in a_outputs_dict.
        """
        parent_name = module._parent_name
        if parent_name not in self.a_outputs_dict:
            self.a_outputs_dict[parent_name] = output
        else:
            self.a_outputs_dict[parent_name] = torch.cat(
                [self.a_outputs_dict[parent_name], output],
                dim=0
            )

    def _register_lora_hooks(self):
        """
        register hook for LoRA_A layer.
        """
        L_lora = len('.lora_A.default')
        for name, layer in self.model.named_modules():
            if 'lora_A.default' in name:
                parent_name = name[:-L_lora]
                layer._parent_name = parent_name
                layer.register_forward_hook(self.hook_fc)

    # ------------------- KL散度核心实现部分 (新代码) -------------------

    def _kl_divergence_diag_gaussian(self, mu1, var1, mu2, var2, epsilon=1e-8):
        """
        compute the diagonal KL divergence between two Gaussian distributions.
        D_KL(N(mu1, var1) || N(mu2, var2))
        """
        # add epsilon to avoid division by zero
        var1_eps = var1 + epsilon
        var2_eps = var2 + epsilon

        log_var_ratio = torch.log(var2_eps) - torch.log(var1_eps)
        trace_term = var1_eps / var2_eps
        mu_diff_term = (mu2 - mu1)**2 / var2_eps

        # sum up the components of the KL divergence
        kl_div = 0.5 * (log_var_ratio + trace_term + mu_diff_term - 1)
        
        # sum over the feature dimension and average over the sample/sequence dimension
        return torch.mean(torch.sum(kl_div, dim=-1))

    def kl_loss(self, source_features, target_features):
        """
        compute the symmetric KL divergence loss between source and target features.
        - source_features: (N_source, S, r)
        - target_features: (N_target, S, r)
        """
        assert len(source_features.shape) == 3 and len(target_features.shape) == 3, \
            f"输入必须是三维张量 (got {source_features.shape} and {target_features.shape})"

       
        # compute the mean and variance along the batch dimension (dim=0)
        mu_source = torch.mean(source_features, dim=0)
        var_source = torch.var(source_features, dim=0)
        
        mu_target = torch.mean(target_features, dim=0)
        var_target = torch.var(target_features, dim=0)
        
        # compute the KL divergence
        kl_source_target = self._kl_divergence_diag_gaussian(mu_source, var_source, mu_target, var_target)
        kl_target_source = self._kl_divergence_diag_gaussian(mu_target, var_target, mu_source, var_source)
        
        symmetric_kl_loss = (kl_source_target + kl_target_source) / 2.0
        return symmetric_kl_loss



    def compute_representation_loss(self):
        """
        计算所有LoRA层、所有任务对之间的平均KL散度损失。
        """
        total_loss = torch.tensor(0.0, device=self.model.device)
        num_layers = 0
        for layer_name, features in self.a_outputs_dict.items():
            steps = self.grad_step
            step_chunks = torch.split(features, steps, dim=0)

            groups = [torch.cat([torch.chunk(step, self.num_tasks, dim=0)[task_idx] for step in step_chunks], dim=0)
                      for task_idx in range(self.num_tasks)]

            layer_total = torch.tensor(0.0, device=self.model.device)
            count = 0
            # compute pairwise KL divergence for each pair of groups
            for group1, group2 in combinations(groups, 2):
                layer_total += self.kl_loss(group1, group2)
                count += 1
                
            if count > 0:
                total_loss += layer_total / count
                num_layers += 1
                
        return total_loss / num_layers if num_layers > 0 else torch.tensor(0.0, device=self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        inputs['labels'] = inputs['labels'].squeeze(0)
        
        outputs = model(**inputs)
        total_loss = outputs.loss
        
        self.global_step += 1
        

        if self.use_kl and self.global_step == self.grad_step:
            rep_loss = self.compute_representation_loss()
            
            self.original_loss_values.append(total_loss.item())
            self.kl_loss_values.append(rep_loss.item())
            
            total_loss += self.lamda * rep_loss
            
            self.total_loss_values.append(total_loss.item())
            # 清理和日志记录
            self.a_outputs_dict.clear() # 清空本次累积的特征
            self.global_step = 0      # 重置步数计数器
            self.log({"KL_loss": rep_loss.item(), "Total_loss": total_loss.item()})
            
        return (total_loss, outputs) if return_outputs else total_loss


    def save_losses_to_file(self, output_dir="./loss_records"):
        import os
        import json
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为包含三种损失的字典，共享同一套索引（隐含步长）
        loss_data = {
            "original_loss": self.original_loss_values,
            "kl_loss": self.kl_loss_values,
            "total_loss": self.total_loss_values,
            "num_records": self.record_step  # 记录总次数
        }
        
        with open(os.path.join(output_dir, "losses.json"), "w") as f:
            json.dump(loss_data, f, indent=2)
        
        print(f"损失数据已保存到 {output_dir}/losses.json，共{self.record_step}条记录")