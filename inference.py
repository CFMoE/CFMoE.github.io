#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer parallelism warnings

from config_i import Config
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, AutoConfig, GenerationConfig
from peft import PeftModelForCausalLM
import torch
import torch.distributed as dist
import json
import os
import torch.nn.functional as F
import argparse
from utils import get_arguments
import csv
import random
import pandas as pd
from collections import deque, defaultdict
from logger import setup_distributed_inference_logging
import time
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from dataclasses import dataclass
import math
from tqdm import tqdm
import numpy as np
import evaluate
from rouge import Rouge 

@dataclass
class GenMetric:
    got:str = None
    expected:str = None
    is_correct:int = 0
    ppl:float = None
    rouge1:float = -1.0
    rouge2:float = -1.0
    rougeL:float = -1.0
    # rougeLsum:float = -1.0

class InferenceEngine:
    def __init__(self, config, rank, device):
        self.config = config
        self.rank = rank
        self.device = device

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True, padding_side="left")

        model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
        model_config.use_cache = True

        if config.use_quant:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                config=model_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
                device_map={"": self.rank}
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                config=model_config,
                trust_remote_code=True,
                device_map={"": self.rank}
            )

        if config.use_quant:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                config=model_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
                device_map={"": self.rank}
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                config=model_config,
                trust_remote_code=True,
                device_map={"": self.rank}
            )
        if self.config.use_lora:
            self.model = PeftModelForCausalLM.from_pretrained(model, config.lora_save_path)
        else:
            self.model = model

        for model in [self.base_model, self.model]:
            model.generation_config.do_sample = config.do_sample
            model.generation_config.temperature = config.temperature
            model.generation_config.top_p = config.top_p
            model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            model.generation_config.repetition_penalty = config.repetition_penalty
            try:
                model.generation_config.validate()
            except ValueError as e:
                print(f"Generation config validation failed: {e}")

    def get_decoder_layers(self, model):
        if isinstance(model, PeftModelForCausalLM):
            return model.base_model.model.model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        else:
            raise ValueError("Unsupported model type, cannot access decoder layers")

    def generate(self, input_texts, use_lora, return_locality_metrics=True):
        model = self.model if use_lora else self.base_model
        model.eval()
        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=self.config.max_words_len).to(self.device)
        with torch.no_grad():
            gconf = GenerationConfig()
            gconf.output_logits = True
            gconf.return_dict_in_generate = True
            outputs = model.generate(**inputs,
                                     generation_config = gconf,
                                     max_new_tokens=self.config.max_words_len,
                                     do_sample=self.config.do_sample,
                                     temperature=self.config.temperature)
        generated_texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        if not return_locality_metrics:
            return generated_texts, None, None

        layers = self.get_decoder_layers(model)
        all_expert_activations = []
        all_gate_scores = []
        moe_module_name = config.moe_module_name
        for layer in layers:
            try:
                moe_module = getattr(layer, moe_module_name)
            except ValueError as e:
                print(f"The attribute name of moe module is incorrect, please check the config file: {e}")
                
            if hasattr(moe_module, 'expert_activations'):
                result_expert_tensor = torch.cat(moe_module.expert_activations, dim=1).to(self.device)
                result_gate_tensor = torch.cat(moe_module.gate_scores, dim=1).to(self.device)
                all_expert_activations.append(result_expert_tensor.cpu())
                all_gate_scores.append(result_gate_tensor.cpu())
        for layer in layers:
            try:
                moe_module = getattr(layer, moe_module_name)
            except ValueError as e:
                print(f"The attribute name of moe module is incorrect, please check the config file: {e}")
                
            if hasattr(moe_module, 'reset_metrics'):
                moe_module.reset_metrics()
        # shape of all_expert_activations: [tensor(bsz, seq_len, topk)], len(all_expert_activations)=layer_num
        # shape of all_gate_scores: [tensor(bsz, seq_len, exp_num)], len(all_gate_scores)=layer_num
        return generated_texts, all_expert_activations, all_gate_scores, outputs.logits
    
    def get_moe_metainfo(self,):
        # return the number of moe layers and number of experts in each layer.
        model = self.base_model
        layers = self.get_decoder_layers(model)
        moe_module_name = config.moe_module_name
        moe_n = len(layers)
        
        first_layer = layers[1] # the first layer of deepseek-moe-16b dont have experts
        try:
            moe_module = getattr(first_layer, moe_module_name)
            experts_module = getattr(moe_module, "experts")
            experts_n = len(experts_module)
        except ValueError as e:
            print(f"The attribute name of moe module is incorrect, please check the config file: {e}")
        
        return moe_n, experts_n


# 新增：定义缓存策略的基类
class CachePolicy:
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.cache_hits = None

    def access(self, expert_id, current_position=None):
        raise NotImplementedError

# 实现最优缓存策略（Belady算法）
class OptimalCachePolicy(CachePolicy):
    def __init__(self, cache_size, access_sequence):
        super().__init__(cache_size)
        self.cache = set()
        self.cache_hits = 0
        self.future_positions = defaultdict(deque)
        # 预处理未来访问位置
        for pos, expert in enumerate(access_sequence):
            self.future_positions[expert].append(pos)
        self.access_sequence = access_sequence

    def access(self, current_position):
        expert_id = self.access_sequence[current_position]
        # 移除当前访问位置
        self.future_positions[expert_id].popleft()

        if expert_id in self.cache:
            self.cache_hits += 1
            return

        if len(self.cache) < self.cache_size:
            self.cache.add(expert_id)
            return

        # 找到缓存中下次使用最远的专家
        farthest = -1
        expert_to_remove = None
        for cached_expert in self.cache:
            if not self.future_positions[cached_expert]:
                # 如果缓存中的专家在未来不再被访问，优先移除
                expert_to_remove = cached_expert
                break
            next_use = self.future_positions[cached_expert][0]
            if next_use > farthest:
                farthest = next_use
                expert_to_remove = cached_expert

        if expert_to_remove is not None:
            self.cache.remove(expert_to_remove)
        self.cache.add(expert_id)

# 实现有限视野最优缓存策略（结合有限视野和LRU）
class LimitedOptimalCachePolicy(CachePolicy):
    def __init__(self, cache_size, access_sequence, lookahead=6*27*2): # iteration * n_moe * n_activated_experts
        super().__init__(cache_size)
        self.cache = set()
        self.cache_hits = 0
        self.lookahead = lookahead
        self.future_positions = defaultdict(deque)
        # 预处理未来访问位置
        for pos, expert in enumerate(access_sequence):
            self.future_positions[expert].append(pos)
        self.access_sequence = access_sequence
        # 使用 deque 来维护LRU顺序，左侧为最久未使用
        self.lru_order = deque()

    def access(self, current_position):
        expert_id = self.access_sequence[current_position]
        # 移除当前访问位置
        self.future_positions[expert_id].popleft()

        if expert_id in self.cache:
            self.cache_hits += 1
            # 更新LRU顺序
            if expert_id in self.lru_order:
                self.lru_order.remove(expert_id)
            self.lru_order.append(expert_id)
            return

        if len(self.cache) < self.cache_size:
            self.cache.add(expert_id)
            self.lru_order.append(expert_id)
            return

        # 定义窗口结束位置
        window_end = current_position + self.lookahead

        # 分为两类缓存中的专家：
        # 1. 下次使用在lookahead窗口内
        # 2. 下次使用在lookahead窗口外或不再被使用
        candidates_outside = []
        candidates_inside = []

        for cached_expert in self.cache:
            # 清理已过去的访问位置
            while self.future_positions[cached_expert] and self.future_positions[cached_expert][0] <= current_position:
                self.future_positions[cached_expert].popleft()
            if not self.future_positions[cached_expert]:
                # 不再被使用，优先考虑移除
                candidates_outside.append(cached_expert)
            else:
                next_use = self.future_positions[cached_expert][0]
                if next_use > window_end:
                    candidates_outside.append(cached_expert)
                else:
                    candidates_inside.append((cached_expert, next_use))

        if candidates_outside:
            # 如果有专家的下次使用在lookahead窗口外，按照LRU移除
            # 找到最久未使用的专家
            for expert in self.lru_order:
                if expert in candidates_outside:
                    expert_to_remove = expert
                    break
            self.cache.remove(expert_to_remove)
            self.lru_order.remove(expert_to_remove)
        else:
            # 如果所有专家的下次使用都在lookahead窗口内，移除下次使用最远的专家
            farthest = -1
            expert_to_remove = None
            for cached_expert, next_use in candidates_inside:
                if next_use > farthest:
                    farthest = next_use
                    expert_to_remove = cached_expert
            if expert_to_remove is not None:
                self.cache.remove(expert_to_remove)
                self.lru_order.remove(expert_to_remove)

        # 添加新的专家到缓存和LRU顺序
        self.cache.add(expert_id)
        self.lru_order.append(expert_id)

# 实现最近最少使用（LRU）缓存策略
class LRUCachePolicy(CachePolicy):
    def __init__(self, cache_size):
        super().__init__(cache_size)
        self.cache = set()
        self.order = deque()
        self.cache_hits = 0

    def access(self, expert_id):
        if expert_id in self.cache:
            self.cache_hits += 1
            # 更新使用顺序
            self.order.remove(expert_id)
            self.order.append(expert_id)
        else:
            if len(self.cache) >= self.cache_size:
                # 移除最久未使用的专家
                lru_expert = self.order.popleft()
                self.cache.remove(lru_expert)
            self.cache.add(expert_id)
            self.order.append(expert_id)

# 实现最不经常使用（LFU）缓存策略
class LFUCachePolicy(CachePolicy):
    def __init__(self, cache_size):
        super().__init__(cache_size)
        self.cache = set()
        self.freq = defaultdict(int)
        self.cache_hits = 0

    def access(self, expert_id):
        if expert_id in self.cache:
            self.cache_hits += 1
            self.freq[expert_id] += 1
        else:
            if len(self.cache) >= self.cache_size:
                # 找到频率最低的专家
                min_freq = min(self.freq.values()) if self.freq else 0
                experts_with_min_freq = [exp for exp in self.cache if self.freq[exp] == min_freq]
                expert_to_remove = random.choice(experts_with_min_freq)
                self.cache.remove(expert_to_remove)
                del self.freq[expert_to_remove]
            self.cache.add(expert_id)
            self.freq[expert_id] += 1

class CacheSimulator:
    def __init__(self, cache_policy, cache_sizes, lookahead=2*32*32):
        self.cache_policy = cache_policy
        self.cache_sizes = cache_sizes
        # to be deprecated #
        self.cache_state = []
        self.freq = {}
        self.hit_count = 0
        # to be deprecated #
        self.total_access = 0
        self.lookahead = lookahead

    # 这些功能暂时保留，但是后续直接用policy class统一实现
    def is_in_cache(self, expert_id):
        return expert_id in self.cache_state

    def update_cache_hit(self, expert_id):
        if self.cache_policy == 'LRU':
            self.cache_state.remove(expert_id)
            self.cache_state.append(expert_id)
        elif self.cache_policy == 'LFU':
            self.freq[expert_id] += 1
            self.cache_state.remove(expert_id)
            self.cache_state.append(expert_id)

    def update_cache_miss(self, expert_id, cache_size):
        if len(self.cache_state) < cache_size:
            self.cache_state.append(expert_id)
            if self.cache_policy == 'LFU':
                self.freq[expert_id] = 1
        else:
            if self.cache_policy == 'LRU':
                self.cache_state.pop(0)
                self.cache_state.append(expert_id)
            elif self.cache_policy == 'LFU':
                min_freq = min(self.freq[expert] for expert in self.cache_state)
                candidates = [expert for expert in self.cache_state if self.freq[expert] == min_freq]
                to_remove = next(expert for expert in self.cache_state if expert in candidates)
                self.cache_state.remove(to_remove)
                del self.freq[to_remove]
                self.cache_state.append(expert_id)
                self.freq[expert_id] = 1

    def simulate(self, all_expert_activations, cache_size):
        self.cache_state = []
        if self.cache_policy == 'LFU':
            self.freq = {}
        self.hit_count = 0
        self.total_access = 0
        batch_size, seq_len = all_expert_activations[0].shape[:2]
        num_moe_layers = len(all_expert_activations)
        chr_detail = [{'layer_idx': layer_idx, 'cache_size': cache_size, 'cache_policy': self.cache_policy, 'chr': 0.0} for layer_idx in range(num_moe_layers)] # chr means cache hit rate
        for b in range(batch_size): # batch_size = 1
            for t in range(seq_len):
                for layer_idx in range(num_moe_layers):
                    experts = all_expert_activations[layer_idx][b, t].tolist()
                    # chr_layer_ = {'layer_idx': layer_idx, 'cache_size': cache_size, 'cache_policy': self.cache_policy}
                    start_access_ = self.total_access
                    start_hit_count_ = self.hit_count
                    for expert in experts:
                        expert_id = (layer_idx, expert)
                        self.total_access += 1
                        if self.is_in_cache(expert_id):
                            self.hit_count += 1
                            self.update_cache_hit(expert_id)
                        else:
                            self.update_cache_miss(expert_id, cache_size)
                    chr_layer_ = (self.hit_count - start_hit_count_) / (self.total_access - start_access_) if (self.total_access - start_access_) > 0 else 0.0
                    chr_detail[layer_idx]['chr'] = chr_detail[layer_idx]['chr'] + chr_layer_
        
        for layer_idx in range(num_moe_layers):
            chr_detail[layer_idx]['chr'] = chr_detail[layer_idx]['chr'] / seq_len
        
        return (self.hit_count / self.total_access, chr_detail) if self.total_access > 0 else (0, [])
    
    def simulate_v2(self, all_expert_activations, cache_size):
        self.total_access = 0
        batch_size, seq_len = all_expert_activations[0].shape[:2]
        num_moe_layers = len(all_expert_activations)
        chr_detail = [{'layer_idx': layer_idx, 'cache_size': cache_size, 'cache_policy': self.cache_policy, 'chr': 0.0} for layer_idx in range(num_moe_layers)] # chr means cache hit rate
        
        # 预存访问序列（用于OptimalCache和LimitedOptimalCache）
        if self.cache_policy in ['Optimal', 'LimitedOptimal']:
            access_sequence = []
            for b in range(batch_size): # batch_size = 1
                for t in range(seq_len):
                    for layer_idx in range(num_moe_layers):
                        experts = all_expert_activations[layer_idx][b, t].tolist()
                        for expert in experts:
                            expert_id = (layer_idx, expert)
                            access_sequence.append(expert_id)
        else:
            access_sequence = []
            
        # 初始化对应的缓存策略实例
        if self.cache_policy == 'Optimal':
            self.cache_instance = OptimalCachePolicy(cache_size=cache_size, access_sequence=access_sequence)
        elif self.cache_policy == 'LimitedOptimal':
            self.cache_instance = LimitedOptimalCachePolicy(cache_size=cache_size, access_sequence=access_sequence, lookahead=self.lookahead)
        elif self.cache_policy == 'LRU':
            self.cache_instance = LRUCachePolicy(cache_size=cache_size)
        elif self.cache_policy == 'LFU':
            self.cache_instance = LFUCachePolicy(cache_size=cache_size)
        else:
            raise ValueError(f"Unsupported cache policy: {self.cache_policy}")
        
        # simulate the inference processing
        for b in range(batch_size): # batch_size = 1
            for t in range(seq_len):
                for layer_idx in range(num_moe_layers):
                    experts = all_expert_activations[layer_idx][b, t].tolist()
                    start_access_ = self.total_access
                    start_hit_count_ = self.cache_instance.cache_hits
                    for expert in experts:
                        expert_id = (layer_idx, expert)
                        if self.cache_policy in ['LRU', 'LFU']:
                            self.cache_instance.access(expert_id)
                        else:
                            self.cache_instance.access(self.total_access) # for optimal policy, the input is the current position idx
                        self.total_access += 1
                    chr_layer_ = (self.cache_instance.cache_hits - start_hit_count_) / (self.total_access - start_access_) if (self.total_access - start_access_) > 0 else 0.0
                    chr_detail[layer_idx]['chr'] = chr_detail[layer_idx]['chr'] + chr_layer_
        
        for layer_idx in range(num_moe_layers):
            chr_detail[layer_idx]['chr'] = chr_detail[layer_idx]['chr'] / seq_len
        
        return (self.cache_instance.cache_hits / self.total_access, chr_detail) if self.total_access > 0 else (0, [])

def simulate_cache(expert_activations, cache_sizes, cache_policy, lookahead):
    simulator = CacheSimulator(cache_policy, cache_sizes, lookahead)
    hit_rates = []
    chr_details = []
    for cache_size in cache_sizes:
        # temp_res = simulator.simulate(expert_activations, cache_size)
        temp_res = simulator.simulate_v2(expert_activations, cache_size)
        hit_rate, chr_detail = temp_res
        hit_rates.append(hit_rate)
        chr_details.extend(chr_detail)
    # print('check detail chr: ', chr_details)
    return hit_rates, chr_details

class MetricsCalculator:
    def calculate_metrics(self, all_expert_activations, all_gate_scores):
        # shape of all_expert_activations: [tensor(1, seq_len, topk)], len(all_expert_activations)=layer_num
        # shape of all_gate_scores: [tensor(1, seq_len, exp_num)], len(all_gate_scores)=layer_num
        seq_len = all_expert_activations[0].shape[1]
        num_moe_layers = len(all_expert_activations)
        top_k = all_expert_activations[0].shape[2]
        total_ecr = total_gds = total_lal = 0.0
        num_pairs = seq_len - 1
        routing_sim_detail = [{'layer_idx': layer_idx, 'ecr': 0.0, 'gds': 0.0, 'lal': 0.0} for layer_idx in range(num_moe_layers)]
        for t in range(num_pairs):
            current_experts = [layer[0, t].tolist() for layer in all_expert_activations]
            next_experts = [layer[0, t+1].tolist() for layer in all_expert_activations]
            current_gates = [layer[0, t] for layer in all_gate_scores]
            next_gates = [layer[0, t+1] for layer in all_gate_scores]
            ecr, ecr_details = self.calculate_ecr(current_experts, next_experts, top_k)
            gds, gds_details = self.calculate_gds(current_gates, next_gates)
            lal, lal_details = self.calculate_lal(current_experts, next_experts, num_moe_layers, top_k)
            total_ecr += ecr
            total_gds += gds
            total_lal += lal
            for layer_idx in range(num_moe_layers):
                routing_sim_detail[layer_idx]['ecr'] = routing_sim_detail[layer_idx]['ecr'] + ecr_details[layer_idx]
                routing_sim_detail[layer_idx]['gds'] = routing_sim_detail[layer_idx]['gds'] + gds_details[layer_idx]
                routing_sim_detail[layer_idx]['lal'] = routing_sim_detail[layer_idx]['lal'] + lal_details[layer_idx]
                
        avg_ecr = total_ecr / num_pairs if num_pairs > 0 else 0
        avg_gds = total_gds / num_pairs if num_pairs > 0 else 0
        avg_lal = total_lal / num_pairs if num_pairs > 0 else 0
        if num_pairs > 0:
            for layer_idx in range(num_moe_layers):
                routing_sim_detail[layer_idx]['ecr'] = routing_sim_detail[layer_idx]['ecr'] / num_pairs
                routing_sim_detail[layer_idx]['gds'] = routing_sim_detail[layer_idx]['gds'] / num_pairs
                routing_sim_detail[layer_idx]['lal'] = routing_sim_detail[layer_idx]['lal'] / num_pairs
        return (avg_ecr, avg_gds, avg_lal), routing_sim_detail

    def calculate_ecr(self, current_experts, next_experts, top_k):
        total_score = 0.0
        num_layers = len(current_experts)
        detail_layers = []
        for curr, nxt in zip(current_experts, next_experts):
            common = len(set(curr) & set(nxt))
            total_score += common / top_k
            detail_layers.append(common / top_k)
        return total_score / num_layers, detail_layers

    def calculate_gds(self, current_gates, next_gates):
        total_sim = 0.0
        num_layers = len(current_gates)
        detail_layers = []
        for g1, g2 in zip(current_gates, next_gates):
            cos_sim = F.cosine_similarity(g1, g2, dim=0)
            total_sim += (cos_sim + 1) / 2
            detail_layers.append(((cos_sim + 1) / 2).item())
        return total_sim / num_layers, detail_layers

    def calculate_lal(self, current_experts, next_experts, num_moe_layers, top_k):
        layer_indices = torch.arange(1, num_moe_layers + 1, dtype=torch.float32)
        weights = layer_indices / num_moe_layers
        total_score = 0.0
        detail_layers = []
        for layer_idx, (curr, nxt) in enumerate(zip(current_experts, next_experts)):
            common_experts = len(set(curr) & set(nxt))
            layer_ecr = common_experts / top_k
            total_score += weights[layer_idx] * layer_ecr
            detail_layers.append((weights[layer_idx] * layer_ecr).item())
        return total_score.item(), detail_layers

class Inferencer:
    def __init__(self, config):
        self.config = config
        self.rank = dist.get_rank()
        self.device = torch.device(f'cuda:{self.rank}')
        self.inference_engine = InferenceEngine(config, self.rank, self.device)
        moe_n, experts_n = self.inference_engine.get_moe_metainfo()
        self.cache_simulator = CacheSimulator(config.cache_policy, [experts_n * i for i in range(1, moe_n)], lookahead=2 * moe_n * self.config.num_active_experts)
        self.metrics_calculator = MetricsCalculator()
        self.cache_policies = ['lru', 'lfu', 'opt', 'lopt']

        self.eval_handlers = {
            'alpaca':   self.eval_alpaca,
            'aqua':     self.eval_aqua,
            'boolq':    self.eval_boolq,
            'gsm8k':    self.eval_gsm8k,
            'mathqa':   self.eval_mathqa,
            'mawps':    self.eval_mawps,
            'mmlu':     self.eval_mmlu,
            'obqa':     self.eval_obqa,
            'piqa':     self.eval_piqa,
            'samsum':   self.eval_samsum,
            'siqa':     self.eval_siqa,
            'wikitext': self.eval_wikitext,
            'common-v1': self.eval_common_v1,
            'common-v2': self.eval_common_v2,
        }

    def _distribute_test_cases(self, test_cases, case_num, case_start):
        world_size = dist.get_world_size()
        total_cases = min(case_num, len(test_cases))
        if case_start >= total_cases:
            case_start = 0
        per_process = (total_cases - case_start) // world_size
        remainder = (total_cases - case_start) % world_size # fix a bug when case_start != 0 
        start = self.rank * per_process + min(self.rank, remainder)
        end = start + per_process + (1 if self.rank < remainder else 0)
        return test_cases[case_start + start : case_start + end], total_cases

    def _process_batch(self, batch_questions, use_lora):
        start_ts = time.time()
        generated_texts, all_expert_activations, all_gate_scores, logits = self.inference_engine.generate(
            batch_questions, use_lora)
        infer_ts = time.time()
        logger.info(f"Inference cost {(infer_ts - start_ts):.2f} s, Batch size: {len(batch_questions)}")
        return generated_texts, all_expert_activations, all_gate_scores, infer_ts, logits

    def chat_with_model(self, use_lora=False, log_path=''):
        # 20250928: 直接与模型对话
        import re
        print(f"Start to chat to model with LoRA set to {use_lora}: {self.inference_engine.model if use_lora else self.inference_engine.base_model}. \n")
        chat_idx = 0
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ['exit', 'quit', 'q', 'e']:
                    print("Goodbye!")
                    break
                response = self.inference_engine.generate([user_input], use_lora)[0][0]
                response = response.split('##响应：')[-1]# .encode('utf-8').decode('unicode_escape') # 显示换行符
                response = re.sub(r'\\n', '\n', response)   # 关键修复
                user_input = re.sub(r'\\n', '\n', user_input)
                print(f"Model: {response}")
                
                 # 新增：追加写入日志
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"Chat idx: {chat_idx}\nUser: {user_input}\nBot: {response}\n\n")
                    chat_idx += 1
            except KeyboardInterrupt:
                print("\nSession ended by user.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                break
    
    def _compute_metrics_and_cache(self, all_expert_activations, all_gate_scores, batch_size_actual):
        all_metrics = []
        cache_hit_rates = {f"{policy}_hit_rate_cacheSize_{size}": [[] for _ in range(batch_size_actual)]
                           for policy in ['lru', 'lfu', 'opt', 'lopt'] for size in self.cache_simulator.cache_sizes}
        routing_sim_details = []
        chr_details = []

        if all_expert_activations and all_gate_scores:
            tasks = [( [layer[b:b+1] for layer in all_expert_activations], self.cache_simulator.cache_sizes)
                     for b in range(batch_size_actual)]

            with mp.Pool(processes=batch_size_actual) as pool:
                lru_results = pool.starmap(simulate_cache, [(task[0], task[1], "LRU", self.cache_simulator.lookahead) for task in tasks]) # add the lookahead parameter
                lfu_results = pool.starmap(simulate_cache, [(task[0], task[1], "LFU", self.cache_simulator.lookahead) for task in tasks])
                opt_results = pool.starmap(simulate_cache, [(task[0], task[1], "Optimal", self.cache_simulator.lookahead) for task in tasks])
                lopt_results = pool.starmap(simulate_cache, [(task[0], task[1], "LimitedOptimal", self.cache_simulator.lookahead) for task in tasks])

            for b in range(batch_size_actual):
                # calculate ecr, gds and lal for each sentence.
                batch_expert_activations = [layer[b:b+1] for layer in all_expert_activations]
                batch_gate_scores = [layer[b:b+1] for layer in all_gate_scores]
                (ecr, gds, lal), routing_sim_detail = self.metrics_calculator.calculate_metrics(batch_expert_activations, batch_gate_scores)
                all_metrics.append((ecr, gds, lal))
                routing_sim_details.append(routing_sim_detail)

                for cache_size, lru_rate, lfu_rate, opt_rate, lopt_rate in zip(self.cache_simulator.cache_sizes, lru_results[b][0], lfu_results[b][0], opt_results[b][0], lopt_results[b][0]):
                    cache_hit_rates[f"lru_hit_rate_cacheSize_{cache_size}"][b].append(lru_rate)
                    cache_hit_rates[f"lfu_hit_rate_cacheSize_{cache_size}"][b].append(lfu_rate)
                    cache_hit_rates[f"opt_hit_rate_cacheSize_{cache_size}"][b].append(opt_rate)
                    cache_hit_rates[f"lopt_hit_rate_cacheSize_{cache_size}"][b].append(lopt_rate)
                
                chr_detail_lru = lru_results[b][1]
                chr_detail_lfu = lfu_results[b][1]
                chr_detail_opt = opt_results[b][1]
                chr_detail_lopt = lopt_results[b][1]
                chr_details.append(chr_detail_lru + chr_detail_lfu + chr_detail_opt + chr_detail_lopt)

        return all_metrics, cache_hit_rates, routing_sim_details, chr_details

    def _aggregate_results(self, correct_count, local_performance, local_test_cases,
                                 all_metrics, cache_hit_rates, total_cases,
                                 compute_acc_only):
        performance_metrics = {}
        local_correct = torch.tensor(correct_count, device=self.device)
        local_total = torch.tensor(len(local_test_cases), device=self.device)
        dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total, op=dist.ReduceOp.SUM)
        
        for key in local_performance.keys():
            # local_ppl = torch.tensor(local_ppl, device=self.device)
            # dist.all_reduce(local_ppl, op=dist.ReduceOp.SUM)
            local_met = local_performance[key] # TODO：预防出错
            local_met = torch.tensor(local_met, device=self.device)
            dist.all_reduce(local_met, op=dist.ReduceOp.SUM)
            performance_metrics[key] = local_met.item()
            
        total = local_total.item()
        accuracy = local_correct.item() / total if total > 0 else 0
        if compute_acc_only:
            result_map = { "accuracy": f"{accuracy:.4f}" }
            return accuracy, result_map
        all_ecr, all_gds, all_lal = zip(*all_metrics) if all_metrics else ([], [], [])
        ecr_tensor = torch.tensor(all_ecr, dtype=torch.float32, device=self.device)
        gds_tensor = torch.tensor(all_gds, dtype=torch.float32, device=self.device)
        lal_tensor = torch.tensor(all_lal, dtype=torch.float32, device=self.device)
        dist.all_reduce(ecr_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(gds_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(lal_tensor, op=dist.ReduceOp.SUM)

        result_map = {}
        total_cases = float(total_cases)
        if total_cases > 0:
            result_map = {
                "accuracy": f"{accuracy:.4f}",
                # "mean_ppl": f"{local_ppl.item() / total_cases:.4f}",
                "mean_ecr": f"{ecr_tensor.sum().item() / total_cases:.4f}",
                "mean_gds": f"{gds_tensor.sum().item() / total_cases:.4f}",
                "mean_lal": f"{lal_tensor.sum().item() / total_cases:.4f}"
            }
            result_map.update({
                f"mean_{key}": f"{performance_metrics[key] / total_cases:.4f}" for key in performance_metrics.keys()
            })
        
        all_hit_rates = {key: [] for key in self.cache_policies}
        for policy in self.cache_policies: # ['lru', 'lfu']:
            for size in self.cache_simulator.cache_sizes:
                key = f"{policy}_hit_rate_cacheSize_{size}"
                flat_list = [item for sublist in cache_hit_rates[key] for item in sublist]
                tensor = torch.tensor(flat_list, dtype=torch.float32, device=self.device)
                gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered, tensor)
                if self.rank == 0:
                    combined = torch.cat(gathered)
                    avg_hit_rate = combined.mean().item() if combined.numel() > 0 else 0.0
                    result_map[key] = f"{avg_hit_rate:.4f}"
                    all_hit_rates[policy].append(avg_hit_rate)

        return accuracy, result_map, all_hit_rates

    def general_load_prompt(self, dataset_dir):
        with open(os.path.join(dataset_dir, "prompt.json"), "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        return prompt_data

    def compute_acc_metric(self, output, expected):
        """
            Compute accuracy for single pair of [output, expected], according to dataset name. 
        """
        def simple_text_acc(output, expected):
            output = output.strip().lower()
            expected = expected.strip().lower()
            return (1 if output == expected else 0), {}
        
        def qa_w_rationale_acc(output, expected):
            output = output.split("####")
            expected = expected.split("####")
            
            if len(output) != 2:
                # the format of answer was wrong
                return 0, {}
            
            expected = expected[-1].replace(" ", "").upper()
            output = output[-1].replace(" ", "").upper()
            return (1 if output == expected else 0), {}

        def math_calculation_acc(output, expected):
            output = output.split("####")
            expected = expected.split("####")
            
            if len(output) != 2:
                # the format of answer was wrong
                return 0, {}
            
            expected = expected[-1].replace(" ", "").replace(",", "")
            output = output[-1].replace(" ", "").replace(",", "")
            try:
                expected, output = eval(expected), eval(output)
                return (1 if abs(expected - output) < 1e-5 else 0), {}
            except:
                return 0, {}
        
        def text_ans_acc(output, expected):
            # for samsum, wikitext, alpaca
            output = output.strip().lower()
            expected = expected.strip().lower()
            is_correct = (1 if output == expected else 0)
            
            # rouge_metric = evaluate.load('./metrics/rouge')
            # rouge_res = rouge_metric.compute(
            #     predictions=[output],
            #     references=[expected]
            # )
            rouger = Rouge()
            rouge_res = rouger.get_scores(output, expected)[0]
            rouge_return = {
                'rouge1': rouge_res['rouge-1']['f'],
                'rouge2': rouge_res['rouge-2']['f'],
                'rougeL': rouge_res['rouge-l']['f']
            }
            
            return is_correct, rouge_return
        
        acc_handlers = {
            'alpaca':    text_ans_acc,
            'aqua':      qa_w_rationale_acc,
            'boolq':     simple_text_acc,
            'gsm8k':     math_calculation_acc,
            'mathqa':    qa_w_rationale_acc,
            'mawps':     math_calculation_acc,
            'mmlu':      simple_text_acc,
            'obqa':      simple_text_acc,
            'piqa':      simple_text_acc,
            'samsum':    text_ans_acc,
            'siqa':      simple_text_acc,
            'wikitext':  text_ans_acc,
            'common-v1': simple_text_acc,
            'common-v2': simple_text_acc,
        }
        
        return acc_handlers[self.config.dataset_name](output, expected)
    
    def compute_ppl_metric(self, output, expected, logits_idx):
        """
            idx       the prompt (and question) number in a batch.
            logits    a tuple, the element amount is maximum generated
                        tokens count in the batch. each element is:

            [generated_token_idx] = <class 'torch.Tensor'> that shape is [batch, voca_size]
        """
        def internal_ppl_compute(logits_idx, verbose=False):
            logits, prompt_idx = logits_idx
            max_nr_tokens = len(logits)
            probs = 0.0
            nr_tok = 0
            for tok_idx in range(max_nr_tokens):
                tok_lg = logits[tok_idx][prompt_idx]
                tok_probs = torch.softmax(tok_lg, -1)
                tok_id = torch.argmax(tok_probs)
                if tok_id == self.inference_engine.tokenizer.eos_token_id:
                    break
                probs += math.log(tok_probs[tok_id].item()) # hauser: 这里好像计算ppl的方式并没有考虑true label？是正常的么
                if verbose:
                    logger.info(f"\t{nr_tok}/{max_nr_tokens}, {tok_id}, {tok_probs[tok_id].item()}, {probs}")
                nr_tok += 1
            return nr_tok, probs

        if not logits_idx or len(logits_idx) != 2:
            return GenMetric(
                        got        = output,
                        expected   = exptected,
                        is_correct = 0,
                        ppl        = None) # 异常处理，添加了ppl=None返回值
        nr_tok, probs = internal_ppl_compute(logits_idx)
        ppl = None
        ppl_e = math.exp(probs)
        try:
            if ppl_e < 1e-300:
                ppl = 999 # 异常处理，用于指代ppl趋于无穷
            else:
                ppl = ppl_e ** (-1.0/nr_tok)
        except ZeroDivisionError:
            logger.info(f"Zero nr_tokens or other bug when calcuating PPL_E, internal states: {nr_tok}, {probs}, {ppl_e}")
            logger.info("Detailed error information: ")
            internal_ppl_compute(logits_idx, True)
        
        is_correct, other_metrics = self.compute_acc_metric(output, expected) # add other metrics like: rouge scores
        
        return GenMetric(
                        got        = output,
                        expected   = expected,
                        is_correct = is_correct, # (1 if output == expected else 0)
                        ppl        = ppl,
                        **other_metrics # rouge scores and other metrics
                        )

    def general_eval(self, test_generic, compute_metric_func,
                         extract_ground_truth_func, use_lora=False,
                         case_num=100, setting="", compute_acc_only = False,
                         case_start=0):
        # load test cases
        dataset_dir = os.path.join(self.config.dataset_path, self.config.dataset_name)
        test_file = "test.json" if test_generic else "train.json"
        with open(os.path.join(dataset_dir, test_file), "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        prompt = self.general_load_prompt(dataset_dir)
        if case_num == 0:
            # test only 1/20 train dataset if case_num is not specified
            case_num = len(test_cases) if test_generic else max(len(test_cases) // 20, 1) 
        local_test_cases, total_cases = self._distribute_test_cases(test_cases, case_num, case_start)
        local_test_cases = local_test_cases[-total_cases:]
        cache_hit_rates = {f"{policy}_hit_rate_cacheSize_{size}": [[] for _ in range(len(local_test_cases))]
                           for policy in ['lru', 'lfu', 'opt', 'lopt'] for size in self.cache_simulator.cache_sizes}
        logger.info(f"Number of local test cases: {len(local_test_cases)}, Number of total test cases: {total_cases}")
        
        batch_size = max(1, self.config.inference_batch)
        batch_size = min(512, batch_size)
        running_idx = 0
        correct_count = 0
        # local_ppl = 0.0
        local_performance = {key: 0.0 for key in ["ppl", "rouge1", "rouge2", "rougeL"]} # contain ppl, rouge scores
        all_metrics = []
        routing_sim_details = []
        chr_details = []
        output_df = []
        batch_metrics = None
        for i in tqdm(range(0, len(local_test_cases), batch_size), desc="Progress of inference"):
            batch_cases = local_test_cases[i:i + batch_size]
            if "prompt_wo_input" in prompt.keys():
                batch_questions = [ prompt["prompt_wo_input"].format_map(case) if case["input"] == ""
                                    else prompt["prompt_with_input"].format_map(case) for case
                                    in batch_cases ]
            else:
                batch_questions = [ prompt["prompt"].format_map(case) for case in batch_cases ]
            batch_expected_answers = [extract_ground_truth_func(case) for case in batch_cases]

            #logger.info(f"Inference #{running_idx}: question is {batch_questions}")
            generated_texts, all_expert_activations, all_gate_scores, infer_ts, logits = self._process_batch(batch_questions, use_lora)
            if not compute_acc_only:
                batch_metrics, batch_cache_hit_rates, batch_routing_sim_details, batch_chr_details = self._compute_metrics_and_cache(all_expert_activations, all_gate_scores, len(batch_questions))
                all_metrics.extend(batch_metrics)
                for key in cache_hit_rates:
                    for b in range(len(batch_questions)):
                        cache_hit_rates[key][i + b] = batch_cache_hit_rates[key][b]
                logger.info(f"Simulate cache cost {(time.time() - infer_ts):.2f} s, Batch size: {len(batch_questions)}")
            for idx, (text, expected) in enumerate(zip(generated_texts, batch_expected_answers)):
                output_only = text.split("### Response:")[-1].strip() if "### Response:" in text else "" # ensure get the text after ### Response:
                m = compute_metric_func(str(output_only), str(expected), (logits, idx))
                expected = "\"" + m.expected.replace("\n", " ") + "\""
                got = "\"" + m.got.replace("\n", " ") + "\""
                
                routing_sim_detail_idx, chr_detail_idx = batch_routing_sim_details[idx], batch_chr_details[idx]
                for i in range(len(routing_sim_detail_idx)):
                    routing_sim_detail_idx[i]['sample_id'] = running_idx
                for i in range(len(chr_detail_idx)):
                    chr_detail_idx[i]['sample_id'] = running_idx
                routing_sim_details.extend(routing_sim_detail_idx)
                ecr_this_sample = np.mean([x['ecr'] for x in routing_sim_detail_idx])
                chr_details.extend(chr_detail_idx)
                
                if m.ppl is None:
                    logger.error(f"Inference #{running_idx}: compute_metric_func output abnormal value, Expected {expected}, Got {got}, (logits-example, idx) {(logits[0].shape, idx)}")
                    running_idx += 1
                    continue
                    
                correct_count += m.is_correct
                # local_ppl += m.ppl
                for met_nm in local_performance.keys():
                    local_performance[met_nm] = local_performance[met_nm] + getattr(m, met_nm)
                
                ppl = int(m.ppl * 1000)/1000.0
                logger.info(f"Inference #{running_idx}: Expected {expected}, Got {got}, PPL {ppl}, Correct {m.is_correct}")
                output_df.append({'running_idx': running_idx, 'Expected': expected, 'Got': got, 'ECR': ecr_this_sample})
                if batch_metrics:
                    logger.info("=" * 30)
                    logger.info(f"ECR: {batch_metrics[idx][0]:.4f}, GDS: {batch_metrics[idx][1]:.4f}, LAL: {batch_metrics[idx][2]:.4f}")
                running_idx += 1
        
        if self.rank == 0:
            plt_dir = f"./plt/{self.config.model_key}"
            output_df = pd.DataFrame(output_df)
            output_df.to_csv(f'{plt_dir}/{self.config.dataset_name}_{self.config.consecutive_expert_loss_weight}.csv', index=False, encoding="utf_8_sig")
        
        if compute_acc_only:
            accuracy, result_map = self._aggregate_results(correct_count,
                                                           local_test_cases,
                                                           compute_acc_only = compute_acc_only)
        else:
            accuracy, result_map, all_hit_rates = self._aggregate_results(correct_count,
                                                           local_performance,
                                                           local_test_cases,
                                                           all_metrics,
                                                           cache_hit_rates,
                                                           total_cases,
                                                           compute_acc_only)
            if self.rank == 0:
                logger.info("Plot figure")
                self.plot_memory_usage_vs_throughput(
                    self.cache_simulator.cache_sizes,
                    {policy: list(all_hit_rates[policy]) for policy in self.cache_policies}
                    )#list(all_hit_rates["lru"]), list(all_hit_rates["lfu"]))
                plt_dir = f"./plt/{self.config.model_key}"
                output_df = pd.DataFrame(output_df)
                output_df.to_csv(f'{plt_dir}/{self.config.dataset_name}_{self.config.consecutive_expert_loss_weight}.csv', index=False, encoding="utf_8_sig")
        if self.rank == 0:
            print(f"\nAccuracy: {accuracy:.4%}")
        return (result_map, pd.DataFrame(routing_sim_details), pd.DataFrame(chr_details))

    def eval_mmlu(self, use_lora=False, case_num = 0, test_generic = True, setting=""):
        def extract_ground_truth_func(example):
            return example["answer"]
        return self.general_eval(test_generic, self.compute_ppl_metric, extract_ground_truth_func, use_lora, case_num, setting)

    def eval_alpaca(self, use_lora=False, case_num = 0, test_generic = True, setting=""):
        def extract_ground_truth_func(example):
            return example["output"]
        return self.general_eval(test_generic, self.compute_ppl_metric, extract_ground_truth_func, use_lora, case_num, setting)

    def eval_aqua(self, use_lora=False, case_num = 0, test_generic = True, setting=""):
        def extract_ground_truth_func(example):
            return f"{example['rationale']}\n#### {example['answer']}" #TODO: CHECK IF RATIONALE necessary
        return self.general_eval(test_generic, self.compute_ppl_metric, extract_ground_truth_func, use_lora, case_num, setting)

    def eval_boolq(self, use_lora=False, case_num = 0, test_generic = True, setting=""):
        def extract_ground_truth_func(example):
            return example["answer"]
        return self.general_eval(test_generic, self.compute_ppl_metric, extract_ground_truth_func, use_lora, case_num, setting)

    def eval_gsm8k(self, use_lora=False, case_num = 0, test_generic = True, setting=""):
        def extract_ground_truth_func(example):
            return example["answer"]
        return self.general_eval(test_generic, self.compute_ppl_metric, extract_ground_truth_func, use_lora, case_num, setting)

    def eval_mathqa(self, use_lora=False, case_num = 0, test_generic = True, setting=""):
        def extract_ground_truth_func(example):
            return f"{example['rationale']}\n#### {example['answer']}" #TODO: CHECK IF RATIONALE necessary
        return self.general_eval(test_generic, self.compute_ppl_metric, extract_ground_truth_func, use_lora, case_num, setting)

    def eval_mawps(self, use_lora=False, case_num = 0, test_generic = True, setting=""):
        def extract_ground_truth_func(example):
            # return f"{example['rationale']}\n#### {example['answer']}"
            return f"Equation is {example['equation'].lower()}.\nThe solution is x={example['ans']}.\n#### {example['ans']}" # fix a bug
        return self.general_eval(test_generic, self.compute_ppl_metric, extract_ground_truth_func, use_lora, case_num, setting)

    def eval_obqa(self, use_lora=False, case_num = 0, test_generic = True, setting=""):
        def extract_ground_truth_func(example):
            return example["answer"]
        return self.general_eval(test_generic, self.compute_ppl_metric, extract_ground_truth_func, use_lora, case_num, setting)

    def eval_piqa(self, use_lora=False, case_num = 0, test_generic = True, setting=""):
        def extract_ground_truth_func(example):
            return str(example["answer"])
        return self.general_eval(test_generic, self.compute_ppl_metric, extract_ground_truth_func, use_lora, case_num, setting)

    def eval_samsum(self, use_lora=False, case_num = 0, test_generic = True, setting=""):
        def extract_ground_truth_func(example):
            return str(example["summary"])
        return self.general_eval(test_generic, self.compute_ppl_metric, extract_ground_truth_func, use_lora, case_num, setting)

    def eval_siqa(self, use_lora=False, case_num = 0, test_generic = True, setting=""):
        def extract_ground_truth_func(example):
            return str(example["answer"])
        return self.general_eval(test_generic, self.compute_ppl_metric, extract_ground_truth_func, use_lora, case_num, setting)

    def eval_wikitext(self, use_lora=False, case_num = 0, test_generic = True, setting=""):
        def extract_ground_truth_func(example):
            return example["content"]
        return self.general_eval(test_generic, self.compute_ppl_metric, extract_ground_truth_func, use_lora, case_num, setting)

    def eval_common_v1(self, use_lora=False, case_num = 0, test_generic = True, setting=""):
        def extract_ground_truth_func(example):
            return example["output"]
        return self.general_eval(test_generic, self.compute_ppl_metric, extract_ground_truth_func, use_lora, case_num, setting)

    def eval_common_v2(self, use_lora=False, case_num = 0, test_generic = True, setting=""):
        def extract_ground_truth_func(example):
            return example["output"]
        return self.general_eval(test_generic, self.compute_ppl_metric, extract_ground_truth_func, use_lora, case_num, setting)

    def plot_memory_usage_vs_throughput(self, cache_sizes, hit_rates_policies): # lru_hit_rates, lfu_hit_rates
        # 这部分是模型参数后续留出接口：
        # 量化后
        W_q = W_k = W_v = W_o = 2.25  # MB
        W_tok_embed = 112.5  # MB
        N_experts = 64
        E_size = 4.866  # MB
        N_shared_experts = 2
        L = 28
        N_act_experts = 6
        # 计算常驻内存和动态内存
        W_resident = (W_q + W_k + W_v + W_o) * L + W_tok_embed + N_experts * E_size + N_shared_experts * E_size * (L - 1)
        W_dynamic = N_act_experts * E_size * (L - 1)
        
        # 以下是硬件参数不用修改：
        cpu_mem_b = 45 * 1024 # MB/s
        npu_mem_b = 75 * 1024 # MB/s
        B_IO = 4.5 * 1024 # MB/s

        # 整理cache-sizes、计算吞吐、保存画图数据
        memory_cost = [(W_resident + E_size * i) / 1024 for i in cache_sizes]
        cpu_datas, npu_datas, data_list = [], [], []
        for policy, hit_rates in hit_rates_policies.items():
            cpu_data = [1/((W_resident + W_dynamic * rate) / cpu_mem_b + (W_dynamic * (1 - rate) / B_IO)) for rate in hit_rates]
            npu_data = [1/((W_resident + W_dynamic * rate) / npu_mem_b + (W_dynamic * (1 - rate) / B_IO)) for rate in hit_rates]
            data_list.append((cpu_data, npu_data, f'CPU_{policy}', f'NPU_{policy}', 'b', 'c', 's', '^', policy))

        sub_plot_num = len(data_list)
        plt.figure(figsize=(14, 8))
        fig, axes = plt.subplots(sub_plot_num // 2, 2, figsize=(14, 8))
        # 扁平化 axes 便于索引
        axes = axes.flatten()

        for ax, (data1, data2, label1, label2, color1, color2, marker1, marker2, method) in zip(axes, data_list):
            ax.plot(memory_cost, data1, label=label1, marker=marker1, color=color1)
            ax.plot(memory_cost, data2, label=label2, marker=marker2, color=color2)
            
            ax.set_xlabel('Memory Usage (GB)', fontsize=14)
            ax.set_ylabel('Decode Throughput (Tokens/s)', fontsize=14)
            ax.set_title(f'Memory Usage vs. Decode Throughput using {method}', fontsize=14)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        model_key = self.config.model_key
        plt_dir = f"./plt/{model_key}"
        os.makedirs(plt_dir, exist_ok=True)
        dataset = self.config.dataset_name
        plt.savefig(f"{plt_dir}/memory_throughput_{dataset}_{setting}.png")

def record_result(dataset, setting, result_lora, model_key = '_qwen2_4b'):
    result_lora, routing_sim_details, chr_details = result_lora
    
    result_path = f"./eval_result/{model_key}" 
    os.makedirs(result_path, exist_ok=True)
    csv_file = f"{result_path}/{dataset}_{setting}.csv"
    excel_file = f"{result_path}/{dataset}_{setting}.xlsx"
    file_exist = os.path.exists(csv_file)
    with open(csv_file, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exist:
            writer.writerow([setting] + [None] * len(result_lora.keys()))
            writer.writerow([None] + list(result_lora.keys()))
        writer.writerow(["lora"] + list(result_lora.values()))
    df = pd.read_csv(csv_file, header=None)
    df.to_excel(excel_file, index=False, header=False, engine='openpyxl')
    print(f"CSV file '{csv_file}' has been converted to Excel as '{excel_file}'.")
    
    routing_sim_details.to_csv(csv_file.replace('.csv', '_routingSim.csv'), index=False)
    chr_details.to_csv(csv_file.replace('.csv', '_chrDetail.csv'), index=False)

def update_config(config, new_config):
    for attribute in new_config.__dict__:
        if attribute not in ['target_modules', 'lora_extra_modules']:
            setattr(config, attribute, getattr(new_config, attribute))
        else:
            attr_now = getattr(new_config, attribute)
            attr_now = tuple(attr_now.split(',')) if len(attr_now) > 0 else ()
            setattr(config, attribute, attr_now)
        print(f"iterate over attribute: {attribute}, set to {getattr(config, attribute)} ")
    
    model_key = config.model_key # 新增：用于区分qwen还是deepseek模型
    batch_size = config.batch_size * config.gradient_accumulation_steps
    lora_rank = config.lora_rank
    # config.logging_dir = f"./logs{model_key}/{config.dataset_name}/lr_{config.learning_rate}_aux_{config.aux_loss_alpha}_type_{config.consecutive_expert_loss_type}_weight_{config.consecutive_expert_loss_weight}_bs_{batch_size}_rank_{lora_rank}"

if __name__ == "__main__":
    dist.init_process_group(backend='nccl')
    config = Config()
    args = get_arguments()
    model_key = args.model_key
    update_config(config, args)
    infer = Inferencer(config)
    logger = setup_distributed_inference_logging(model_key = model_key)
    case_num = 50 # maximum cases (set to a small number in order to speed up, set to zero for inference all cases in the dataset)
    use_lora = args.use_lora
    
    if config.chat_mode: 
        log_path = f'./logs/chat_history/{model_key}_{config.dataset_name}_lora_{use_lora}.txt'
        infer.chat_with_model(use_lora=use_lora, log_path=log_path)
        exit() # 直接对话，不需要批量推理

    for is_generic in [True]: # only use test set now. original is [True, False]
        setting = f"lora_{use_lora}_generic_{is_generic}_lr_{args.learning_rate}_aux_{args.aux_loss_alpha}_celType_{args.consecutive_expert_loss_type}_celWeight_{args.consecutive_expert_loss_weight}"
        if dist.get_rank() == 0:
            print("Base+LoRA Model Inference:")
            logger.info(f"Start inference for dataset: {config.dataset_name}")
            logger.info(f"Setting: {setting}")
        result_lora = infer.eval_handlers[config.dataset_name](use_lora=use_lora, case_num=case_num, test_generic=is_generic, setting=setting)
        if dist.get_rank() == 0:
            record_result(config.dataset_name, setting, result_lora, model_key+f'_{case_num}')
    dist.destroy_process_group()
