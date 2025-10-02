from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    """微调框架的配置类"""
    model_name: str = "./models/smallthinker-moe-4b-thinker"        # 模型名称 (deepseek-moe-16b-chat)
    task_type: str = "causal_lm"                                    # 任务类型
    device: str = "cuda" if torch.cuda.is_available() else "cpu"    # 设备
    
    # model args
    use_lora: bool = True                                           # 是否使用 LoRA
    lora_rank: int = 8                                              # LoRA 秩
    lora_alpha: int = 32                                            # LoRA alpha 参数
    lora_extra_modules: tuple[str] = ()			                    # LoRA微调时更新的扩展模块, 即LoRAConfig的modules_to_save字段（"gate",） for deepseek-moe
    target_modules: tuple[str] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                  "up", "gate", "down",
                                  "primary_router")                 # LoRA微调时更新的目标模块（"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") for deepseek-moe
    lora_dropout: float = 0.1                                       # LoRA微调时的dropout，原本是0.1
    mixed_precision: bool = False                                   # 是否使用混合精度
    seed: int = 42                                                  # 随机种子
    use_quant: bool = True                                          # 是否启用量化微调
    do_sample: bool = True                                          # 启用采样
    temperature: float = 0.2                                        # 设置温度
    top_p: float = 0.9                                              # 设置 top_p
    repetition_penalty = 1.5                                        # 重复词汇的惩罚
    moe_module_name = 'block_sparse_moe'                            # 设置模型中 moe模块的变量名用于inference时获取激活experts信息（mlp for deepseek-moe）
    
    # log and model saving
    checkpoint_path: str = "./checkpoints"                          # 检查点保存目录
    save_checkpoint: bool = False                                   # 是否保存检查点
    resume_checkpoint: bool = True                                  # 是否从checkpoint上加载
    save_lora_adapter: bool = True                                  # 是否保存Lora适配器
    save_merged_model: bool = False                                 # 是否保存完整模型（注意需要大存储空间）
    save_best_model: bool = True                                    # 是否保存最佳模型
    best_metric: str = "loss"                                       # 评估指标（当前仅支持loss）
    merged_model_save_path: str = "./merged_models"                 # 保存微调后的完整模型
    
    # dataset args
    data_preprocess_only: bool = False
    dataset_path: str = "./datasets-hauser4train" # "./datasets-hauser4train"
    dataset_name: str = "common-v1"
    max_words_len: int = 1024

    # train args
    lora_save_path: str = "./LoRA_models-smallthinker"              # 保存微调后的LoRA模型
    num_train_epochs: int = 10                                      # 训练轮数
    batch_size: int = 20                                            # 批量大小
    gradient_accumulation_steps: int = 1                            # 梯度累积步数
    learning_rate: float = 1e-3                                     # 学习率
    logging_dir: str = "./logs"                                     # 日志目录
    logging_steps: int = 1
    save_steps: int = 100                                           # 每多少步保存检查点
    save_on_each_node: bool = True
    save_total_limit: int = 50                                      # 限制总的checkpoints数目减少内存消耗
    gradient_checkpointing: bool = True
    label_names: list[str] = field(default_factory=lambda: ["labels"])
    eval_strategy: str = "steps"
    eval_steps: int = 10000
    lr_scheduler_type: str = "reduce_lr_on_plateau" # "constant_with_warmup"
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"factor": 0.95, "patience": 5, "threshold": 1e-3, "min_lr": 1e-5}) # field(default_factory=lambda: {"num_warmup_steps": 30})
    num_warmup_steps: int = 50
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    ds_config: str = "./configs/ds_config_zero2_no_offload.json" # "./configs/ds_config_zero2_no_offload.json"

    expert_cache_size: int = 256
    cache_policy: str = "LRU"
    inference_batch: int = 32
    eval_dataset: str = "test.json"                                 # 这个最后没用到，inference中显示指定test.json了
