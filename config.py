from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    """Configuration class for CFMoE framework."""
    model_name: str = "./models/deepseek-moe-16b-chat"
    task_type: str = "causal_lm"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # model args
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_extra_modules: tuple[str] = ()
    target_modules: tuple[str] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                  "up", "gate", "down",
                                  "primary_router")
    lora_dropout: float = 0.1
    mixed_precision: bool = False
    seed: int = 42
    use_quant: bool = True
    do_sample: bool = True
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty = 1.5
    moe_module_name = 'block_sparse_moe'
    
    # log and model saving
    checkpoint_path: str = "./checkpoints"                          
    save_checkpoint: bool = False                                  
    resume_checkpoint: bool = True                                  
    save_lora_adapter: bool = True                                 
    save_merged_model: bool = False                                 
    save_best_model: bool = True                                    
    best_metric: str = "loss"                                       
    merged_model_save_path: str = "./merged_models"                 
    
    # dataset args
    data_preprocess_only: bool = False
    dataset_path: str = "./datasets"
    dataset_name: str = "mmlu"
    max_words_len: int = 1024

    # train args
    lora_save_path: str = "./LoRA_models"              
    num_train_epochs: int = 10                                      
    batch_size: int = 20                                            
    gradient_accumulation_steps: int = 1                            
    learning_rate: float = 1e-3                                     
    logging_dir: str = "./logs"                                    
    logging_steps: int = 1
    save_steps: int = 100                                           
    save_on_each_node: bool = True
    save_total_limit: int = 50                                      
    gradient_checkpointing: bool = True
    label_names: list[str] = field(default_factory=lambda: ["labels"])
    eval_strategy: str = "steps"
    eval_steps: int = 10000
    lr_scheduler_type: str = "reduce_lr_on_plateau"
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"factor": 0.95, "patience": 5, "threshold": 1e-3, "min_lr": 1e-5}) # field(default_factory=lambda: {"num_warmup_steps": 30})
    num_warmup_steps: int = 50
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    ds_config: str = "./configs/ds_config_zero2_no_offload.json"
    daft_threshold: float = 1.0 # 1.0 is the default value, means to fine-tune all layers

    # inference args
    expert_cache_size: int = 256
    cache_policy: str = "LRU"
    inference_batch: int = 20
    eval_dataset: str = "test.json"
    num_active_experts: int = 4
    chat_mode: bool = False
