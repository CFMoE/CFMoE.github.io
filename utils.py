import torch
import argparse
import evaluate

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_arguments():
    parser = argparse.ArgumentParser(description='extra args for experiments')

    parser.add_argument(
        "--aux_loss_alpha",
        type=float,
        default=0,
        help="Deep Fake Detection using Custom Dataset"
    )

    parser.add_argument(
        "--consecutive_expert_loss_weight",
        type=float,
        default=1.0,
        help="Weight of auxiliary consecutive expert loss"
    )
    
    parser.add_argument(
        "--consecutive_expert_loss_type",
        type=str,
        default="js",
        help="Distance function to calculate the distribution difference between consecutive tokens"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Deep Fake Detection using Custom Dataset"
    )
    
    parser.add_argument(
        "--lora_save_path",
        type=str,
        default="./LoRA_models",
        help="Deep Fake Detection using Custom Dataset"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mmlu",
        help="Dataset used for train and eval"
    )
    
    parser.add_argument(
        "--model_key",
        type=str,
        default="deepseek16b-cel",
        help="Indicator for the target model"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient descent every * steps."
    )
    
    parser.add_argument(
        "--use_lora",
        default=True,
        action='store_false',
        help="Use Lora or not."
    )
    
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="Lora rank parameter."
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Lora alpha parameter."
    )
    
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Train epoch."
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Batch size."
    )
    
    parser.add_argument(
        "--max_words_len",
        type=int,
        default=1024,
        help="Max words len."
    )
    
    parser.add_argument(
        "--daft_threshold",
        type=float,
        default=1.0,
        help="DAFT threshold."
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="./models/smallthinker-moe-4b-thinker",
        help="Path to the original model"
    )
    
    parser.add_argument(
        "--moe_module_name",
        type=str,
        default="block_sparse_moe",
        help="Name of the moe module."
    )
    
    parser.add_argument(
        "--lora_extra_modules",
        type=str,
        default="",
        help="."
    )
    
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,up,gate,down,primary_router",
        help="."
    )
    
    args, unknown = parser.parse_known_args()
    return args

def compute_metrics(eval_pred):
    """Compute metrics for trainer."""
    rouge_score = evaluate.load("rouge")
    