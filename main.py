from config import Config
from model_loader import ModelLoader
from data_preparer import DataPreparer
from trainer import Trainer
from utils import set_seed
# from inference import Inference
from logger import setup_distributed_train_logging
import torch.distributed
from torch.distributed import init_process_group
from datetime import timedelta
import os
import sys
import argparse
from utils import get_arguments

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
    config.logging_dir = f"./logs{model_key}/{config.dataset_name}/lr_{config.learning_rate}_aux_{config.aux_loss_alpha}_type_{config.consecutive_expert_loss_type}_weight_{config.consecutive_expert_loss_weight}_bs_{batch_size}_rank_{lora_rank}"

class FineTuner:
    def __init__(self, config):
        self.config = config
        set_seed(self.config.seed)

        model_loader = ModelLoader(self.config)

        model_loader.load_tokenizer()
        self.tokenizer = model_loader.tokenizer

        self.data_preparer = DataPreparer(self.config, self.tokenizer)
        train_dataset = self.data_preparer.load_data(self.config.dataset_path, self.config.dataset_name, "train")
        eval_dataset = self.data_preparer.load_data(self.config.dataset_path, self.config.dataset_name, "eval")
        self.train_dataloader = self.data_preparer.get_dataloader(train_dataset)
        self.eval_dataloader = self.data_preparer.get_dataloader(eval_dataset, shuffle=False)

        if self.config.data_preprocess_only:
            sys.exit(0)

        model_loader.load_model()
        self.model = model_loader.model

        print('Special token of Model: ', self.model.generation_config.bos_token_id, self.model.generation_config.eos_token_id, self.model.generation_config.pad_token_id)
        print('Special token of Tokenizer: ', self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
        
        self.trainer = Trainer(self.config, self.model, self.tokenizer, self.train_dataloader, self.eval_dataloader)

        # self.inference = Inference(self.config)

    def train(self):
        lora_save_path = self.config.lora_save_path
        if os.path.exists(lora_save_path) and len(os.listdir(lora_save_path)) > 0:
            self.trainer.train(resume_from_checkpoint=self.config.resume_checkpoint)
        else:
            self.trainer.train()
        if self.trainer.is_world_process_zero():
            self.save_models()
        torch.distributed.barrier() # hauser - 类似于wait的作用

        # self.inference = Inference(self.config, adapter_path=self.config.lora_save_path)

    def save_models(self):
        try:
            if self.config.save_lora_adapter:
                lora_save_path = self.config.lora_save_path
                self.model.save_pretrained(lora_save_path)
                print(f"Saving LoRA adapter to {lora_save_path}")

            if self.config.save_merged_model and isinstance(self.model, PeftModelForCausalLM):
                merged_model = self.model.merge_and_unload()
                merged_model_save_path = self.config.merged_model_save_path
                merged_model.save_pretrained(merged_model_save_path)
                print(f"Saving merged model to {merged_model_save_path}")
            else:
                print("Skip merge LoRA adapters.")
        except Exception as e:
            print(f"Error saving models: {e}")

    def evaluate(self):
        eval_results = self.trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        print(f"Perplexity: {eval_results['eval_perplexity']:.4f}")

    def infer(self, instruction, input_text):
        return self.inference.generate(instruction, input_text)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TORCH_CUDA_ARCH_LIST'] = '' # set this to avoid warnings
    config = Config()
    new_config = get_arguments()
    update_config(config, new_config)
    print(f"after update, aux_loss_alpha: {config.aux_loss_alpha}, consecutive_expert_loss_weight: {config.consecutive_expert_loss_weight}")
    timeout_long_ncll = timedelta(seconds=600000)  # 100 minutes
    init_process_group(backend="nccl", timeout=timeout_long_ncll) # hauser - for timeout error
    
    fine_tuner = FineTuner(config)
    # fine_tuner.evaluate()
    
    fine_tuner.train()

    # fine_tuner.evaluate()
