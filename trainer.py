import torch
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from peft import PeftModelForCausalLM

from transformers import Trainer as HFTrainer, TrainingArguments, DataCollatorForSeq2Seq
import evaluate
from nltk.tokenize import sent_tokenize

class Trainer(HFTrainer):
    """Custom trainer class extending HuggingFace Trainer for CFMoE fine-tuning."""
    def __init__(self, config, model, tokenizer, train_dataloader, eval_dataloader):
        self.config = config
        self.global_step = 0
        self.best_loss = float('inf')

        training_args = TrainingArguments(
            output_dir                      = self.config.lora_save_path,
            num_train_epochs                = self.config.num_train_epochs,
            per_device_train_batch_size     = self.config.batch_size,
            per_device_eval_batch_size      = self.config.batch_size // 4,
            gradient_accumulation_steps     = self.config.gradient_accumulation_steps,
            learning_rate                   = self.config.learning_rate,
            logging_dir                     = self.config.logging_dir,
            logging_steps                   = self.config.logging_steps,
            save_steps	                    = self.config.save_steps,
            save_on_each_node               = self.config.save_on_each_node,
            save_total_limit                = self.config.save_total_limit,
	        gradient_checkpointing          = self.config.gradient_checkpointing,
            label_names                     = self.config.label_names,
            eval_strategy                   = self.config.eval_strategy,
            eval_steps                      = self.config.eval_steps,
            eval_accumulation_steps         = self.config.gradient_accumulation_steps,
            lr_scheduler_type               = self.config.lr_scheduler_type,
            lr_scheduler_kwargs             = self.config.lr_scheduler_kwargs,
            warmup_steps                    = self.config.num_warmup_steps,
            deepspeed                       = self.config.ds_config
        )

        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=eval_dataloader.dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
            processing_class=tokenizer,
            compute_metrics=self._compute_met_func
        )
    
    def _compute_met_func(self, eval_preds, compute_result: bool = True):
        return {}
