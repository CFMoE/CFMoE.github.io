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
            eval_accumulation_steps         = self.config.gradient_accumulation_steps, # 新增：需要设置不然会导致OFM错误
            lr_scheduler_type               = self.config.lr_scheduler_type,
            lr_scheduler_kwargs             = self.config.lr_scheduler_kwargs,
            warmup_steps                    = self.config.num_warmup_steps,
            # load_best_model_at_end          = self.config.load_best_model_at_end,
            # metric_for_best_model           = self.config.metric_for_best_model,
            # greater_is_better               = self.config.greater_is_better,
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
    
    # 新增：为了计算eval metrics
    def _compute_met_func(self, eval_preds, compute_result: bool = True):
        # Not implement now #
        return {}
        
        logits, labels = eval_preds.predictions[:1], eval_preds.label_ids[:1]
        # In case the model returns more than the prediction logits
        if isinstance(logits, tuple):
            logits = logits[0]

        preds = np.argmax(logits, axis=-1)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=True)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.processing_class.pad_token_id)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=True)
        
        start_idx = 0
        for idx in range(len(labels[0])):
            if labels[0][idx] != 151643:
                print('start from ', idx)
                start_idx = idx - 1
                break
            
        print('Preds and labels: ', preds[0][start_idx:500], labels[0][start_idx:500])
        print('Test samples: ', decoded_preds, decoded_labels)
        return {}
        
        ## Compute the metrics ##
        rouge_score = evaluate.load("./metrics/rouge")
        # 1.ROUGE
        # decoded_preds_rouge = ["\n".join(sent_tokenize(pred.strip().lower())) for pred in decoded_preds] # TODO：后续考虑加入sentence预处理操作，ROUGE expects a newline after each sentence
        # decoded_labels_rouge = ["\n".join(sent_tokenize(label.strip().lower())) for label in decoded_labels]
        
        decoded_preds_rouge = [pred.strip().lower() for pred in decoded_preds]
        decoded_labels_rouge = [label.strip().lower() for label in decoded_labels]
        
        result_rouge = rouge_score.compute(predictions=decoded_preds_rouge, references=decoded_labels_rouge, use_stemmer=True)
        result_rouge = {key: round(value.mid.fmeasure * 100, 4) for key, value in result_rouge.items()}
        
        # 2. ACC
        acc = np.mean([int(x==y) for x, y in zip(decoded_preds_rouge, decoded_labels_rouge)]) * 100
        acc = round(acc, 4)
        result_rouge["accuracy"] = acc
        
        return result_rouge

    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     with torch.no_grad():  # 如果仅计算输出无需梯度
    #         outputs = model(**inputs)
    #     loss = outputs.loss
    #     print(f"batch_size: {num_items_in_batch}, loss: {loss}")
    #     del outputs  # 显式删除 outputs
    #     torch.cuda.empty_cache()
    #     return (loss, outputs) if return_outputs else loss
    
    # def training_step(self, model, inputs, num_items_in_batch=None):
    #     print(f"Before step, GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    #     loss = super().training_step(model, inputs)
    #     torch.cuda.empty_cache()  # 释放显存缓存
    #     current_step = self.state.global_step
    #     if current_step % self.args.logging_steps == 0:
    #         print(f"Step {current_step}: train_loss = {loss.item():.4f}, GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    #     return loss

    # def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
    #     """重写 evaluate 方法以计算困惑度"""
    #     output = super().evaluate(self.eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
    #     eval_loss = output["eval_loss"]
    #     perplexity = np.exp(eval_loss)
    #     output["eval_perplexity"] = perplexity
    #     return output