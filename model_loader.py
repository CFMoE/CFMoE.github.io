from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
import torch

class ModelLoader:
    """Model loading class for CFMoE framework."""
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        
    def load_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        config = self.config

        model_config = AutoConfig.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        model_config.use_cache = False
        model_config.consecutive_expert_loss_type =  config.consecutive_expert_loss_type
        model_config.consecutive_expert_loss_weight = config.consecutive_expert_loss_weight
        model_config.aux_loss_alpha = config.aux_loss_alpha
        if hasattr(model_config, 'router_aux_loss_coef'):
            model_config.router_aux_loss_coef = config.aux_loss_alpha
            print("Aux loss coef: ", model_config.aux_loss_alpha, model_config.router_aux_loss_coef)

        if config.use_quant:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                config=model_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                config=model_config,
                trust_remote_code=True
            )
        
        self.model.enable_input_require_grads()
        self.model.generation_config.do_sample = config.do_sample
        self.model.generation_config.temperature = config.temperature
        self.model.generation_config.top_p = config.top_p
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.repetition_penalty = config.repetition_penalty

        if config.use_lora:
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=config.target_modules,
                lora_dropout=config.lora_dropout,
                modules_to_save = config.lora_extra_modules,
                use_rslora=True
            )
            self.model = get_peft_model(self.model, peft_config)
        print(self.model.print_trainable_parameters())

        try:
            self.model.generation_config.validate()
        except ValueError as e:
            print(f"Generation config validation failed: {e}")
