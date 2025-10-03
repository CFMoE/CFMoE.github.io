from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
import torch
import os
import pandas as pd


class ModelLoader:
    """Model loading class for CFMoE framework."""
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None

    def get_dynamic_target_modules(self, daft_df, daft_threshold, base_target_modules):
        """
        Generate dynamic target_modules based on ECR threshold for DeepSeek MoE model.
        
        Args:
            daft_df: DataFrame with 'layer_idx' and 'ECR' columns
            daft_threshold: ECR threshold for determining which layers to fine-tune
            base_target_modules: Base target modules list (e.g., ["q_proj", "k_proj", ...])
        
        Returns:
            List of target modules for layers with ECR < threshold
        """
        if daft_df is None:
            return base_target_modules
        
        # Get layers that need fine-tuning (ECR < threshold)
        layers_to_finetune = daft_df[daft_df['ECR'] < daft_threshold]['layer_idx'].tolist()
        
        # Detect the number of experts dynamically
        num_experts = self._detect_num_experts()
        
        # Generate target modules for these layers
        dynamic_target_modules = []
        for layer_idx in layers_to_finetune:
            for module in base_target_modules:
                if module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    # Attention modules
                    dynamic_target_modules.append(f"model.layers.{layer_idx}.self_attn.{module}")
                elif module in ["gate_proj", "up_proj", "down_proj"]:
                    # MoE expert modules - include all experts + shared_experts
                    # Add shared_experts
                    dynamic_target_modules.append(f"model.layers.{layer_idx}.mlp.shared_experts.{module}")
                    # Add all individual experts
                    for expert_idx in range(num_experts):
                        dynamic_target_modules.append(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{module}")
                elif module == "gate":
                    # MoE gate module
                    dynamic_target_modules.append(f"model.layers.{layer_idx}.mlp.gate")
        
        print(f"DAFT: {len(layers_to_finetune)} layers selected for fine-tuning out of {len(daft_df)} total layers")
        print(f"Selected layers: {layers_to_finetune}")
        print(f"Detected {num_experts} experts per MoE layer")
        
        return dynamic_target_modules

    def _detect_num_experts(self):
        """
        Dynamically detect the number of experts in the MoE model.
        
        Returns:
            Number of experts per MoE layer
        """
        if self.model is None:
            # Fallback: try to detect from model config or use default
            try:
                # Try to get from model config
                if hasattr(self.config, 'model_name'):
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(self.config.model_name, trust_remote_code=True)
                    if hasattr(config, 'num_experts'):
                        return config.num_experts
                    elif hasattr(config, 'num_local_experts'):
                        return config.num_local_experts
            except:
                pass
            
            # Default fallback
            print("Warning: Cannot detect number of experts, using default value 64")
            return 64
        
        try:
            # Find the first MoE layer and count experts
            for layer_idx in range(len(self.model.model.layers)):
                layer = self.model.model.layers[layer_idx]
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                    # Found MoE layer, count experts
                    num_experts = len(layer.mlp.experts)
                    print(f"Detected {num_experts} experts from layer {layer_idx}")
                    return num_experts
            
            # If no MoE layer found, use default
            print("Warning: No MoE layer found, using default value 64")
            return 64
            
        except Exception as e:
            print(f"Warning: Error detecting experts ({e}), using default value 64")
            return 64

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
            daft_threshold = config.daft_threshold
            offline_ecr_profile = os.path.join(config.model_name, "offline_ecr_profile.csv")
            if daft_threshold == 1.0 or (not os.path.exists(offline_ecr_profile)):
                target_modules = config.target_modules
            else:
                daft_df = pd.read_csv(offline_ecr_profile)
                # data-aware elastic fine-tuning
                target_modules = self.get_dynamic_target_modules(
                    daft_df, daft_threshold, config.target_modules
                )
            
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=target_modules,
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
