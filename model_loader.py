from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
import torch

class ModelLoader:
    """模型加载类"""
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
        self.model.generation_config.repetition_penalty = config.repetition_penalty # 新增：避免生成连续重复词汇

        if config.use_lora:
            # 根据实际模型结构调整，注意peft 不支持MoEGate()，仅支持 `torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, `transformers.pytorch_utils.Conv1D`
            # target_modules = [
            #         "q_proj", "k_proj", "v_proj", "o_proj",
            #         "gate_proj", "up_proj", "down_proj"
            # ]
            
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=config.target_modules,
                lora_dropout=config.lora_dropout,
                modules_to_save = config.lora_extra_modules,
                use_rslora=True # 新增：看看mmlu效果
            )
            self.model = get_peft_model(self.model, peft_config)
        print(self.model.print_trainable_parameters())

        try:
            self.model.generation_config.validate()
        except ValueError as e:
            print(f"生成配置验证失败: {e}")

"""
DeepseekForCausalLM(
(model): DeepseekModel(
    (embed_tokens): Embedding(102400, 2048)
    (layers): ModuleList(
    (0): DeepseekDecoderLayer(
        (self_attn): DeepseekSdpaAttention(
        (q_proj): Linear4bit(in_features=2048, out_features=2048, bias=False)
        (k_proj): Linear4bit(in_features=2048, out_features=2048, bias=False)
        (v_proj): Linear4bit(in_features=2048, out_features=2048, bias=False)
        (o_proj): Linear4bit(in_features=2048, out_features=2048, bias=False)
        (rotary_emb): DeepseekRotaryEmbedding()
        )
        (mlp): DeepseekMLP(
        (gate_proj): Linear4bit(in_features=2048, out_features=10944, bias=False)
        (up_proj): Linear4bit(in_features=2048, out_features=10944, bias=False)
        (down_proj): Linear4bit(in_features=10944, out_features=2048, bias=False)
        (act_fn): SiLU()
        )
        (input_layernorm): DeepseekRMSNorm()
        (post_attention_layernorm): DeepseekRMSNorm()
    )
    (1-27): 27 x DeepseekDecoderLayer(
        (self_attn): DeepseekSdpaAttention(
        (q_proj): Linear4bit(in_features=2048, out_features=2048, bias=False)
        (k_proj): Linear4bit(in_features=2048, out_features=2048, bias=False)
        (v_proj): Linear4bit(in_features=2048, out_features=2048, bias=False)
        (o_proj): Linear4bit(in_features=2048, out_features=2048, bias=False)
        (rotary_emb): DeepseekRotaryEmbedding()
        )
        (mlp): DeepseekMoE(
        (experts): ModuleList(
            (0-63): 64 x DeepseekMLP(
            (gate_proj): Linear4bit(in_features=2048, out_features=1408, bias=False)
            (up_proj): Linear4bit(in_features=2048, out_features=1408, bias=False)
            (down_proj): Linear4bit(in_features=1408, out_features=2048, bias=False)
            (act_fn): SiLU()
            )
        )
        (gate): MoEGate()
        (shared_experts): DeepseekMLP(
            (gate_proj): Linear4bit(in_features=2048, out_features=2816, bias=False)
            (up_proj): Linear4bit(in_features=2048, out_features=2816, bias=False)
            (down_proj): Linear4bit(in_features=2816, out_features=2048, bias=False)
            (act_fn): SiLU()
        )
        )
        (input_layernorm): DeepseekRMSNorm()
        (post_attention_layernorm): DeepseekRMSNorm()
    )
    )
    (norm): DeepseekRMSNorm()
)
(lm_head): Linear(in_features=2048, out_features=102400, bias=False)
)
"""

"""
SmallthinkerForCausalLM(
  (model): SmallthinkerModel(
    (embed_tokens): Embedding(151936, 1536)
    (layers): ModuleList(
      (0-31): 32 x SmallthinkerDecoderLayer(
        (self_attn): SmallthinkerAttention(
          (q_proj): Linear4bit(in_features=1536, out_features=1536, bias=False)
          (k_proj): Linear4bit(in_features=1536, out_features=256, bias=False)
          (v_proj): Linear4bit(in_features=1536, out_features=256, bias=False)
          (o_proj): Linear4bit(in_features=1536, out_features=1536, bias=False)
        )
        (block_sparse_moe): SmallthinkerMoeBlock(
          (primary_router): Linear4bit(in_features=1536, out_features=32, bias=False)
          (experts): ModuleList(
            (0-31): 32 x SmallthinkerHierarchicalMLP(
              (up): Linear4bit(in_features=1536, out_features=768, bias=False)
              (gate): Linear4bit(in_features=1536, out_features=768, bias=False)
              (down): Linear4bit(in_features=768, out_features=1536, bias=False)
            )
          )
        )
        (input_layernorm): SmallthinkerRMSNorm((1536,), eps=1e-06)
        (post_attention_layernorm): SmallthinkerRMSNorm((1536,), eps=1e-06)
      )
    )
    (norm): SmallthinkerRMSNorm((1536,), eps=1e-06)
    (rotary_emb): SmallthinkerRotaryEmbedding()
  )
  (lm_head): Linear(in_features=1536, out_features=151936, bias=False)
)
"""