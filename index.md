![license](https://img.shields.io/badge/Platform-Android-green "Android")
![license](https://img.shields.io/badge/Version-Beta-yellow "Version")
![license](https://img.shields.io/badge/Licence-Apache%202.0-blue.svg "Apache")


## Table of Contents
[Introduction](#introduction)

[Codebase Organization](#codebase-organization)

[Data Release](#data-release)

## Introduction

Deploying large language models (LLMs) on edge devices has garnered significant attention due to their inherent privacy and security benefits. However, the enormous parameter scale of modern LLMs poses fundamental challenges for resource-constrained edge devices. Mixture-of-Experts (MoE) models emerge as a promising solution by dynamically activating only a subset of experts during inference, and thus enhancing computational efficiency. Nevertheless, the runtime uncertainty in expert activation still leads to high memory demands during inference. To address this, prior works concentrate on either compressing or offloading experts, all striving around the tradeoffs among the model precision, memory consumption, and serving throughput and thus forming an “impossible triangle”.

To re-conciliate the “impossible triangle”, we propose CFMoE, a Cache-Friendly MoE serving system that holistically integrates optimizations spanning the model architecture, fine-tuning strategy, and inference engine. The key insight behind CFMoE is that the edge-deployed MoE serving system needs to strengthen and leverage the locality of expert routing in order to break through the theoretical caching ceiling caused by the inherent load-balancing constraints in MoE models. Specifically, CFMoE integrates three novel techniques: 1) locality-aware expert routing mechanism, 2) data-aware elastic fine-tuning strategy, and 3) prefill-decoding disaggregated caching management. Evaluation results from state-of-the-art MoE models demonstrate that CFMoE significantly reduces memory usage by 35.7% on average without decreasing serving throughput or degrading model precision.

This repository contains our code for training and serving MoE models using CFMoE. We also release a portion of our evaluation data.

### [The entire codebase and sample data are available in our [Github repo](https://github.com/CFMoE/CFMoE.github.io#).]


## Codebase Organization

The CFMoE codebase is organized into two main components:

**Training (`main.py`)**: Fine-tunes MoE models using CFMoE's locality-aware expert routing and data-aware elastic fine-tuning strategies. The training script supports distributed training with DeepSpeed, and configuration parameters can be provided both via `config.py` and through command-line arguments.

**Inference (`inference.py`)**: We provide an inference engine to simulate the serving process of fine-tuned MoE models. The inference engine not only evaluates model performance and measures expert consistency rates (ECR) during serving, but also records cache hit rates (CHR) under various cache sizes and cache replacement policies, including OPT, LRU, LFU, and LOPT.

Both components use `config.py` as the central configuration file and provide `argparse` interfaces for command-line parameter customization.

### File Tree

The CFMoE codebase is organized as follows:

```
├── main.py
├── inference.py
├── config.py
├── trainer.py
├── model_loader.py
├── data_preparer.py
├── utils.py
├── logger.py
├── requirements.txt
├── wikitext_evaluation.csv
├── README.md
├── LICENSE
├── datasets/
│   ├── alpaca/
│   ├── aqua/
│   ├── boolq/
│   ├── common-v1/
│   ├── gsm8k/
│   ├── mathqa/
│   ├── mawps/
│   ├── mmlu/
│   ├── obqa/
│   ├── piqa/
│   ├── samsum/
│   ├── siqa/
│   └── wikitext/
└── models/
    └── deepseek-moe-16b-chat/
```

The `models/` directory contains the MoE models. 
For example, we have modified the DeepSeek MoE model in 
`models/deepseek-moe-16b-chat/` to include locality 
profiling modules.

### Usage

**Installation**:
```bash
# Create a new conda environment
conda create --name cfmoe python=3.12
conda activate cfmoe

# Install required dependencies
pip install -r requirements.txt
```

**Training**:
```bash
deepspeed --master_port 29505 main.py \
    --aux_loss_alpha 0.0 \
    --consecutive_expert_loss_weight 0.001 \
    --consecutive_expert_loss_type 'js' \
    --learning_rate 0.0001 \
    --lora_save_path ./LoRA_models_dataset_mmlu \
    --dataset_name mmlu \
    --model_key "_deepseek_16b" \
    --model_name "./models/deepseek-moe-16b-chat" \
    --moe_module_name "mlp" \
    --lora_extra_modules "gate" \
    --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --max_words_len 512
```

**Inference**:
```bash
torchrun --standalone --nnodes=1 --master_port=29505 --nproc_per_node=1 inference.py \
    --aux_loss_alpha 0.0 \
    --consecutive_expert_loss_weight 0.001 \
    --consecutive_expert_loss_type 'js' \
    --lora_save_path ./LoRA_models_dataset_mmlu/checkpoint-1000 \
    --dataset_name mmlu \
    --model_key "_deepseek_16b" \
    --model_name "./models/deepseek-moe-16b-chat" \
    --moe_module_name "mlp" \
    --lora_extra_modules "gate" \
    --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --max_words_len 512
```

**Training Monitoring**:
```bash
# Start TensorBoard to monitor training progress
tensorboard --logdir=./logs${model_key} --port 6006 --bind_all
```

Access the training dashboard by opening `http://server_ip:6006/` in your browser, where `server_ip` is your server's IP address. Choose a custom port to avoid conflicts with other services.

### Key Parameters

- **`model_name`**: Base model path. We use DeepSeek-MoE as an example, where we have modified the model code in `./models/deepseek-moe-16b-chat` to add locality profiling modules
- **`dataset_name`**: Target task dataset. We provide 12 standard datasets consistent with the paper, including: alpaca, aqua, boolq, gsm8k, mathqa, mawps, mmlu, obqa, piqa, samsum, siqa, wikitext
- **`consecutive_expert_loss_type`** and **`consecutive_expert_loss_weight`**: Parameters that control cache affinity constraints to optimize expert routing locality
- **`model_key`**: Model identifier used to distinguish between different model configurations
- **`moe_module_name`**: MoE module name specifying which modules need CFMoE optimization (e.g., "mlp" for `deepseek-moe-16b`)
- **`lora_extra_modules`**: Additional LoRA modules specifying extra modules that require LoRA adaptation
- **`target_modules`**: LoRA target modules specifying the exact module list that requires LoRA adaptation
- **`aux_loss_alpha`**: Auxiliary loss weight for load balancing constraint
- **`daft_threshold`**: Threshold for the data-aware elastic fine-tuning strategy


## Data Release
We provide a portion of the evaluation data using the WikiText dataset as an example [here](https://github.com/CFMoE/CFMoE.github.io/tree/main/wikitext_evaluation.csv). The shared data includes the input sequences (`Input`), the corresponding outputs from both the baseline and CFMoE models (`Output_Baseline` and `Output_CFMoE`), as well as their expert consistency rates (ECR) during inference (`ECR_Baseline` and `ECR_CFMoE`).

