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

This repository contains our code for the training-phase optimization (i.e., the locality-aware expert routing mechanism and the data-aware elastic fine-tuning strategy). We also release a portion of our evaluation data.

### [The entire codebase and sample data are available in our [Github repo](https://github.com/CFMoE/CFMoE.github.io#).]


## Codebase Organization

## Data Release
We provide a portion of the evaluation data using the WikiText dataset as an example [here](https://github.com/CFMoE/CFMoE.github.io/tree/main/Dataset). The shared data includes the input sequences (`Input`), the corresponding outputs from both the baseline and CFMoE models (`Output_Baseline` and `Output_CFMoE`), as well as their expert consistency rates (ECR) during inference (`ECR_Baseline` and `ECR_CFMoE`).

