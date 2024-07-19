# A Survey on LoRA of Large Language Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of papers and resources about LoRA of Large Language Models  based on our survey paper: [A Survey on LoRA of Large Language Models](https://arxiv.org/abs/2407.11046). 

**This repo will be continuously updated. Don't forget to star <img src="./fig/star.svg" width="15" height="15" /> it and keep tuned!**

**Please cite the paper in [Citations](#citations) if you find the resource helpful for your research. Thanks!**

![lora](./fig/lora.svg)

## LoRA of LLMs

Low-Rank Adaptation(LoRA), which updates the dense neural network layers with pluggable low-rank matrices, is one of the best performed parameter efficient fine-tuning paradigms. Furthermore, it has significant advantages in cross-task generalization and privacy-preserving. Hence, LoRA has gained much attention recently, and the number of related literature demonstrates exponential growth. It is necessary to conduct a comprehensive overview of the current progress on LoRA. This survey categorizes and reviews the progress from the perspectives of (1) downstream adaptation improving variants that improve LoRA's performance on downstream tasks; (2) cross-task generalization methods that mix multiple LoRA plugins to achieve cross-task generalization; (3) efficiency-improving methods that boost the computation-efficiency of LoRA; (4) data privacy-preserving methods that use LoRA in federated learning; (5) application. Besides, this survey also discusses the future directions in this field.

## Contents

- [A Survey on LoRA of Large Language Models ](#a-survey-on-lora-of-large-language-models-)

  - [LoRA of LLMs](#lora-of-llms)

  - [Low-Rank Adaptation](#low-rank-adaptation)
    - [Theoretical Analysis](#theoretical-analysis)
    - [Beyond Fine-tuning](#beyond-fine-tuning)

  - [Downstream Adaptation Improving](#downstream-adaptation-improving)
    - [Breaking the Low-rank Bottleneck](#breaking-the-low-rank-bottleneck)
      - [Stacking LoRAs along Fine-tuning](#stacking-loras-along-fine-tuning)
      - [Updating as gradient compressor](#updating-as-gradient-compressor)
      - [Co-learning LLM and LoRA](#co-learning-llm-and-lora)
    - [Dynamic Rank Allocation](#dynamic-rank-allocation)
      - [SVD-Based Methods](#svd-based-methods)
      - [SRD-based Methods](#srd-based-methods)
      - [Rank Sampling-based Methods](#rank-sampling-based-methods)
    - [Optimizing the Learning Procedure](#optimizing-the-learning-procedure)
      - [Initialization Improvement](#initialization-improvement)
      - [Gradient Update Optimization](#gradient-update-optimization)
      - [Overfitting Mitigation](#overfitting-mitigation)
    - [Combining with other Learning Paradigms](#combining-with-other-learning-paradigms)

  - [Cross-task Generalization](#cross-task-generalization)
    - [Mixture with Manually Designed Weights](#mixture-with-manually-designed-weights)
    - [Mixture with Learnt Weights](#mixture-with-learnt-weights)
    - [Mixture of LoRA Experts](#mixture-of-lora-experts)

  - [Efficiency Improving](#efficiency-improving)
    - [Parameter Reduction](#parameter-reduction)
      - [Parameter Freezing](#parameter-freezing)
      - [Parameter Pruning](#parameter-pruning)
      - [Parameter Sharing](#parameter-sharing)
    - [Parameter Quantization](#parameter-quantization)
      - [PTQ-based methods](#ptq-based-methods)
      - [QAT-based methods](#qat-base-methods)
    - [Parallel LoRA Computing Frameworks](#parallel-lora-computing-frameworks)
      - [Parallel Fine-tuning](#parallel-fine-tuning)
      - [Parallel Inference](#parallel-inference)

  - [LoRA for Federate Learning](#lora-for-federate-learning)
    - [Data Heterogeneity](#data-heterogeneity)
    - [Device Heterogeneity](#device-heterogeneity)
    - [Model Heterogeneity](#model-heterogeneity)
    - [Parameter Privacy](#parameter-privacy)

  - [Applications of LoRA](#applications-of-lora)
    - [Language Tasks](#language-tasks)
    - [Vision Task](#vision-task)
    - [Multimodal Tasks](#multimodal-tasks)

## Low-Rank Adaptation

### <img src="./fig/star.svg" width="15" height="15" /> Theoretical Analysis



### <img src="./fig/star.svg" width="15" height="15" /> Beyond Fine-tuning



## Downstream Adaptation Improving

### <img src="./fig/star.svg" width="15" height="15" /> Breaking the Low-rank Bottleneck

#### Stacking LoRAs along Fine-tuning

1. **样例Can Language Models Solve Graph Problems in Natural Language?** `preprint`

   *Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, Yulia Tsvetkov.* [[PDF](https://browse.arxiv.org/pdf/2305.10037.pdf)] [[Code](https://github.com/Arthur-Heng/NLGraph)], 2023.5, ![](https://img.shields.io/badge/DecoderOnly-blue) ![](https://img.shields.io/badge/LLM-red)

2. **样例Knowledge Graph Prompting for Multi-Document Question Answering.** `AAAI 2024`

   *Yu Wang, Nedim Lipka, Ryan Rossi, Alex Siu, Ruiyi Zhang, Tyler Derr.* [[PDF](https://arxiv.org/abs/2308.11730)] [[Code](https://github.com/YuWVandy/KG-LLM-MDQA)], 2023.8, ![](https://img.shields.io/badge/DecoderOnly-blue) ![](https://img.shields.io/badge/LLM-red)



#### Updating as gradient compressor



#### Co-learning LLM and LoRA

### <img src="./fig/star.svg" width="15" height="15" /> Dynamic Rank Allocation

#### SVD-Based Methods



#### SRD-Based Methods



#### Rank Sampling-Based Methods



### <img src="./fig/star.svg" width="15" height="15" /> Optimizing the Learning Procedure

#### Initialization Improvement



#### Gradient Update Optimization



#### Overfitting Mitigation



### <img src="./fig/star.svg" width="15" height="15" /> Combining with other Learning Paradigms



## Cross-task Generalization

### <img src="./fig/star.svg" width="15" height="15" /> Mixture with Manually Designed Weights



### <img src="./fig/star.svg" width="15" height="15" /> Mixture with Learnt Weights



### <img src="./fig/star.svg" width="15" height="15" /> Mixture of LoRA Experts



## Efficiency Improving

### <img src="./fig/star.svg" width="15" height="15" /> Parameter Reduction

#### Parameter Freezing



#### Parameter Pruning



#### Parameter Sharing



### <img src="./fig/star.svg" width="15" height="15" /> Parameter Quantization

#### PTQ-Based Methods



#### QAT-Based Methods



### <img src="./fig/star.svg" width="15" height="15" /> Parallel LoRA Computing Frameworks

#### Parallel Fine-tuning



#### Parallel Inference



## LoRA for Federated Learning

### <img src="./fig/star.svg" width="15" height="15" /> Data Heterogeneity



### <img src="./fig/star.svg" width="15" height="15" /> Device Heterogeneity



### <img src="./fig/star.svg" width="15" height="15" /> Model Heterogeneity



### <img src="./fig/star.svg" width="15" height="15" /> Parameter Privacy



## Applications of LoRA

### <img src="./fig/star.svg" width="15" height="15" /> Language Tasks



### <img src="./fig/star.svg" width="15" height="15" /> Vision Tasks



### <img src="./fig/star.svg" width="15" height="15" /> Multimodal Tasks



## Contribution

Contributions to this repository are welcome!

If you find any error or have relevant resources, feel free to open an issue or a pull request.

Paper format:

```
1. **[paper title].** `[]`

    *[authors].* [[PDF]([pdf link])] [[Code]([code link])], published time, ![](https://img.shields.io/badge/[architecture]-blue) ![](https://img.shields.io/badge/[size]-red)
```

## Citations

Please cite the following paper if you find the resource helpful for your research.

```
@article{,
  title={A Survey on LoRA of Large Language Models},
  author={},
  journal={arXiv preprint arXiv:2407.11046},
  year={2024}
}
```


Thank you for your support!
