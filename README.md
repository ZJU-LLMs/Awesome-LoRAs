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
  - [Contents](#contents)
  - [Low-Rank Adaptation](#low-rank-adaptation)
    - [ Theoretical Analysis](#-theoretical-analysis)
    - [ Beyond Fine-tuning](#-beyond-fine-tuning)
  - [Downstream Adaptation Improving](#downstream-adaptation-improving)
    - [ Breaking the Low-rank Bottleneck](#-breaking-the-low-rank-bottleneck)
      - [Stacking LoRAs along Fine-tuning](#stacking-loras-along-fine-tuning)
      - [Updating as gradient compressor](#updating-as-gradient-compressor)
      - [Co-learning LLM and LoRA](#co-learning-llm-and-lora)
    - [ Dynamic Rank Allocation](#-dynamic-rank-allocation)
      - [SVD-Based Methods](#svd-based-methods)
      - [SRD-Based Methods](#srd-based-methods)
      - [Rank Sampling-Based Methods](#rank-sampling-based-methods)
    - [ Optimizing the Learning Procedure](#-optimizing-the-learning-procedure)
      - [Initialization Improvement](#initialization-improvement)
      - [Gradient Update Optimization](#gradient-update-optimization)
      - [Overfitting Mitigation](#overfitting-mitigation)
    - [ Combining with other Learning Paradigms](#-combining-with-other-learning-paradigms)
  - [Cross-task Generalization](#cross-task-generalization)
    - [ Mixture with Manually Designed Weights](#-mixture-with-manually-designed-weights)
    - [ Mixture with Learnt Weights](#-mixture-with-learnt-weights)
    - [ Mixture of LoRA Experts](#-mixture-of-lora-experts)
  - [Efficiency Improving](#efficiency-improving)
    - [ Parameter Reduction](#-parameter-reduction)
      - [Parameter Freezing](#parameter-freezing)
      - [Parameter Pruning](#parameter-pruning)
      - [Parameter Sharing](#parameter-sharing)
    - [ Parameter Quantization](#-parameter-quantization)
      - [PTQ-Based Methods](#ptq-based-methods)
      - [QAT-Based Methods](#qat-based-methods)
    - [ Parallel LoRA Computing Frameworks](#-parallel-lora-computing-frameworks)
      - [Parallel Fine-tuning](#parallel-fine-tuning)
      - [Parallel Inference](#parallel-inference)
  - [LoRA for Federated Learning](#lora-for-federated-learning)
    - [Data Heterogeneity](#data-heterogeneity)
    - [ Device Heterogeneity](#-device-heterogeneity)
    - [ Model Heterogeneity](#-model-heterogeneity)
    - [ Parameter Privacy](#-parameter-privacy)
  - [Applications of LoRA](#applications-of-lora)
    - [ Language Tasks](#-language-tasks)
      - [Traditional NLP Task](#traditional-nlp-task)
      - [Code Task](#code-task)
      - [Model Alignment Task](#model-alignment-task)
      - [Vertical Domain Task](#vertical-domain-task)
    - [ Vision Tasks](#-vision-tasks)
      - [Image Generation Tasks](#image-generation-tasks)
      - [Image Segmentation Task](#image-segmentation-task)
    - [ Multimodal Tasks](#-multimodal-tasks)
      - [Audio-Text](#audio-text)
      - [Image-Text](#image-text)
      - [Video-Text](#video-text)
  - [Contribution](#contribution)
  - [Citations](#citations)

## Low-Rank Adaptation

1. **LoRA: Low-Rank Adaptation of Large Language Models.** `ICLR`  
   *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen* [[PDF](https://arxiv.org/abs/2106.09685)] [[Code](https://github.com/microsoft/LoRA)], 2022

### <img src="./fig/star.svg" width="15" height="15" /> Theoretical Analysis

1. **A Kernel-Based View of Language Model Fine-Tuning.** `ICML`  
    *Malladi S., Wettig A., Yu D., Chen D., Arora S.* [[PDF](https://arxiv.org/abs/2210.05643)] [[Code](https://github.com/princeton-nlp/LM-Kernel-FT)], 2023

2. **The Impact of LoRA on the Emergence of Clusters in Transformers.** `preprint`  
    *Koubbi H., Boussard M., Hernandez L.* [[PDF](https://arxiv.org/abs/2402.15415)] [[Code](https://github.com/HugoKoubbi/Transformers-2024-LoRA)], 2024

3. **LoRA Training in the NTK Regime Has No Spurious Local Minima.** `preprint`  
    *Jang U., Lee J. D., Ryu E. K.* [[PDF](https://arxiv.org/abs/2402.11867)] [[Code](https://github.com/UijeongJang/LoRA-NTK)], 2024

4. **Asymmetry in Low-Rank Adapters of Foundation Models.** `preprint`  
     *Zhu J., Greenewald K. H., Nadjahi K., Ocáriz Borde d H. S., Gabrielsson R. B., Choshen L., Ghassemi M., Yurochkin M., Solomon J.* [[PDF](https://arxiv.org/abs/2402.16842)] [[Code](https://github.com/Jiacheng-Zhu-AIML/AsymmetryLoRA)], 2024

5. **The Expressive Power of Low-Rank Adaptation.** `preprint`       
     *Zeng Y., Lee K.* [[PDF](https://arxiv.org/abs/2310.17513)] [[Code](https://github.com/UW-Madison-Lee-Lab/Expressive_Power_of_LoRA)], 2023

### <img src="./fig/star.svg" width="15" height="15" /> Beyond Fine-tuning

1. **ReLoRA: High-rank training through low-rank updates.** `NeurIPS Workshop`.  
  *Lialin V, Muckatira S, Shivagunde N, Rumshisky A.* [[PDF](https://arxiv.org/abs/2307.05695)] [[Code](https://github.com/Guitaricet/relora)], 2023

2. **MoRA: High-rank updating for parameter-efficient fine-tuning.** `preprint`  
  *Jiang T, Huang S, Luo S, Zhang Z, Huang H, Wei F, Deng W, Sun F, Zhang Q, Wang D, others.* [[PDF](https://arxiv.org/abs/2405.12130)] [[Code](https://github.com/kongds/MoRA)], 2024

3. **Training neural networks from scratch with parallel low-rank adapters.** `preprint`  
  *Huh M, Cheung B, Bernstein J, Isola P, Agrawal P.* [[PDF](https://arxiv.org/abs/2402.16828)] [[Code](https://minyoungg.github.io/LTE/)], 2024

4. **InfLoRA: Interference-free low-rank adaptation for continual learning.** `preprint`  
   *Liang Y, Li W.* [[PDF](https://arxiv.org/abs/2404.00228)] [[Code](https://github.com/liangyanshuo/InfLoRA)], 2024
   
5. **GS-LoRA: Continual forgetting for pre-trained vision models.** `preprint`  
   *Zhao H, Ni B, Wang H, Fan J, Zhu F, Wang Y, Chen Y, Meng G, Zhang Z.* [[PDF](https://arxiv.org/abs/2403.11530)] [[Code](https://github.com/bjzhb666/GS-LoRA)], 2024

6. **I-LoRA: Analyzing and reducing catastrophic forgetting in parameter-efficient tuning.** `preprint`  
   *Ren W, Li X, Wang L, Zhao T, Qin W.* [[PDF](https://arxiv.org/abs/2402.18865)] [[Code](https://github.com/which47/LLMCL)], 2024

7. **LongLoRA: Efficient fine-tuning of long-context large language models.** `preprint`  
   *Y. Chen, S. Qian, H. Tang, X. Lai, Z. Liu, S. Han, J. Jia.* [[PDF](https://arxiv.org/abs/2309.12307)] [[Code](https://github.com/dvlab-research/LongLoRA)], 2023

8. **SinkLoRA: Enhanced efficiency and chat capabilities for long-context large language models.** `preprint`  
   *Zhang H.* [[PDF](https://arxiv.org/abs/2406.05678)] [[Code](https://github.com/Dexter-GT-86/SinkLoRA)], 2023

## Downstream Adaptation Improving

### <img src="./fig/star.svg" width="15" height="15" /> Breaking the Low-rank Bottleneck

#### Stacking LoRAs along Fine-tuning

<!-- 1. **样例Can Language Models Solve Graph Problems in Natural Language?** `preprint`

   *Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, Yulia Tsvetkov.* [[PDF](https://browse.arxiv.org/pdf/2305.10037.pdf)] [[Code](https://github.com/Arthur-Heng/NLGraph)], 2023.5, ![](https://img.shields.io/badge/DecoderOnly-blue) ![](https://img.shields.io/badge/LLM-red)

2. **样例Knowledge Graph Prompting for Multi-Document Question Answering.** `AAAI 2024`

   *Yu Wang, Nedim Lipka, Ryan Rossi, Alex Siu, Ruiyi Zhang, Tyler Derr.* [[PDF](https://arxiv.org/abs/2308.11730)] [[Code](https://github.com/YuWVandy/KG-LLM-MDQA)], 2023.8, ![](https://img.shields.io/badge/DecoderOnly-blue) ![](https://img.shields.io/badge/LLM-red) -->

1. **ReLoRA: High-Rank Training Through Low-Rank Updates.** `NeurIPS Workshop`  
   *Lialin V., Muckatira S., Shivagunde N., Rumshisky A.* [[PDF](https://arxiv.org/abs/2307.05695)] [[Code](https://github.com/Guitaricet/relora)], 2023

2. **Chain of LoRA: Efficient fine-tuning of language models via residual learning.** `preprint`  
   *Xia W, Qin C, Hazan E.* [[PDF](https://arxiv.org/abs/2401.04151)], 2024

3. **Mini-ensemble low-rank adapters for parameter-efficient fine-tuning.** `preprint`  
   *Ren P, Shi C, Wu S, Zhang M, Ren Z, Rijke d M, Chen Z, Pei J.* [[PDF](https://arxiv.org/abs/2402.17263)] [[Code](https://https://github.com/ChasonShi/MELoRA)], 2024

#### Updating as gradient compressor

1. **FLoRA: Low-rank adapters are secretly gradient compressors.** `preprint`  
   *Hao Y, Cao Y, Mou L.* [[PDF](https://arxiv.org/abs/2402.03293)] [[Code](https://github.com/BorealisAI/flora-opt)], 2024

#### Co-learning LLM and LoRA

1. **Delta-LoRA: Fine-tuning high-rank parameters with the delta of low-rank matrices.** `preprint`
    *Zi B, Qi X, Wang L, Wang J, Wong K, Zhang L.* [[PDF](https://arxiv.org/abs/2309.02411)], 2023

### <img src="./fig/star.svg" width="15" height="15" /> Dynamic Rank Allocation

#### SVD-Based Methods

1. **AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.** `ICLR 2023`  
   *Zhang Q., Chen M., Bukharin A., He P., Cheng Y., Chen W., Zhao T. [[PDF](https://arxiv.org/abs/2303.10512)] [[Code](https://github.com/QingruZhang/AdaLoRA)], 2023*

2. **SaLoRA: Structure-aware low-rank adaptation for parameter-efficient fine-tuning.** `Mathematics`  
    *Hu Y, Xie Y, Wang T, Chen M, Pan Z.* [[PDF](https://www.mdpi.com/2227-7390/11/20/4317)], 2023

3. **IncreLoRA: Incremental Parameter Allocation Method for Parameter-Efficient Fine-Tuning.** `preprint`  
   *Zhang F., Li L., Chen J., Jiang Z., Wang B., Qian Y. [[PDF](https://arxiv.org/abs/2308.12043)] [[Code](https://github.com/FeiyuZhang98/IncreLoRA)], 2023*

#### SRD-Based Methods

1. **DoRA: Enhancing parameter-efficient fine-tuning with dynamic rank distribution.** `preprint`  
    *Mao Y, Huang K, Guan C, Bao G, Mo F, Xu J.* [[PDF](https://arxiv.org/abs/2405.17357)] [[Code](https://github.com/MIkumikumi0116/DoRA)], 2024

2. **AutoLoRA: Automatically tuning matrix ranks in low-rank adaptation based on meta learning.** `preprint`  
    *Zhang R, Qiang R, Somayajula S A, Xie P.* [[PDF](https://arxiv.org/abs/2403.09113)], 2024

3. **SoRA: Sparse low-rank adaptation of pre-trained language models.** `EMNLP`  
    *Ding N, Lv X, Wang Q, Chen Y, Zhou B, Liu Z, Sun M.* [[PDF](https://arxiv.org/abs/2311.11696)] [[Code](https://github.com/TsinghuaC3I/SoRA)], 2023

4. **ALoRA: Allocating low-rank adaptation for fine-tuning large language models.** `preprint`  
    *Liu Z, Lyn J, Zhu W, Tian X, Graham Y.* [[PDF](https://arxiv.org/abs/2403.16187)], 2024

#### Rank Sampling-Based Methods

1. **DyLoRA: Parameter-Efficient Tuning of Pre-trained Models Using Dynamic Search-Free Low-Rank Adaptation.** `EACL 2023`  
   *Valipour M., Rezagholizadeh M., Kobyzev I., Ghodsi A. [[PDF](https://arxiv.org/abs/2210.07558)] [[Code](https://github.com/huawei-noah/Efficient-NLP/tree/main/DyLoRA)], 2023*

### <img src="./fig/star.svg" width="15" height="15" /> Optimizing the Learning Procedure

#### Initialization Improvement

1. **The impact of initialization on LoRA finetuning dynamics.** `preprint`     
   *Hayou S, Ghosh N, Yu B.* [[PDF](https://arxiv.org/abs/2406.08447)], 2024

2. **PISSA: Principal singular values and singular vectors adaptation of large language models.** `preprint`

   *Meng F, Wang Z, Zhang M.* [[PDF](https://arxiv.org/abs/2404.02948)] [[Code](https://github.com/GraphPKU/PiSSA)], 2024

4. **MiLoRA: Harnessing minor singular components for parameter-efficient LLM finetuning.** `preprint`  
   *Wang H, Xiao Z, Li Y, Wang S, Chen G, Chen Y.* [[PDF](https://arxiv.org/abs/2406.09044)], 2024

#### Gradient Update Optimization

1. **Riemannian preconditioned LoRA for fine-tuning foundation models.** `preprint`  
   *Zhang F, Pilanci M.* [[PDF](https://arxiv.org/abs/2402.02347)] [[Code](https://github.com/pilancilab/Riemannian_Preconditioned_LoRA)], 2024

2. **LoRA+: Efficient low rank adaptation of large models.** `preprint`  
   *Hayou S, Ghosh N, Yu B.* [[PDF](https://arxiv.org/abs/2402.12354)] [[Code](https://github.com/nikhil-ghosh-berkeley/loraplus)], 2024

3. **ResLoRA: Identity residual mapping in low-rank adaption.** `preprint`  
   *Shi S, Huang S, Song M, Li Z, Zhang Z, Huang H, Wei F, Deng W, Sun F, Zhang Q.* [[PDF](https://arxiv.org/abs/2402.18039)] [[Code](https://github.com/microsoft/LMOps/tree/main/reslora)], 2024

4. **SIBO: A simple booster for parameter-efficient fine-tuning.** `preprint`  
   *Wen Z, Zhang J, Fang Y.* [[PDF](https://arxiv.org/abs/2402.11896)], 2024
   *Hayou S, Ghosh N, Yu B.* 2024

#### Overfitting Mitigation

1. **BiLoRA: A bi-level optimization framework for overfitting-resilient low-rank adaptation of large pre-trained models.** `preprint`  
   *Qiang R, Zhang R, Xie P.* [[PDF](https://arxiv.org/abs/2403.13037)], 2024

2. **LoRA dropout as a sparsity regularizer for overfitting control.** `preprint`  
   *Lin Y, Ma X, Chu X, Jin Y, Yang Z, Wang Y, Mei H.* [[PDF](https://arxiv.org/abs/2404.09610)], 2024

3. **LoRA meets dropout under a unified framework.** `preprint`  
   *Wang S, Chen L, Jiang J, Xue B, Kong L, Wu C.* [[PDF](https://arxiv.org/abs/2403.00812)] [[Code](https://github.com/TsinghuaC3I/SoRA)], 2024

### <img src="./fig/star.svg" width="15" height="15" /> Combining with other Learning Paradigms

1. **Laplace-LoRA: Bayesian low-rank adaptation for large language models.** `preprint`  
   *Yang A X, Robeyns M, Wang X, Aitchison L.* [[PDF](https://arxiv.org/abs/2308.13111)] [[Code](https://github.com/adamxyang/laplace-lora)], 2023

2. **PILLOW: Enhancing efficient instruction fine-tuning via prompt matching.** `EMNLP`  
   *Qi Z, Tan X, Shi S, Qu C, Xu Y, Qi Y.* [[PDF](https://arxiv.org/abs/2312.05621)], 2023

3. **STAR: Constraint LoRA with dynamic active learning for data-efficient fine-tuning of large language models.** `preprint`  
   *Zhang L, Wu J, Zhou D, Xu G.* [[PDF](https://arxiv.org/abs/2403.01165)] [[Code](https://github.com/callanwu/STAR)], 2024
    
## Cross-task Generalization

### <img src="./fig/star.svg" width="15" height="15" /> Mixture with Manually Designed Weights

1. **LoRA Ensembles for large language model fine-tuning.** `preprint`

   *Wang X, Aitchison L, Rudolph M.* [[PDF](https://arxiv.org/abs/2310.00035)], 2023

2. **LoRAretriever: Input-aware LoRA retrieval and composition for mixed tasks in the wild.** `preprint`

   *Zhao Z, Gan L, Wang G, Zhou W, Yang H, Kuang K, Wu F.* [[PDF](https://arxiv.org/abs/2402.09997)], 2024

3. **Token-level adaptation of LoRA adapters for downstream task generalization.** `AICCC`

   *Belofsky J.* [[PDF](https://arxiv.org/abs/2311.10847)] [[Code](https://github.com/jb-01/LoRA-TLE)], 2023

4. **Effective and parameter-efficient reusing fine-tuned models.** `preprint`

   *Jiang W, Lin B, Shi H, Zhang Y, Li Z, Kwok J T.*[[PDF](https://arxiv.org/abs/2310.01886)] [[Code]()], 2023

5. **Composing parameter-efficient modules with arithmetic operations.**  `preprint`
  
   *Zhang J, Chen S, Liu J, He J.*[[PDF](https://proceedings.neurips.cc/paper_files/paper/2023/hash/299a08ee712d4752c890938da99a77c6-Abstract-Conference.html)] [[Code](https://github.com/hkust-nlp/PEM_composition)], 2023

6. **Task arithmetic with LoRA for continual learning.** `preprint`

   *Chitale R, Vaidya A, Kane A, Ghotkar A.* [[PDF](https://arxiv.org/abs/2311.02428)], 2023    

### <img src="./fig/star.svg" width="15" height="15" /> Mixture with Learnt Weights

1. **LoRAHub: Efficient cross-task generalization via dynamic LoRA composition.** `preprint`

   *Huang C, Liu Q, Lin B Y, Pang T, Du C, Lin M.* [[PDF](https://arxiv.org/abs/2307.13269)] [[Code](https://github.com/sail-sg/lorahub)], 2023

2. **ComPEFT: Compression for communicating parameter efficient updates via sparsification and quantization.** `preprint`

   *Yadav P, Choshen L, Raffel C, Bansal M.* [[PDF](https://arxiv.org/abs/2311.13171)] [[Code](https://github.com/prateeky2806/compeft)], 2023

3. **L-LoRA: Parameter efficient multi-task model fusion with partial linearization.** `preprint`

   *Tang A, Shen L, Luo Y, Zhan Y, Hu H, Du B, Chen Y, Tao D.* [[PDF](https://arxiv.org/abs/2310.04742)] [[Code](https://github.com/tanganke/peta)], 2023

4. **MixLoRA: Multimodal instruction tuning with conditional mixture of LoRA.** `preprint`

   *Shen Y, Xu Z, Wang Q, Cheng Y, Yin W, Huang L.* [[PDF](https://arxiv.org/abs/2402.15896)], 2024

5. **X-LoRA: Mixture of low-rank adapter experts, a flexible framework for large language models with applications in protein mechanics and design.** `preprint`

   *Buehler E L, Buehler M J.* [[PDF](https://pubs.aip.org/aip/aml/article/2/2/026119/3294581)], 2024

### <img src="./fig/star.svg" width="15" height="15" /> Mixture of LoRA Experts

1. **MoRAL: MoE augmented LoRA for LLMs’ lifelong learning.** `preprint`
  
   *Yang S, Ali M A, Wang C, Hu L, Wang D.* [[PDF](https://arxiv.org/abs/2402.11260)], 2024

2. **LoRAMoE: Alleviate world knowledge forgetting in large language models via MoE-style plugin.** `preprint`

   *Dou S, Zhou E, Liu Y, Gao S, Zhao J, Shen W, Zhou Y, Xi Z, Wang X, Fan X, Pu S, Zhu J, Zheng R, Gui T, Zhang Q, Huang X.* [[PDF](https://arxiv.org/abs/2312.09979)] [[Code](https://github.com/Ablustrund/LoRAMoE)], 2023

3. **MoCLE: Mixture of cluster-conditional LoRA experts for vision-language instruction tuning.** `preprint`

   *Gou Y, Liu Z, Chen K, Hong L, Xu H, Li A, Yeung D, Kwok J T, Zhang Y.* [[PDF](https://arxiv.org/abs/2312.12379)][[Code](https://gyhdog99.github.io/projects/mocle/)], 2023

4. **MOELoRA: An MoE-based parameter efficient fine-tuning method for multi-task medical applications.** `preprint`

   *Liu Q, Wu X, Zhao X, Zhu Y, Xu D, Tian F, Zheng Y.* [[PDF](https://arxiv.org/abs/2310.18339)] [[Code](https://github.com/Applied-Machine-Learning-Lab/MOELoRA-peft)], 2023

5. **Mixture-of-LoRAs: An efficient multitask tuning method for large language models.** `LREC/COLING`

   *Feng W, Hao C, Zhang Y, Han Y, Wang H.* [[PDF](https://aclanthology.org/2024.lrec-main.994/)], 2024

6. **MultiLoRA: Democratizing LoRA for better multi-task learning.** `preprint`

   *Wang Y, Lin Y, Zeng X, Zhang G.* [[PDF](https://arxiv.org/abs/2311.11501)], 2023

7. **MLoRE: Multi-task dense prediction via mixture of low-rank experts.** `preprint`

   *Yang Y, Jiang P, Hou Q, Zhang H, Chen J, Li B.* [[PDF](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Multi-Task_Dense_Prediction_via_Mixture_of_Low-Rank_Experts_CVPR_2024_paper.html)] [[Code](https://github.com/YuqiYang213/MLoRE)], 2024

8. **MTLoRA: Low-rank adaptation approach for efficient multi-task learning.** `CVPR`

   *Agiza A R SN. M.* [[PDF](https://openaccess.thecvf.com/content/CVPR2024/html/Agiza_MTLoRA_Low-Rank_Adaptation_Approach_for_Efficient_Multi-Task_Learning_CVPR_2024_paper.html)] [[Code](https://github.com/scale-lab/MTLoRA)], 2024

9.  **MoLA: Higher layers need more LoRA experts.** `preprint`

    *Gao C, Chen K, Rao J, Sun B, Liu R, Peng D, Zhang Y, Guo X, Yang J, Subrahmanian V S.* [[PDF](https://arxiv.org/abs/2402.08562)] [[Code](https://github.com/GCYZSL/MoLA)], 2024

10. **LLaVA-MoLE: Sparse mixture of LoRA experts for mitigating data conflicts in instruction finetuning MLLMs.** `preprint`
  
    *Chen S, Jie Z, Ma L.* [[PDF](https://arxiv.org/abs/2401.16160)], 2024

11. **SiRA: Sparse mixture of low rank adaptation.** `preprint`

    *Zhu Y, Wichers N, Lin C, Wang X, Chen T, Shu L, Lu H, Liu C, Luo L, Chen J, Meng L.* [[PDF](https://arxiv.org/abs/2311.09179)], 2023

12. **Octavius: Mitigating task interference in MLLMs via MoE.** `preprint`  
    *Chen Z, Wang Z, Wang Z, Liu H, Yin Z, Liu S, Sheng L, Ouyang W, Qiao Y, Shao J.* [[PDF](https://arxiv.org/abs/2311.02684)] [[Code](https://github.com/OpenGVLab/LAMM)], 2023

13. **Fast LoRA: Batched low-rank adaptation of foundation models.** `preprint`  
    *Wen Y, Chaudhuri S.* [[PDF](https://arxiv.org/abs/2312.05677)], 2023

14. **I-LoRA: Analyzing and reducing catastrophic forgetting in parameter-efficient tuning.** `preprint`
    *Ren W, Li X, Wang L, Zhao T, Qin W.* [[PDF](https://arxiv.org/abs/2402.18865)] [[Code](https://github.com/which47/LLMCL)], 2024

## Efficiency Improving

### <img src=".\fig\star.svg" width="15" height="15" /> Parameter Reduction

#### Parameter Freezing

1. **LoRA-SP: Streamlined Partial Parameter Adaptation for Resource Efficient Fine-Tuning of Large Language Models** `arXiv`  
   *Y. Wu, Y. Xiang, S. Huo, Y. Gong, P. Liang*. [[PDF](https://arxiv.org/pdf/2403.08822)] 2024 

2. **LoRA-FA: Memory-Efficient Low-Rank Adaptation for Large Language Models Fine-Tuning** `arXiv`  
   *L. Zhang, L. Zhang, S. Shi, X. Chu, B. Li*. [[PDF](https://arxiv.org/pdf/2308.03303)] 2023 

3. **AFLoRA: Adaptive Freezing of Low Rank Adaptation in Parameter Efficient Fine-Tuning of Large Models** `arXiv`  
   *Z. Liu, S. Kundu, A. Li, J. Wan, L. Jiang, P. A. Beerel*. [[PDF](https://arxiv.org/pdf/2403.13269)] 2024 

4. **DropBP: Accelerating Fine-Tuning of Large Language Models by Dropping Backward Propagation** `arXiv`  
   *S. Woo, B. Park, B. Kim, M. Jo, S. Kwon, D. Jeon, D. Lee*. [[PDF](https://arxiv.org/pdf/2402.17812)] [[Code](https://github.com/woosunghyeon/dropbp)] 2024 

5. **LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters** `arXiv`  
   *K. Bałazy, M. Banaei, K. Aberer, J. Tabor*. [[PDF](https://arxiv.org/pdf/2405.17604)] [[Code](https://github.com/mohammadrezabanaei/lora-xs)] 2024 

6. **BYOM-LoRA: Effective and Parameter-Efficient Reusing Fine-Tuned Models** `arXiv`  
   *W. Jiang, B. Lin, H. Shi, Y. Zhang, Z. Li, J. T. Kwok*. [[PDF](https://arxiv.org/pdf/2310.01886)]  2023 

#### Parameter Pruning

1. **LoRA-Drop: Efficient LoRA Parameter Pruning Based on Output Evaluation** `arXiv`  
   *H. Zhou, X. Lu, W. Xu, C. Zhu, T. Zhao*. [[PDF](https://arxiv.org/pdf/2402.07721)] 2024 

2. **LoRAPrune: Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning** `arXiv`  
   *M. Zhang, H. Chen, C. Shen, Z. Yang, L. Ou, X. Zhuang, B. Zhu*. [[PDF](https://arxiv.org/pdf/2305.18403)] 2023 

3. **LoRAShear: Efficient Large Language Model Structured Pruning and Knowledge Recovery** `arXiv`  
   *T. Chen, T. Ding, B. Yadav, I. Zharkov, L. Liang*.  [[PDF](https://arxiv.org/pdf/2310.18356)] [[Code](https://github.com/tianyic/only_train_once)]2023

4. **Parameter-Efficient Fine-Tuning with Layer Pruning on Free-Text Sequence-to-Sequence Modeling** `arXiv`  
   *Y. Zhu, X. Yang, Y. Wu, W. Zhang*. [[PDF](https://arxiv.org/pdf/2305.08285)] [[Code](https://github.com/zhuyunqi96/LoraLPrun)] 2023 

#### Parameter Sharing

1. **VeRA: Vector-Based Random Matrix Adaptation** `arXiv`  
   *D. J. Kopiczko, T. Blankevoort, Y. M. Asano*.  [[PDF](https://arxiv.org/pdf/2310.11454)] 2023

2. **VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks** `arXiv`  
   *Y. Li, S. Han, S. Ji*. [[PDF](https://arxiv.org/pdf/2405.15179)] [[Code](https://github.com/leo-yangli/vb-lora)] 2024 

3. **Parameter-Efficient Fine-Tuning with Discrete Fourier Transform** `arXiv`  
   *Z. Gao, Q. Wang, A. Chen, Z. Liu, B. Wu, L. Chen, J. Li*. [[PDF](https://arxiv.org/pdf/2405.03003)] [[Code](https://github.com/chaos96/fourierft)] 2024 

### <img src=".\fig\star.svg" width="15" height="15" /> Parameter Quantization

#### PTQ-Based Methods

1. **QLoRA: Efficient Fine-Tuning of Quantized LLMs** `NeurIPS`  
   *T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer*. 2024 [[PDF](https://arxiv.org/pdf/2305.14314)] [[Code](https://github.com/artidoro/qlora)]

2. **QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models** `arXiv`  
   *Y. Xu, L. Xie, X. Gu, X. Chen, H. Chang, H. Zhang, Z. Chen, X. Zhang, Q. Tian*. 2023 [[PDF](https://arxiv.org/pdf/2309.14717)] [[Code](https://github.com/intel-analytics/ipex-llm)]

#### QAT-Based Methods

1. **LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models** `arXiv`  
   *Y. Li, Y. Yu, C. Liang, P. He, N. Karampatziakis, W. Chen, T. Zhao*. [[PDF](https://arxiv.org/pdf/2310.08659)] [[Code](https://github.com/yxli2123/loftq)] 2023

2. **ApiQ: Finetuning of 2-Bit Quantized Large Language Model** `arXiv`  
   *B. Liao, C. Monz*. [[PDF](https://arxiv.org/pdf/2402.05147)] [[Code](https://github.com/baohaoliao/apiq)] 2024 

3. **L4Q: Parameter Efficient Quantization-Aware Training on Large Language Models via LoRA-Wise LSQ** `arXiv`  
   *H. Jeon, Y. Kim, J. Kim*. 2024 [[PDF](https://arxiv.org/pdf/2402.04902)]

### <img src=".\fig\star.svg" width="15" height="15" /> Parallel LoRA Computing Frameworks

#### Parallel Fine-tuning

1. **ASPEN: High-Throughput LoRA Fine-Tuning of Large Language Models with a Single GPU** `arXiv`  
   *Z. Ye, D. Li, J. Tian, T. Lan, J. Zuo, L. Duan, Y. Jiang, J. Sha, K. Zhang, M. Tang*. [[PDF](https://arxiv.org/pdf/2312.02515)] [[Code](https://github.com/TUDB-Labs/mLoRA)] 2023 

#### Parallel Inference

1. **Punica: Multi-Tenant LoRA Serving** `MLSys`  
   *L. Chen, Z. Ye, Y. Wu, D. Zhuo, L. Ceze, A. Krishnamurthy*. [[PDF](https://arxiv.org/pdf/2310.18547)] [[Code](https://github.com/punica-ai/punica)]  2024 

2. **S-LoRA: Serving Thousands of Concurrent LoRA Adapters** `arXiv`  
   *Y. Sheng, S. Cao, D. Li, C. Hooper, N. Lee, S. Yang, C.-C. Chou, B. Zheng, K. Keutzer*. [[PDF](https://arxiv.org/pdf/2311.03285)] [[Code](https://github.com/s-lora/s-lora)] 2023 

3. **CARASERVE: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference** `arXiv`  
   *S. Li, H. Lu, T. Wu, M. Yu, Q. Weng, X. Chen, Y. Shan, B. Yuan, W. Wang*. [[PDF](https://arxiv.org/pdf/2401.11240)] 2024 

## LoRA for Federated Learning

###  Data Heterogeneity

1. **SLoRA: Federated parameter efficient fine-tuning of language models.** `arXiv preprint`  
   *Babakniya S, Elkordy A R, Ezzeldin Y H, Liu Q, Song K, El-Khamy M, Avestimehr S.* [[PDF](https://arxiv.org/pdf/2308.06522)], 2023

2. **FeDeRA: Efficient fine-tuning of language models in federated learning leveraging weight decomposition.** `arXiv preprint`  
   *Yan Y, Tang S, Shi Z, Yang Q.* [[PDF](https://arxiv.org/pdf/2404.18848)], 2024

3. **Improving LoRA in privacy-preserving federated learning.** `arXiv preprint`  
   *Sun Y, Li Z, Li Y, Ding B.* [[PDF](https://arxiv.org/pdf/2403.12313)], 2024

### <img src="./fig/star.svg" width="15" height="15" /> Device Heterogeneity

1. **FedMS: Federated learning with mixture of sparsely activated foundation models.** `arXiv preprint`  
   *Wu P, Li K, Wang T, Wang F.* [[PDF](https://arxiv.org/pdf/2312.15926)], 2023

2. **Federated fine-tuning of large language models under heterogeneous language tasks and client resources.** `arXiv preprint`  
*Bai J, Chen D, Qian B, Yao L, Li Y.* [[PDF](https://arxiv.org/pdf/2402.11505)] [[Code](https://github.com/alibaba/FederatedScope/tree/FlexLoRA)], 2024

3. **Heterogeneous LoRA for federated fine-tuning of on-device foundation models.** `NeurIPS`  
   *Cho Y J, Liu L, Xu Z, Fahrezi A, Barnes M, Joshi G.* [[PDF](https://openreview.net/pdf?id=EmV9sGpZ7q)], 2023

### <img src="./fig/star.svg" width="15" height="15" /> Model Heterogeneity

1. **pFedLoRA: Model-Heterogeneous Personalized Federated Learning with LoRA Tuning.** `arXiv preprint`  
   *Yi L, Yu H, Wang G, Liu X, Li X.* [[PDF](https://arxiv.org/pdf/2310.13283)], 2023

### <img src="./fig/star.svg" width="15" height="15" /> Parameter Privacy

1. **A fast, performant, secure distributed training framework for large language model.** `arXiv preprint`  
   *Huang W, Wang Y, Cheng A, Zhou A, Yu C, Wang L.* [[PDF](https://arxiv.org/pdf/2401.09796)], 2024

2. **PrivateLoRA for efficient privacy-preserving LLM.** `arXiv preprint`  
   *Wang Y, Lin Y, Zeng X, Zhang G.* [[PDF](https://arxiv.org/pdf/2311.14030)], 2023

## Applications of LoRA

### <img src="./fig/star.svg" width="15" height="15" /> Language Tasks

#### Traditional NLP Task

1. **DialogueLLM: Context and Emotion Knowledge-Tuned Large Language Models for Emotion Recognition in Conversations.** `preprint`

    *Zhang Y, Wang M, Wu Y, Tiwari P, Li Q, Wang B, Qin J.* [[PDF](http://arxiv.org/abs/2310.11374v4)], 2024.

2. **Label Supervised LLaMA Finetuning.** `preprint`

    *Li Z, Li X, Liu Y, Xie H, Li J, Wang F L, Li Q, Zhong X.* [[PDF](http://arxiv.org/abs/2310.01208v1)][[Code](https://github.com/4AI/LS-LLaMA)], 2023.

3. **Speaker Attribution in German Parliamentary Debates with QLoRA-Adapted Large Language Models.** `preprint`

    *Bornheim T, Grieger N, Blaneck P G, Bialonski S.* [[PDF](http://arxiv.org/abs/2310.10083v2)], 2024.

4. **AutoRE: Document-Level Relation Extraction with Large Language Models.** `preprint`

    *Xue L, Zhang D, Dong Y, Tang J.* [[PDF](http://arxiv.org/abs/2308.10252v1)] [[Code](https://github.com/THUDM/AutoRE)], 2024.

5. **Steering Large Language Models for Machine Translation with Finetuning and In-Context Learning.** EMNLP

    *Alves D M, Guerreiro N M, Alves J, Pombal J, Rei R, Souza D J G C, Colombo P, Martins A F T.* [[PDF](http://arxiv.org/abs/2310.13448v1)] [[Code](https://github.com/deep-spin/translation_llm)], 2023.

6. **Finetuning Large Language Models for Domain-Specific Machine Translation.** `preprint`

    *Zheng J, Hong H, Wang X, Su J, Liang Y, Wu S.* [[PDF](http://arxiv.org/abs/2402.15061v1)], 2024.

7. **Assessing Translation Capabilities of Large Language Models Involving English and Indian Languages.** `preprint`

    *Mujadia V, Urlana A, Bhaskar Y, Pavani P A, Shravya K, Krishnamurthy P, Sharma D M.* [[PDF](http://arxiv.org/abs/2402.13929v3)], 2023.

8. **Personalized LoRA for Human-Centered Text Understanding.** AAAI

    *Zhang Y, Wang J, Yu L, Xu D, Zhang X.* [[PDF](http://arxiv.org/abs/2403.06208v1)] [[Code](https://github.com/yoyo-yun/PLoRA.)], 2024.

9. **Y-tuning: An Efficient Tuning Paradigm for Large-Scale Pre-Trained Models via Label Representation Learning.** Frontiers of Computer Science

    *Liu Y, An C, Qiu X.* [[PDF](https://arxiv.org/abs/2202.09817)], 2024.


#### Code Task

1. **Delving into parameter-efficient fine-tuning in code change learning: An empirical study.** `preprint`

    *Liu S, Keung J, Yang Z, Liu F, Zhou Q, Liao Y.* [[PDF](https://arxiv.org/abs/2402.06247)], 2024.

2. **An empirical study on jit defect prediction based on bert-style model.** `preprint`

    *Guo Y, Gao X, Jiang B.* [[PDF](http://arxiv.org/abs/2403.11158)], 2024.

3. **Parameter-efficient finetuning of transformers for source code.** `preprint`

    *Ayupov S, Chirkova N.* [[PDF](https://arxiv.org/abs/2212.05901)][[Code](https://github.com/ShamerD/source-code-efficient-ft)], 2022.

4. **Repairllama: Efficient representations and fine-tuned adapters for program repair.** `preprint`

    *Silva A, Fang S, Monperrus M.* [[PDF](https://arxiv.org/abs/2312.15698)][[Code](https://github.com/ASSERT-KTH/repairllama)], 2023.

5. **Analyzing the effectiveness of large language models on text-to-sql synthesis.** `preprint`

    *Roberson R, Kaki G, Trivedi A.* [[PDF](https://arxiv.org/abs/2401.12379)], 2024.

6. **Stelocoder: a decoder-only LLM for multi-language to python code translation.** `preprint`

    *Pan J, Sadé A, Kim J, Soriano E, Sole G, Flamant S.* [[PDF](http://arxiv.org/abs/2310.15539v2)][[Code](https://github.com/sade-adrien/SteloCoder)], 2023.

#### Model Alignment Task

1. **Perl: parameter efficient reinforcement learning from human feedback.** `preprint`

    *H. Sidahmed, S. Phatale, A. Hutcheson, Z. Lin, Z. Chen, Z. Yu, J. Jin, R. Komarytsia, C. Ahlheim, Y. Zhu, S. Chaudhary, B. Li, S. Ganesh, B. Byrne, J. Hoffmann, H. Mansoor, W. Li, A. Rastogi, L. Dixon.* [[PDF](http://arxiv.org/abs/2403.10704v1)][[Code](https://github.com/google-research-datasets/Taskmaster/tree/master/TM-4-2024)], 2024

2. **Efficient RLHF: reducing the memory usage of PPO.** `preprint`

    *M. Santacroce, Y. Lu, H. Yu, Y. Li, Y. Shen.* [[PDF](http://arxiv.org/abs/2309.00754v1)], 2023

3. **Exploring the impact of low-rank adaptation on the performance, efficiency, and regularization of RLHF.** `preprint`

    *S. Sun, D. Gupta, M. Iyyer.* [[PDF](http://arxiv.org/abs/2309.09055v1)][[Code](https://github.com/SimengSun/alpaca_farm_lora)], 2023

4. **Dmoerm: Recipes of mixture-of-experts for effective reward modeling.** `preprint`

    *S. Quan.* [[PDF](http://arxiv.org/abs/2403.01197v1)][[Code](https://github.com/quanshr/DMoERM)], 2024

5. **Improving reinforcement learning from human feedback with efficient reward model ensemble.** `preprint`

    *S. Zhang, Z. Chen, S. Chen, Y. Shen, Z. Sun, C. Gan.* [[PDF](http://arxiv.org/abs/2401.16635v1)], 2024

6. **Uncertainty-penalized reinforcement learning from human feedback with diverse reward lora ensembles.** `preprint`

    *Y. Zhai, H. Zhang, Y. Lei, Y. Yu, K. Xu, D. Feng, B. Ding, H. Wang.* [[PDF](http://arxiv.org/abs/2401.00243v1)], 2024

7. **Bayesian reward models for LLM alignment.** `preprint`

    *A. X. Yang, M. Robeyns, T. Coste, J. Wang, H. Bou-Ammar, L. Aitchison.* [[PDF](http://arxiv.org/abs/2402.13210v1)], 2024

8. **Bayesian low-rank adaptation for large language models.** `preprint`

    *A. X. Yang, M. Robeyns, X. Wang, L. Aitchison.* [[PDF](http://arxiv.org/abs/2308.13111v5)][[Code](https://github.com/MaximeRobeyns/bayesian_lora)], 2023

#### Vertical Domain Task

1. **Bioinstruct: Instruction tuning of large language models for biomedical natural language processing.** `preprint`

    *Tran H, Yang Z, Yao Z, Yu H.* [[PDF](http://arxiv.org/abs/2310.19975v2)][[Code](https://github.com/bio-nlp/BioInstruct)], 2023

2. **Parameterefficient fine-tuning of llama for the clinical domain.** `preprint`

    *Gema A P, Daines L, Minervini P, Alex B.* [[PDF](https://arxiv.org/abs/2307.03042)][[Code](https://github.com/aryopg/clinical_peft)], 2023

3. **Clinical camel: An open-source expert-level medical language model with dialogue-based knowledge encoding.** `preprint`

    *Toma A, Lawler P R, Ba J, Krishnan R G, Rubin B B, Wang B.* [[PDF](http://arxiv.org/abs/2305.12031v2)][[Code](https://github.com/bowang-lab/clinical-camel?tab=readme-ov-file
    )], 2023

4. **Suryakiran at mediqa-sum 2023: Leveraging lora for clinical dialogue summarization.** `CLEF`

    *Suri K, Mishra P, Saha S, Singh A.* [[PDF](http://arxiv.org/abs/2307.05162v1)], 2023

5. **Assertion detection large language model in-context learning lora fine-tuning.** `preprint`

    *Ji Y, Yu Z, Wang Y.* [[PDF](http://arxiv.org/abs/2401.17602v1)][[Code](https://github.com/JoyDajunSpaceCraft/Assertion_LLM)], 2024

6. **Ivygpt: Interactive chinese pathway language model in medical domain.** `CAAI`

    *Wang R, Duan Y, Lam C, Chen J, Xu J, Chen H, Liu X, Pang P C, Tan T.* [[PDF](http://arxiv.org/abs/2307.10512v1)], 2023

7. **SM70: A large language model for medical devices.** `preprint`

    *Bhatti A, Parmar S, Lee S.* [[PDF](http://arxiv.org/abs/2312.06974v1)], 2023

8. **Finllama: Financial sentiment classification for algorithmic trading applications.** `preprint`

    *Konstantinidis T, Iacovides G, Xu M, Constantinides T G, Mandic D P.* [[PDF](http://arxiv.org/abs/2403.12285v1)], 2024

9. **Financial news analytics using fine-tuned llama 2 GPT model.** `preprint`

    *Pavlyshenko B M.* [[PDF](http://arxiv.org/abs/2308.13032v2)], 2023

10. **Fingpt: Democratizing internet-scale data for financial large language models.** `preprint`

    *Liu X, Wang G, Zha D.* [[PDF](http://arxiv.org/abs/2307.10485v2)][[Code](https://github.com/AI4Finance-Foundation/FinGPT)], 2023

11. **Ra-cfgpt: Chinese financial assistant with retrievalaugmented large language model.** `Frontiers of Computer Science`

    *Li J, Lei Y, Bian Y, Cheng D, Ding Z, Jiang C.* [[PDF](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-024-31018-5)], 2024

12. **Db-gpt: Large language model meets database.** `Data Science and Engineering`

    *Zhou X, Sun Z, Li G.* [[PDF](https://link.springer.com/article/10.1007/s41019-023-00235-6)][[Code](https://github.com/TsinghuaDatabaseGroup/DB-GPT)], 2024

### <img src="./fig/star.svg" width="15" height="15" /> Vision Tasks

#### Image Generation Tasks

1. **Diffstyler: Diffusion-based localized image style transfer.** `preprint`

   *Li S.* [[PDF](http://arxiv.org/abs/2403.18461v1)], 2024

2. **Implicit style-content separation using b-lora.** `preprint`

   *Frenkel Y, Vinker Y, Shamir A, Cohen-Or D.* [[PDF](http://arxiv.org/abs/2403.14572v1)][[Code](https://github.com/yardenfren1996/B-LoRA)], 2024

3. **Facechain: A playground for human-centric artificial intelligence generated content.** `preprint`

   *Liu Y, Yu C, Shang L, He Y, Wu Z, Wang X, Xu C, Xie H, Wang W, Zhao Y, Zhu L, Cheng C, Chen W, Yao Y, Zhou W, Xu J, Wang Q, Chen Y, Xie X, Sun B.* [[PDF](http://arxiv.org/abs/2308.14256v2)][[Code](https://github.com/modelscope/facechain)], 2023

4. **Calliffusion: Chinese calligraphy generation and style transfer with diffusion modeling.** `preprint`

   *Liao Q, Xia G, Wang Z.* [[PDF](http://arxiv.org/abs/2305.19124v1)], 2023

5. **Style transfer to calvin and hobbes comics using stable diffusion.** `preprint`

   *Shrestha S, Venkataramanan A, others.* [[PDF](http://arxiv.org/abs/2312.03993v1)], 2023

6. **Block-wise lora: Revisiting fine-grained lora for effective personalization and stylization in text-to-image generation.** `preprint`

   *Li L, Zeng H, Yang C, Jia H, Xu D.* [[PDF](http://arxiv.org/abs/2403.07500v1)], 2024

7. **OMG: occlusion-friendly personalized multi-concept generation in diffusion models.** `preprint`

   *Kong Z, Zhang Y, Yang T, Wang T, Zhang K, Wu B, Chen G, Liu W, Luo W.* [[PDF](http://arxiv.org/abs/2403.10983v1)][[Code](https://github.com/kongzhecn/OMG)], 2024

8. **Space narrative: Generating images and 3d scenes of chinese garden from text using deep learning.** `preprint`

   *Shi J, Hua H.* [[PDF](http://arxiv.org/abs/2311.00339v1)], 2023

9. **Generating coherent comic with rich story using chatgpt and stable diffusion.** `preprint`

   *Jin Z, Song Z.* [[PDF](http://arxiv.org/abs/2305.11067v2)], 2023

10. **Customizing 360-degree panoramas through text-to-image diffusion models.** `WACV`

    *Wang H, Xiang X, Fan Y, Xue J.* [[PDF](http://arxiv.org/abs/2310.18840v2)][[Code](https://github.com/littlewhitesea/StitchDiffusion)], 2024

11. **Smooth diffusion: Crafting smooth latent spaces in diffusion models.** `preprint`

    *Guo J, Xu X, Pu Y, Ni Z, Wang C, Vasu M, Song S, Huang G, Shi H.* [[PDF](http://arxiv.org/abs/2312.04410v1)][[Code](https://github.com/SHI-Labs/Smooth-Diffusion
    )], 2023

12. **Resadapter: Domain consistent resolution adapter for diffusion models.** `preprint`

    *Cheng J, Xie P, Xia X, Li J, Wu J, Ren Y, Li H, Xiao X, Zheng M, Fu L.* [[PDF](http://arxiv.org/abs/2403.02084v1)][[Code](https://github.com/bytedance/res-adapter)], 2024

13. **Continual diffusion with stamina: Stack-and-mask incremental adapters.** `CVPR`

    *Smith J S, Hsu Y C, Kira Z, Shen Y, Jin H.* [[PDF](http://arxiv.org/abs/2311.18763v1)], 2024

14. **Dreamsync: Aligning text-to-image generation with image understanding feedback.** `CVPR`

    *Sun J, Fu D, Hu Y, Wang S, Rassin R, Juan D C, Alon D, Herrmann C, Steenkiste v S, Krishna R, others.* [[PDF](http://arxiv.org/abs/2311.17946v1)], 2023

15. **Styleadapter: A single-pass lora-free model for stylized image generation.** `preprint`

    *Wang Z, Wang X, Xie L, Qi Z, Shan Y, Wang W, Luo P.* [[PDF](http://arxiv.org/abs/2309.01770v1)], 2023

16. **Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models.** `NeurIPS`

    *Gu Y, Wang X, Wu J Z, Shi Y, Chen Y, Fan Z, Xiao W, Zhao R, Chang S, Wu W, Ge Y, Shan Y, Shou M Z.* [[PDF](http://arxiv.org/abs/2305.18292v2)][[Code](https://github.com/TencentARC/Mix-of-Show)], 2023

17. **LCM-lora: A universal stable-diffusion acceleration module.** `preprint`

    *Luo S, Tan Y, Patil S, Gu D, Platen v P, Passos A, Huang L, Li J, Zhao H.* [[PDF](http://arxiv.org/abs/2311.05556v1)][[Code](https://github.com/luosiallen/latent-consistency-model)], 2023

18. **Lora-enhanced distillation on guided diffusion models.** `preprint`

    *Golnari P A.* [[PDF](http://arxiv.org/abs/2312.06899v1)], 2023

19. **Customize-a-video: One-shot motion customization of text-to-video diffusion models.** `preprint`

    *Ren Y, Zhou Y, Yang J, Shi J, Liu D, Liu F, Kwon M, Shrivastava A.* [[PDF](http://arxiv.org/abs/2402.14780v1)], 2024

20. **Dragvideo: Interactive drag-style video editing.** `preprint`

    *Deng Y, Wang R, Zhang Y, Tai Y, Tang C.* [[PDF](http://arxiv.org/abs/2312.02216v1)][[Code](https://github.com/RickySkywalker/DragVideo-Official)], 2023

21. **Rerender A video: Zero-shot text-guided video-to-video translation.** `SIGGRAPH`

    *Yang S, Zhou Y, Liu Z, Loy C C.* [[PDF](http://arxiv.org/abs/2306.07954v2)][[Code](https://github.com/williamyang1991/Rerender_A_Video)], 2023

22. **Infusion: Inject and attention fusion for multi concept zero-shot text-based video editing.** `ICCV`

    *Khandelwal A.* [[PDF](http://arxiv.org/abs/2308.00135v3)][[Code](https://infusion-zero-edit.github.io/)], 2023

23. **Stable video diffusion: Scaling latent video diffusion models to large datasets.** `preprint`

    *Blattmann A, Dockhorn T, Kulal S, Mendelevitch D, Kilian M, Lorenz D, Levi Y, English Z, Voleti V, Letts A, others.* [[PDF](http://arxiv.org/abs/2311.15127v1)], 2023

24. **Animatediff: Animate your personalized text-to-image diffusion models without specific tuning.** `preprint`

    *Guo Y, Yang C, Rao A, Wang Y, Qiao Y, Lin D, Dai B.* [[PDF](http://arxiv.org/abs/2307.04725v2)][[Code](https://github.com/guoyww/AnimateDiff)], 2023

25. **Dreamcontrol: Control-based text-to-3d generation with 3d self-prior.** `preprint`

    *Huang T, Zeng Y, Zhang Z, Xu W, Xu H, Xu S, Lau R W H, Zuo W.* [[PDF](http://arxiv.org/abs/2312.06439v2)][[Code](https://github.com/tyhuang0428/DreamControl)], 2023

26. **X-dreamer: Creating high-quality 3d content by bridging the domain gap between text-to-2d and text-to-3d generation.** `preprint`

    *Ma Y, Fan Y, Ji J, Wang H, Sun X, Jiang G, Shu A, Ji R.* [[PDF](http://arxiv.org/abs/2312.00085v2)][[Code](https://github.com/xmu-xiaoma666/X-Dreamer)], 2023

27. **Boosting3d: High-fidelity image-to-3d by boosting 2d diffusion prior to 3d prior with progressive learning.** `preprint`

    *Yu K, Liu J, Feng M, Cui M, Xie X.* [[PDF](http://arxiv.org/abs/2311.13617v1)], 2023

28. **As-plausible-as-possible: Plausibility-aware mesh deformation using 2d diffusion priors.** `CVPR`

    *Yoo S, Kim K, Kim V G, Sung M.* [[PDF](http://arxiv.org/abs/2311.16739v1)][[Code](https://as-plausible-as-possible.github.io/
    )], 2024

29. **Dragtex: Generative point-based texture editing on 3d mesh.** `preprint`

    *Zhang Y, Xu Q, Zhang L.* [[PDF](http://arxiv.org/abs/2403.02217v1)], 2024

#### Image Segmentation Task

1. **Samlp: A customized segment anything model for license plate detection.** `preprint`

    *Ding H, Gao J, Yuan Y, Wang Q*. [[PDF](http://arxiv.org/abs/2401.06374v1)][[Code](https://github.com/Dinghaoxuan/SamLP
    )], 2024

2. **Sam-based instance segmentation models for the automation of structural damage detection.** `preprint`

    *Ye Z, Lovell L, Faramarzi A, Ninic J*. [[PDF](http://arxiv.org/abs/2401.15266v2)], 2024

3. **Segment any cell: A sam-based auto-prompting fine-tuning framework for nuclei segmentation.** `preprint`

    *Na S, Guo Y, Jiang F, Ma H, Huang J*. [[PDF](http://arxiv.org/abs/2401.13220v1)], 2024

4. **SAM-OCTA: prompting segment-anything for OCTA image segmentation.** `preprint`

    *Chen X, Wang C, Ning H, Li S*. [[PDF](http://arxiv.org/abs/2310.07183v2)][[Code](https://github.com/ShellRedia/SAM-OCTA
    )], 2023

5. **Cheap lunch for medical image segmentation by fine-tuning SAM on few exemplars.** `preprint`

    *Feng W, Zhu L, Yu L*. [[PDF](http://arxiv.org/abs/2308.14133v1)], 2023

6. **Customized segment anything model for medical image segmentation.** `preprint`

    *Zhang K, Liu D*. [[PDF](http://arxiv.org/abs/2304.13785v2)], 2023

7. **SAM meets robotic surgery: An empirical study on generalization, robustness and adaptation.** `MICCAI. `

    *Wang A, Islam M, Xu M, Zhang Y, Ren H*. [[PDF](http://arxiv.org/abs/2308.07156v1)], 2023

8. **Tracking meets lora: Faster training, larger model, stronger performance.** `preprint`

    *Lin L, Fan H, Zhang Z, Wang Y, Xu Y, Ling H*. [[PDF](http://arxiv.org/abs/2403.05231v1)], 2024

9. **Enhancing general face forgery detection via vision transformer with low-rank adaptation.** `MIPR. `

    *Kong C, Li H, Wang S*. [[PDF](http://arxiv.org/abs/2303.00917v2)], 2023

### <img src="./fig/star.svg" width="15" height="15" /> Multimodal Tasks

#### Audio-Text

1. **SALM: speech-augmented language model with in-context learning for speech recognition and translation.** `preprint`

    *Chen Z, Huang H, Andrusenko A, Hrinchuk O, Puvvada KC, Li J, Ghosh S, Balam J, Ginsburg B.* [[PDF](http://arxiv.org/abs/2310.09424v1)], 2023

#### Image-Text

1. **InternLM-XComposer2: Mastering Free-Form Text-Image Composition and Comprehension in Vision-Language Large Model.** `preprint`

    *Chen Z, Huang H, Andrusenko A, Hrinchuk O, Puvvada KC, Li J, Ghosh S, Balam J, Ginsburg B.* [[PDF](http://arxiv.org/abs/2401.16420v1)][[Code](https://github.com/InternLM/InternLM-XComposer)], 2024

2. **mPlug-OWL: Modularization Empowers Large Language Models with Multimodality.** `preprint`

    *Ye Q, Xu H, Xu G, Ye J, Yan M, Zhou Y, Wang J, Hu A, Shi P, Shi Y, Li C, Xu Y, Chen H, Tian J, Qi Q, Zhang J, Huang F.* [[PDF](http://arxiv.org/abs/2304.14178v2)][[Code](https://github.com/X-PLUG/mPLUG-Owl)], 2023

3. **Collavo: Crayon Large Language and Vision Model.** `preprint`

    *Lee B, Park B, Kim CW, Ro YM.* [[PDF](http://arxiv.org/abs/2402.11248v2)][[Code](https://github.com/ByungKwanLee/CoLLaVO)], 2024

#### Video-Text

1. **Where visual speech meets language: VSP-LLM framework for efficient and context-aware visual speech processing.** `preprint`

    *J. H. Yeo, S. Han, M. Kim, Y. M. Ro.* [[PDF](http://arxiv.org/abs/2402.15151v1)][[Code](https://github.com/Sally-SH/VSP-LLM
    )], 2024

2. **Molca: Molecular graph-language modeling with cross-modal projector and uni-modal adapter.** `EMNLP. `

    *Z. Liu, S. Li, Y. Luo, H. Fei, Y. Cao, K. Kawaguchi, X. Wang, T. Chua.* [[PDF](http://arxiv.org/abs/2310.12798v4)][[Code](https://github.com/acharkq/MolCA)], 2023

3. **TPLLM: A traffic prediction framework based on pretrained large language models.** `preprint`

    *Y. Ren, Y. Chen, S. Liu, B. Wang, H. Yu, Z. Cui.* [[PDF](http://arxiv.org/abs/2403.02221v2)], 2024

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


