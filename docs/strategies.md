# Strategies

## nFL — No Federated Learning

| Name | Venue | Year | Paper | URL |
|------|-------|------| ----- | ----|
| LocalOnly | | | | |
| Centralized | | | | |
| SimTS | ICASSP | 2024 | SimTS: Rethinking Contrastive Representation Learning for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2303.18205) - [IEEE](https://ieeexplore.ieee.org/document/10446875) - [GitHub](https://github.com/xingyu617/SimTS_Representation_Learning) |

## tFL — Traditional Federated Learning

| Name | Venue | Year | Paper | URL |
|------|-------|------| ----- | ----|
| FedAvg | AISTATS | 2017 | Communication-Efficient Learning of Deep Networks from Decentralized Data | [Arxiv](https://arxiv.org/abs/1602.05629) |
| SCAFFOLD | ICML | 2020 | SCAFFOLD: Stochastic Controlled Averaging for Federated Learning | [Arxiv](https://arxiv.org/abs/1910.06378) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverscaffold.py) |
| Krum | NeurIPS | 2017 | Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent | [PUB](https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/krum.py) |
| FedMedian | ICML | 2018 | Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | [Arxiv](https://arxiv.org/abs/1803.01498) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedmedian.py) |
| FedTrimmedAvg | ICML | 2018 | Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | [Arxiv](https://arxiv.org/abs/1803.01498) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedtrimmedavg.py) |
| FedAdam | ICLR | 2021 | Adaptive Federated Optimization | [Arxiv](https://arxiv.org/abs/2003.00295) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedadam.py) |
| FedYogi | ICLR | 2021 | Adaptive Federated Optimization | [Arxiv](https://arxiv.org/abs/2003.00295) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedyogi.py) |
| MOON | CVPR | 2021 | Model-Contrastive Federated Learning | [Arxiv](https://arxiv.org/abs/2103.16257) |
| FedRolex | NeurIPS | 2022 | FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction | [Arxiv](https://arxiv.org/abs/2211.11614) - [GitHub](https://github.com/AIoT-MLSys-Lab/FedRolex) |
| Elastic | CVPR | 2023 | Elastic Aggregation for Federated Optimization | [OpenAccess](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Elastic_Aggregation_for_Federated_Optimization_CVPR_2023_paper.html) - [REF](https://github.com/KarhouTam/FL-bench/blob/master/src/server/elastic.py) |
| FedCross | ICDE | 2024 | FedCross: Towards Accurate Federated Learning via Multi-Model Cross-Aggregation | [IEEE](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10597740) - [Arxiv](https://arxiv.org/abs/2210.08285) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/servercross.py) |
| FedRCL | CVPR | 2024 | Relaxed Contrastive Learning for Federated Learning | [OpenAccess](https://openaccess.thecvf.com/content/CVPR2024/html/Seo_Relaxed_Contrastive_Learning_for_Federated_Learning_CVPR_2024_paper.html) - [Arxiv](https://arxiv.org/abs/2401.04928) - [GitHub](https://github.com/skynbe/FedRCL) |
| FedAWA | CVPR | 2025 | FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors | [CVPR](https://cvpr.thecvf.com/virtual/2025/poster/33857) - [Arxiv](https://arxiv.org/abs/2503.15842) |
| FedTrend | Science China Information Sciences | 2026 | Tackling Data Heterogeneity in Federated Time Series Forecasting | [PUB](https://doi.org/10.1007/s11432-025-4553-x) - [Arxiv](https://arxiv.org/abs/2411.15716) |
| FedAvgM | | | Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification | [Arxiv](https://arxiv.org/abs/1909.06335) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavgm.py) |
| FedADMM | ICDE | 2022 | FedADMM: A Robust Federated Deep Learning Framework with Adaptability to System Heterogeneity | [IEEE](https://ieeexplore.ieee.org/abstract/document/9835545) - [REF](https://github.com/KarhouTam/FL-bench/blob/master/src/server/fedadmm.py) |
| FedLAW | ICML | 2023 | Revisiting Weighted Aggregation in Federated Learning with Neural Networks | [Arxiv](https://arxiv.org/abs/2302.10911) - [GitHub](https://github.com/ZexiLee/ICML-2023-FedLAW) |

## pFL — Personalized Federated Learning

| Name | Venue | Year | Paper | URL |
|------|-------|------| ----- | ----|
| FedProx | MLsys | 2020 | Federated Optimization in Heterogeneous Networks | [Arxiv](https://arxiv.org/abs/1812.06127) |
| Ditto | ICML | 2021 | Ditto: Fair and Robust Federated Learning Through Personalization | [Arxiv](https://arxiv.org/abs/2012.04235) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverditto.py) |
| pFedMe | NeurIPS | 2020 | Personalized Federated Learning with Moreau Envelopes | [Arxiv](https://arxiv.org/abs/2006.08848) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverpFedMe.py) |
| APFL | arXiv | 2020 | Adaptive Personalized Federated Learning | [Arxiv](https://arxiv.org/abs/2003.13461) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverapfl.py) |
| PerAvg | NeurIPS | 2020 | Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach | [Arxiv](https://arxiv.org/abs/2002.07948) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverperavg.py) |
| FedAMP | AAAI | 2021 | Personalized Cross-Silo Federated Learning on Non-IID Data | [Arxiv](https://arxiv.org/abs/2007.03797) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serveramp.py) |
| FedBN | ICLR | 2021 | FedBN: Federated Learning on Non-IID Features via Local Batch Normalization | [Arxiv](https://arxiv.org/abs/2102.07623) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverbn.py) |
| FML | NeurIPS | 2020 | Federated Mutual Learning | [Arxiv](https://arxiv.org/abs/2006.16765) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/clients/clientfml.py) |
| FedDyn | ICLR | 2021 | Federated Learning Based on Dynamic Regularization | [Arxiv](https://arxiv.org/abs/2111.04263) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverdyn.py) |
| FedALA | AAAI | 2023 | FedALA: Adaptive Local Aggregation for Personalized Federated Learning | [Arxiv](https://arxiv.org/abs/2212.01197) |
| FedCAC | ICCV | 2023 | Bold but Cautious: Unlocking the Potential of Personalized Federated Learning through Cautiously Aggressive Collaboration | [OpenAccess](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Bold_but_Cautious_Unlocking_the_Potential_of_Personalized_Federated_Learning_ICCV_2023_paper.html) - [Arxiv](https://arxiv.org/abs/2309.11103) |
| FDCR | NeurIPS | 2024 | Parameter Disparities Dissection for Backdoor Defense in Heterogeneous Federated Learning | [Arxiv](https://arxiv.org/abs/2404.10332) - [GitHub](https://github.com/WenkeHuang/FDCR) |
| FedIT | ICASSP | 2024 | Towards Building the Federated GPT: Federated Instruction Tuning | [Arxiv](https://arxiv.org/abs/2305.05644) |
| FFA_LoRA | ICLR | 2024 | Improving LoRA in Privacy-preserving Federated Learning | [Arxiv](https://arxiv.org/abs/2403.12313) |
| FedSA_LoRA | ICLR | 2025 | Selective Aggregation for Low-Rank Adaptation in Federated Learning | [Arxiv](https://arxiv.org/abs/2410.01463) |
| LoRA_FAIR | ICCV | 2025 | LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement | [Arxiv](https://arxiv.org/abs/2411.14961) - [GitHub](https://github.com/jmbian/LoRA-FAIR) |
| pFedHN | ICML | 2021 | Personalized Federated Learning using Hypernetworks | [Arxiv](https://arxiv.org/abs/2103.04628) - [GitHub](https://github.com/AvivSham/pFedHN) |
| CFL | arXiv | 2019 | Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints | [Arxiv](https://arxiv.org/abs/1910.01991) - [REF](https://github.com/KarhouTam/FL-bench/blob/master/src/server/cfl.py) |
| LGFedAvg | arXiv | 2020 | Think Locally, Act Globally: Federated Learning with Local and Global Representations | [Arxiv](https://arxiv.org/abs/2001.01523) - [REF](https://github.com/KarhouTam/FL-bench/blob/master/src/server/lgfedavg.py) |
| FedNova | NeurIPS | 2020 | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization | [Arxiv](https://arxiv.org/abs/2007.07481) - [REF](https://github.com/JYWa/FedNova) |
| FedSelect | CVPR | 2024 | FedSelect: Personalized Federated Learning with Customized Selection of Parameters for Fine-Tuning | [Arxiv](https://arxiv.org/abs/2404.02478) - [GitHub](https://github.com/lapisrocks/fedselect) |

## hFL — Heterogeneous Federated Learning

| Name | Venue | Year | Paper | URL |
|------|-------|------| ----- | ----|
| FedMD | NeurIPS-W | 2019 | FedMD: Heterogenous Federated Learning via Model Distillation | [Arxiv](https://arxiv.org/abs/1910.03581) |
| FedDF* | NeurIPS | 2020 | Ensemble Distillation for Robust Model Fusion in Federated Learning | [Arxiv](https://arxiv.org/abs/2006.07242) |

## dFL — Decentralized Federated Learning

| Name | Venue | Year | Paper | URL |
|------|-------|------| ----- | ----|
| DFedAvg** | AISTATS | 2017 | Communication-Efficient Learning of Deep Networks from Decentralized Data | [Arxiv](https://arxiv.org/abs/1602.05629) |
| DFedProx** | MLsys | 2020 | Federated Optimization in Heterogeneous Networks | [Arxiv](https://arxiv.org/abs/1812.06127) |
| DFedSAM | ICML | 2023 | Improving the Model Consistency of Decentralized Federated Learning | [Arxiv](https://arxiv.org/abs/2302.04083) - [Slide](https://icml.cc/media/icml-2023/Slides/25139.pdf) |
| DFedAWA** | CVPR | 2025 | FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors | [CVPR](https://cvpr.thecvf.com/virtual/2025/poster/33857) - [Arxiv](https://arxiv.org/abs/2503.15842) |
| DFedHPO | Internet of Things | 2025 | Decentralized Federated Learning with Hyperparameter Optimization | [PUB](https://doi.org/10.1016/j.iot.2024.101476) |

---

\* Adapted from classification to regression. Please use with caution.

\*\* Decentralized variant converted from its tFL/pFL counterpart (e.g. FedAvg → DFedAvg).
