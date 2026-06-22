# Strategies

## nFL — No Federated Learning

Standalone baselines and non-FL pre-training methods. No model communication; each client trains independently or is evaluated in a centralized setting.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| LocalOnly | | | Per-client local training only, no communication | | |
| LocalOLS | | | Per-client local OLS regression, no communication | | |
| Centralized | | | Oracle upper bound — all data on one server | | |
| SimTS | ICASSP | 2024 | Contrastive pretraining with instance-pair selection for TSF | SimTS: Rethinking Contrastive Representation Learning for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2303.18205) - [IEEE](https://ieeexplore.ieee.org/document/10446875) - [GitHub](https://github.com/xingyu617/SimTS_Representation_Learning) |
| InfoTS | ICLR | 2023 | Meta-contrastive with learnable augmentation selection for TSF | InfoTS: Information-Aware Time Series Meta-Contrastive Learning | [Arxiv](https://arxiv.org/abs/2303.01186) - [REF](https://github.com/Sec-Sci-Lab/InfoTS) |
| SL | NeurIPS | 2025 | Dual-mask selective loss (uncertainty + anomaly) for deep TSF | Selective Learning for Deep Time Series Forecasting | [OpenReview](https://openreview.net/forum?id=kgzRy6nD6D) - [GitHub](https://github.com/GestaltCogTeam/selective-learning) |

## tFL — Traditional Federated Learning

Central-server FL where a server aggregates client updates each round and broadcasts a single global model. All clients converge toward one shared solution.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| FedAvg | AISTATS | 2017 | Weighted average of client model updates | Communication-Efficient Learning of Deep Networks from Decentralized Data | [Arxiv](https://arxiv.org/abs/1602.05629) |
| FedAvgM | | | FedAvg with server-side Polyak momentum on model updates | Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification | [Arxiv](https://arxiv.org/abs/1909.06335) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavgm.py) |
| SCAFFOLD | ICML | 2020 | Variance-reduced FL via per-client control variates | SCAFFOLD: Stochastic Controlled Averaging for Federated Learning | [Arxiv](https://arxiv.org/abs/1910.06378) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverscaffold.py) |
| FedNova | NeurIPS | 2020 | Normalizes local updates by effective steps to fix objective inconsistency | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization | [Arxiv](https://arxiv.org/abs/2007.07481) - [REF](https://github.com/JYWa/FedNova) |
| FedAdam | ICLR | 2021 | Server-side Adam momentum for global model update | Adaptive Federated Optimization | [Arxiv](https://arxiv.org/abs/2003.00295) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedadam.py) |
| FedYogi | ICLR | 2021 | Server-side Yogi optimizer (sign-scaled momentum) for global update | Adaptive Federated Optimization | [Arxiv](https://arxiv.org/abs/2003.00295) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedyogi.py) |
| Krum | NeurIPS | 2017 | Byzantine-robust aggregation via nearest-neighbor cluster selection | Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent | [PUB](https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/krum.py) |
| FedMedian | ICML | 2018 | Byzantine-robust coordinate-wise median aggregation | Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | [Arxiv](https://arxiv.org/abs/1803.01498) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedmedian.py) |
| FedTrimmedAvg | ICML | 2018 | Byzantine-robust trimmed mean aggregation | Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | [Arxiv](https://arxiv.org/abs/1803.01498) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedtrimmedavg.py) |
| FedPAQ | AISTATS | 2020 | Periodic averaging with stochastic quantization of update vectors | FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization | [Arxiv](https://arxiv.org/abs/1909.13014) |
| MOON | CVPR | 2021 | Model-contrastive regularization with global and previous local models | Model-Contrastive Federated Learning | [Arxiv](https://arxiv.org/abs/2103.16257) |
| FedADMM | ICDE | 2022 | ADMM-based FL robust to system heterogeneity and stragglers | FedADMM: A Robust Federated Deep Learning Framework with Adaptability to System Heterogeneity | [IEEE](https://ieeexplore.ieee.org/abstract/document/9835545) - [REF](https://github.com/KarhouTam/FL-bench/blob/master/src/server/fedadmm.py) |
| FedLAW | ICML | 2023 | Learned aggregation weights via auxiliary network on validation data | Revisiting Weighted Aggregation in Federated Learning with Neural Networks | [Arxiv](https://arxiv.org/abs/2302.10911) - [GitHub](https://github.com/ZexiLee/ICML-2023-FedLAW) |
| Elastic | CVPR | 2023 | Sensitivity-weighted elastic aggregation to preserve task-critical params | Elastic Aggregation for Federated Optimization | [OpenAccess](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Elastic_Aggregation_for_Federated_Optimization_CVPR_2023_paper.html) - [REF](https://github.com/KarhouTam/FL-bench/blob/master/src/server/elastic.py) |
| FedRolex | NeurIPS | 2022 | Heterogeneous FL via rolling sub-model extraction and cyclic training | FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction | [Arxiv](https://arxiv.org/abs/2211.11614) - [GitHub](https://github.com/AIoT-MLSys-Lab/FedRolex) |
| FedCross | ICDE | 2024 | Cross-aggregation of multiple global model candidates per round | FedCross: Towards Accurate Federated Learning via Multi-Model Cross-Aggregation | [IEEE](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10597740) - [Arxiv](https://arxiv.org/abs/2210.08285) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/servercross.py) |
| FedRCL† | CVPR | 2024 | Relaxed supervised contrastive loss with per-pair divergence penalty | Relaxed Contrastive Learning for Federated Learning | [OpenAccess](https://openaccess.thecvf.com/content/CVPR2024/html/Seo_Relaxed_Contrastive_Learning_for_Federated_Learning_CVPR_2024_paper.html) - [Arxiv](https://arxiv.org/abs/2401.04928) - [GitHub](https://github.com/skynbe/FedRCL) |
| FedAWA | CVPR | 2025 | Server-optimized per-client aggregation weights via client vectors | FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors | [CVPR](https://cvpr.thecvf.com/virtual/2025/poster/33857) - [Arxiv](https://arxiv.org/abs/2503.15842) |
| FedTrend | Science China Information Sciences | 2026 | Trend-aware aggregation for heterogeneous federated TSF | Tackling Data Heterogeneity in Federated Time Series Forecasting | [PUB](https://doi.org/10.1007/s11432-025-4553-x) - [Arxiv](https://arxiv.org/abs/2411.15716) |
| FedRidge | arXiv | 2026 | One-shot federated ridge regression via sufficient statistic aggregation | One-Shot Federated Ridge Regression: Exact Recovery via Sufficient Statistic Aggregation | [Arxiv](https://arxiv.org/abs/2601.08216) |
| FedDLSA | JCGS | 2021 | Federated weighted least-squares via precision-matrix aggregation | Least-Square Approximation for a Distributed System | [PUB](https://doi.org/10.1080/10618600.2021.1923517) - [Arxiv](https://arxiv.org/abs/1908.04904) |

## pFL — Personalized Federated Learning

Each client maintains its own model or model component alongside the global model. Aggregation is designed to produce client-specific outputs rather than a single shared solution.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| FedProx | MLsys | 2020 | Proximal term regularizes local objective toward global model | Federated Optimization in Heterogeneous Networks | [Arxiv](https://arxiv.org/abs/1812.06127) |
| Ditto | ICML | 2021 | Jointly trains global and personalized models with proximity constraint | Ditto: Fair and Robust Federated Learning Through Personalization | [Arxiv](https://arxiv.org/abs/2012.04235) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverditto.py) |
| pFedMe | NeurIPS | 2020 | Personalized model found via Moreau envelope of global objective | Personalized Federated Learning with Moreau Envelopes | [Arxiv](https://arxiv.org/abs/2006.08848) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverpFedMe.py) |
| APFL | arXiv | 2020 | Convex mixture of local and global models with adaptive α | Adaptive Personalized Federated Learning | [Arxiv](https://arxiv.org/abs/2003.13461) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverapfl.py) |
| PerAvg | NeurIPS | 2020 | MAML-style meta-learning for fast one-step personalization | Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach | [Arxiv](https://arxiv.org/abs/2002.07948) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverperavg.py) |
| FedAMP | AAAI | 2021 | Attention-mechanism aggregation for personalized cross-silo FL | Personalized Cross-Silo Federated Learning on Non-IID Data | [Arxiv](https://arxiv.org/abs/2007.03797) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serveramp.py) |
| FedBN | ICLR | 2021 | Keeps batch norm layers local to handle feature-distribution shift | FedBN: Federated Learning on Non-IID Features via Local Batch Normalization | [Arxiv](https://arxiv.org/abs/2102.07623) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverbn.py) |
| FML† | NeurIPS | 2020 | Mutual KL distillation between global and local model predictions | Federated Mutual Learning | [Arxiv](https://arxiv.org/abs/2006.16765) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/clients/clientfml.py) |
| FedDyn | ICLR | 2021 | Dynamic per-client regularization toward a moving global anchor | Federated Learning Based on Dynamic Regularization | [Arxiv](https://arxiv.org/abs/2111.04263) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverdyn.py) |
| CFL | arXiv | 2019 | Agglomerative clustering of clients; each cluster trains independently | Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints | [Arxiv](https://arxiv.org/abs/1910.01991) - [REF](https://github.com/KarhouTam/FL-bench/blob/master/src/server/cfl.py) |
| LGFedAvg | arXiv | 2020 | Splits model into shared global layers and personal local layers | Think Locally, Act Globally: Federated Learning with Local and Global Representations | [Arxiv](https://arxiv.org/abs/2001.01523) - [REF](https://github.com/KarhouTam/FL-bench/blob/master/src/server/lgfedavg.py) |
| pFedHN | ICML | 2021 | Hypernetwork generates personalized model weights per client | Personalized Federated Learning using Hypernetworks | [Arxiv](https://arxiv.org/abs/2103.04628) - [GitHub](https://github.com/AvivSham/pFedHN) |
| pFedLA | NeurIPS | 2022 | Layer-wise aggregation weights learned via hypernetwork | Layer-Wise Personalized Federated Learning via Hypernetworks | [Arxiv](https://arxiv.org/abs/2206.01542) - [GitHub](https://github.com/KarhouTam/pFedLA) |
| AirMetapFL | arXiv | 2025 | Over-the-air meta-learning with convergence–generalization tradeoff | Pre-Training and Personalized Fine-Tuning via Over-the-Air Federated Meta-Learning: Convergence-Generalization Trade-Offs | [Arxiv](https://arxiv.org/abs/2406.11569) |
| FedFew | arXiv | 2025 | K server models with STCH-Set soft assignment; near-optimal pFL via few-for-many | Few-for-Many Personalized Federated Learning | [Arxiv](https://arxiv.org/abs/2603.11992) |
| FedALA | AAAI | 2023 | Element-wise adaptive local aggregation via gradient alignment | FedALA: Adaptive Local Aggregation for Personalized Federated Learning | [Arxiv](https://arxiv.org/abs/2212.01197) |
| FedCAC | ICCV | 2023 | Sparsity-based selective collaboration with cautious aggregation | Bold but Cautious: Unlocking the Potential of Personalized Federated Learning through Cautiously Aggressive Collaboration | [OpenAccess](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Bold_but_Cautious_Unlocking_the_Potential_of_Personalized_Federated_Learning_ICCV_2023_paper.html) - [Arxiv](https://arxiv.org/abs/2309.11103) |
| FDCR | NeurIPS | 2024 | Parameter-disparity dissection for backdoor defense in heterogeneous FL | Parameter Disparities Dissection for Backdoor Defense in Heterogeneous Federated Learning | [Arxiv](https://arxiv.org/abs/2404.10332) - [GitHub](https://github.com/WenkeHuang/FDCR) |
| FedIT | ICASSP | 2024 | Federated instruction tuning with LoRA for large language models | Towards Building the Federated GPT: Federated Instruction Tuning | [Arxiv](https://arxiv.org/abs/2305.05644) |
| FFA_LoRA | ICLR | 2024 | Freezes LoRA-A, aggregates only LoRA-B for privacy-preserving fine-tuning | Improving LoRA in Privacy-preserving Federated Learning | [Arxiv](https://arxiv.org/abs/2403.12313) |
| FedSelect | CVPR | 2024 | Sparse parameter selection masks for efficient personalized fine-tuning | FedSelect: Personalized Federated Learning with Customized Selection of Parameters for Fine-Tuning | [Arxiv](https://arxiv.org/abs/2404.02478) - [GitHub](https://github.com/lapisrocks/fedselect) |
| FedSA_LoRA | ICLR | 2025 | Selectively aggregates task-relevant LoRA components | Selective Aggregation for Low-Rank Adaptation in Federated Learning | [Arxiv](https://arxiv.org/abs/2410.01463) |
| LoRA_FAIR | ICCV | 2025 | Refines LoRA aggregation weights and re-initializes A matrices | LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement | [Arxiv](https://arxiv.org/abs/2411.14961) - [GitHub](https://github.com/jmbian/LoRA-FAIR) |
| FlexLoRA | NeurIPS | 2024 | Heterogeneous LoRA ranks via SVD-based redistribution at server | Federated Fine-tuning of Large Language Models under Heterogeneous Tasks and Client Resources | [Arxiv](https://arxiv.org/abs/2402.11505) |

## hFL — Heterogeneous Federated Learning

Clients have architecturally different models. Knowledge is transferred via a shared public dataset or distillation rather than direct parameter averaging.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| FedMD | NeurIPS-W | 2019 | Knowledge distillation via public dataset for heterogeneous model architectures | FedMD: Heterogenous Federated Learning via Model Distillation | [Arxiv](https://arxiv.org/abs/1910.03581) |
| FedDF* | NeurIPS | 2020 | Ensemble distillation on public data refines server model post-aggregation | Ensemble Distillation for Robust Model Fusion in Federated Learning | [Arxiv](https://arxiv.org/abs/2006.07242) |

## dFL — Decentralized Federated Learning

No central server; clients communicate directly with neighbors in a peer-to-peer topology. Each node aggregates only from its local neighborhood.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| DFedAvg** | AISTATS | 2017 | Decentralized FedAvg on peer-to-peer topology | Communication-Efficient Learning of Deep Networks from Decentralized Data | [Arxiv](https://arxiv.org/abs/1602.05629) |
| DFedProx** | MLsys | 2020 | Decentralized FedProx with proximal regularization on P2P topology | Federated Optimization in Heterogeneous Networks | [Arxiv](https://arxiv.org/abs/1812.06127) |
| DFedSAM | ICML | 2023 | SAM-based local training with model-consistency regularization for decentralized FL | Improving the Model Consistency of Decentralized Federated Learning | [Arxiv](https://arxiv.org/abs/2302.04083) - [Slide](https://icml.cc/media/icml-2023/Slides/25139.pdf) |
| DFedAWA** | CVPR | 2025 | Decentralized FedAWA with local adaptive aggregation weights | FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors | [CVPR](https://cvpr.thecvf.com/virtual/2025/poster/33857) - [Arxiv](https://arxiv.org/abs/2503.15842) |
| DFedHPO | Internet of Things | 2025 | Pre-FL distributed hyperparameter optimization with neighbor consensus | Decentralized Federated Learning with Hyperparameter Optimization | [PUB](https://doi.org/10.1016/j.iot.2024.101476) |

## aFL — Asynchronous Federated Learning

No synchronization barrier; clients run continuously and results are aggregated as they arrive. A server-side buffer collects K results before each aggregation step, decoupling client speed from global update frequency.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| FedPSA | arXiv | 2025 | Sensitivity-sketch behavioral staleness + thermometer-controlled softmax aggregation | FedPSA: Modeling Behavioral Staleness in Asynchronous Federated Learning | [Arxiv](https://arxiv.org/abs/2602.15337) |

---

## spFL — Sparse Federated Learning

Dynamic sparse training in FL: clients maintain a binary mask that zeroes selected weights each round. A cosine-decay schedule (`delta_T` adjustment rounds up to `T_end`) controls how aggressively the mask evolves. FedAvg serves as the dense baseline; all other spFL strategies add a mask-update policy on top.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| FedAvg | AISTATS | 2017 | Dense baseline (see tFL group; listed here for benchmark reference) | Communication-Efficient Learning of Deep Networks from Decentralized Data | [Arxiv](https://arxiv.org/abs/1602.05629) - [REF](https://github.com/FedPruning/FedPruning/tree/main/api/distributed/fedavg) |
| PruneFL | TNNLS | 2022 | Server-guided mask: FedAvg of per-client squared gradients → prune smallest active + grow largest-grad² inactive | Model Pruning Enables Efficient Federated Learning on Edge Devices | [Arxiv](https://arxiv.org/abs/1909.12326) - [REF](https://github.com/FedPruning/FedPruning/tree/main/api/distributed/prunefl) |
| FedDST | AAAI | 2022 | Client-local mask update (sparse_update_step after A_epochs); server reconciles via OR-union + magnitude re-prune | Federated Dynamic Sparse Training: Computing Less, Communicating Less, Yet Learning Better | [Arxiv](https://arxiv.org/abs/2112.09824) - [REF](https://github.com/FedPruning/FedPruning/tree/main/api/distributed/feddst) |
| FedTiny | ICDCS | 2023 | Clients send post-training gradients; server FedAvg of gradients → server-side prune + grow (TinyClean variant) | Distributed Pruning Towards Tiny Neural Networks in Federated Learning | [Arxiv](https://arxiv.org/abs/2212.01977) - [REF](https://github.com/FedPruning/FedPruning/tree/main/api/distributed/fedtinyclean) |
| FedMef | CVPR | 2024 | FedTiny + BAE regularization (L2 on low-magnitude active weights) + optional topk-grad server filter | FedMef: Towards Memory-efficient Federated Dynamic Pruning | [Arxiv](https://arxiv.org/abs/2403.14737) - [REF](https://github.com/FedPruning/FedPruning/tree/main/api/distributed/fedmef) |
| FedSGC | ICLRW | 2024 | Direction-coherent local mask update: prune conflict-direction active, grow agreement-direction inactive; server OR-union + reprune + direction map | Gradient-Congruity Guided Federated Sparse Training | [Arxiv](https://arxiv.org/abs/2405.01189) - [REF](https://github.com/FedPruning/FedPruning/tree/main/api/distributed/fedsgc) |
| FedRTS | NeurIPS | 2025 | Per-weight Beta(α,β) distributions; clients vote on core active (top-κ mag) and grow candidates (top-k grad); server Thompson-samples new mask | FedRTS: Federated Robust Pruning via Combinatorial Thompson Sampling | [Arxiv](https://arxiv.org/abs/2501.19122) - [REF](https://github.com/FedPruning/FedPruning/tree/main/api/distributed/fedrts) |

---

\* Adapted from classification to regression. Please use with caution.

\*\* Decentralized variant converted from its tFL/pFL counterpart (e.g. FedAvg → DFedAvg).

† TSF adaptation applied: class-level assumptions replaced with time-series equivalents. Please use with caution.
  - **FedRCL†**: Pseudo-labels from quantile binning replace class labels; multi-level contrastive hooks not applied (TSF models have no intermediate feature extractors).
  - **FML†**: KL divergence computed over the time dimension instead of the class dimension (paper targets classification logits).
