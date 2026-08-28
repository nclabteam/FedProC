# Strategies

The category tree follows the reusable implementation bases; `tFL` is the root.

```mermaid
flowchart TD
    tFL --> qFL
    tFL --> ptFL
    tFL --> peftFL
    tFL --> sFL
    tFL --> pFL
    tFL --> aFL
    tFL --> spFL
    pFL --> mFL
    pFL --> hFL
    pFL --> dFL
    pFL --> nFL
```

## nFL — No Federated Learning

Standalone baselines and non-FL pre-training methods. No model communication; each client trains independently or is evaluated in a centralized setting.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| **InfoTS†** | AAAI | 2023 | Information-aware binary-concrete augmentation selection with global/local contrastive pretraining and ridge forecasting | Time Series Contrastive Learning with Information-Aware Augmentations | [PUB](https://doi.org/10.1609/aaai.v37i4.25575) - [Arxiv](https://arxiv.org/abs/2303.11911) - [GITHUB](https://github.com/chengw07/InfoTS) |
| **SimTS†** | ICASSP | 2024 | Positive-only latent future prediction with stop-gradient, then frozen-encoder ridge forecasting | Simple Contrastive Representation Learning for Time Series Forecasting | [PUB](https://doi.org/10.1109/ICASSP48485.2024.10446875) - [Arxiv](https://arxiv.org/abs/2303.18205) - [GITHUB](https://github.com/xingyu617/SimTS_Representation_Learning) |
| **SL** | NeurIPS | 2025 | Channel-wise selective MSE using residual-entropy uncertainty and DLinear residual-lower-bound anomaly masks | Selective Learning for Deep Time Series Forecasting | [PUB](https://doi.org/10.52202/085713-3277) - [NeurIPS](https://papers.neurips.cc/paper_files/paper/2025/hash/8cf54ff53f44835b9bdab2c546a1ca6d-Abstract-Conference.html) - [Arxiv](https://arxiv.org/abs/2510.25207) - [GITHUB](https://github.com/GestaltCogTeam/selective-learning) |
| **Centralized** | | | Oracle server model trained over every incumbent client's data with no FL communication | | |
| **LocalOLS** | | | Per-client local OLS regression, no communication | | |
| **LocalOnly** | | | Per-client local training only, no communication | | |

## tFL — Traditional Federated Learning

Central-server FL where a server aggregates client updates each round and broadcasts a single global model. All clients converge toward one shared solution.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| **FedAvg** | AISTATS | 2017 | Weighted average of client model updates | Communication-Efficient Learning of Deep Networks from Decentralized Data | [Arxiv](https://arxiv.org/abs/1602.05629) |
| **FedAvgM** | NeurIPS FL Workshop | 2019 | FedAvg with server-side Polyak momentum on model updates | Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification | [PUB](https://research.google/pubs/measuring-the-effects-of-non-identical-data-distribution-for-federated-visual-classification/) - [Arxiv](https://arxiv.org/abs/1909.06335) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavgm.py) |
| **FedNova** | NeurIPS | 2020 | Normalizes local updates by effective steps to fix objective inconsistency | Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization | [PUB](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html) - [Arxiv](https://arxiv.org/abs/2007.07481) - [GITHUB](https://github.com/JYWa/FedNova) |
| **FedProx** | MLSys | 2020 | Single shared model with a proximal local objective anchored to the downloaded global model | Federated Optimization in Heterogeneous Networks | [PUB](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html) - [Arxiv](https://arxiv.org/abs/1812.06127) - [GITHUB](https://github.com/litian96/FedProx) |
| **SCAFFOLD** | ICML | 2020 | Variance-reduced FL via per-client control variates | SCAFFOLD: Stochastic Controlled Averaging for Federated Learning | [PUB](https://proceedings.mlr.press/v119/karimireddy20a.html) - [Arxiv](https://arxiv.org/abs/1910.06378) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverscaffold.py) |
| **DLSA** | JCGS | 2021 | Federated weighted least-squares via precision-matrix aggregation | Least-Square Approximation for a Distributed System | [PUB](https://doi.org/10.1080/10618600.2021.1923517) - [Arxiv](https://arxiv.org/abs/1908.04904) |
| **FedAdam** | ICLR | 2021 | Server-side Adam moments and uncorrected adaptive global update | Adaptive Federated Optimization | [PUB](https://openreview.net/forum?id=LkFG3lB13U5) - [Arxiv](https://arxiv.org/abs/2003.00295) - [GITHUB](https://github.com/google-research/federated/tree/master/optimization) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedadam.py) |
| **FedDyn** | ICLR | 2021 | Dynamic client-gradient state with a quadratic local anchor; the uniform active-model mean is corrected by the mean client state to produce one global model | Federated Learning Based on Dynamic Regularization | [PUB](https://openreview.net/forum?id=B7v4QMR6Z9w) - [Arxiv](https://arxiv.org/abs/2111.04263) - [GITHUB](https://github.com/alpemreacar/FedDyn) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverdyn.py) |
| **FedSPA** | IJCAI | 2021 | Client rand-k sparse noisy-gradient updates and sparse-delta upload; unweighted server adaptive update over u/v moments | Federated Learning with Sparsification-Amplified Privacy and Adaptive Optimization | [PUB](https://doi.org/10.24963/ijcai.2021/202) - [Arxiv](https://arxiv.org/abs/2008.01558) |
| **FedYogi** | ICLR | 2021 | Server-side Yogi sign-controlled second moment and uncorrected adaptive update | Adaptive Federated Optimization | [PUB](https://openreview.net/forum?id=LkFG3lB13U5) - [Arxiv](https://arxiv.org/abs/2003.00295) - [GITHUB](https://github.com/google-research/federated/tree/master/optimization) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedyogi.py) |
| **MOON†** | CVPR | 2021 | Model-contrastive regularization with global and previous local models | Model-Contrastive Federated Learning | [PUB](https://doi.org/10.1109/CVPR46437.2021.01057) - [Arxiv](https://arxiv.org/abs/2103.16257) - [GITHUB](https://github.com/QinbinLi/MOON) |
| **FedADMM** | ICDE | 2022 | Persistent dual variables, augmented-model delta upload, and tracked global ADMM update | FedADMM: A Robust Federated Deep Learning Framework with Adaptivity to System Heterogeneity | [PUB](https://doi.org/10.1109/ICDE53745.2022.00238) - [Arxiv](https://arxiv.org/abs/2204.03529) - [GITHUB](https://github.com/YonghaiGong/FedADMM) - [REF](https://github.com/KarhouTam/FL-bench/blob/master/src/server/fedadmm.py) |
| **Elastic** | CVPR | 2023 | Element-wise elastic updates from unlabeled output-norm parameter sensitivity | Elastic Aggregation for Federated Optimization | [PUB](https://doi.org/10.1109/CVPR52729.2023.01173) - [OpenAccess](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Elastic_Aggregation_for_Federated_Optimization_CVPR_2023_paper.html) - [REF](https://github.com/KarhouTam/FL-bench/blob/master/src/server/elastic.py) |
| **FedLAW** | ICML | 2023 | Proxy-data optimization of normalized client weights and a global shrinking factor | Revisiting Weighted Aggregation in Federated Learning with Neural Networks | [PUB](https://proceedings.mlr.press/v202/li23s.html) - [Arxiv](https://arxiv.org/abs/2302.10911) - [GITHUB](https://github.com/ZexiLee/ICML-2023-FedLAW) |
| **FedOBD†** | IJCAI | 2023 | Mean-block-difference ranking uploads only the most-changed blocks, quantized by adaptive deterministic quantization, in a two-stage schedule | FedOBD: Opportunistic Block Dropout for Efficiently Training Large-scale Neural Networks through Federated Learning | [PUB](https://doi.org/10.24963/ijcai.2023/394) - [Arxiv](https://arxiv.org/abs/2208.05174) - [GITHUB](https://github.com/cyyever/distributed_learning_simulator) |
| **Caesar†** | arXiv | 2024 | Staleness-aware top-K+1-bit model download with client recovery; importance-ranked gradient upload sparsification | Caesar: A Low-deviation Compression Approach for Efficient Federated Learning | [Arxiv](https://arxiv.org/abs/2412.19989) |
| **FedCross** | ICDE | 2024 | Random dispatch and pairwise cross-aggregation of persistent middleware model slots | FedCross: Towards Accurate Federated Learning via Multi-Model Cross-Aggregation | [PUB](https://doi.org/10.1109/ICDE60146.2024.00170) - [Arxiv](https://arxiv.org/abs/2210.08285) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/servercross.py) |
| **FedRCL†** | CVPR | 2024 | Relaxed supervised contrastive loss with per-pair divergence penalty | Relaxed Contrastive Learning for Federated Learning | [PUB](https://doi.org/10.1109/CVPR52733.2024.01167) - [OpenAccess](https://openaccess.thecvf.com/content/CVPR2024/html/Seo_Relaxed_Contrastive_Learning_for_Federated_Learning_CVPR_2024_paper.html) - [Arxiv](https://arxiv.org/abs/2401.04928) - [GITHUB](https://github.com/skynbe/FedRCL) |
| **DeComFL‡** | ICLR | 2025 | Zeroth-order optimization: clients upload only scalar loss-difference gradients (dimension-free uplink); server reconstructs the update from shared perturbation seeds | Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization | [PUB](https://openreview.net/forum?id=omrLHFzC37) - [Arxiv](https://arxiv.org/abs/2405.15861) - [GITHUB](https://github.com/ZidongLiu/DeComFL) |
| **FedAWA** | CVPR | 2025 | Server optimization of persistent aggregation weights using client update vectors | FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors | [PUB](https://doi.org/10.1109/CVPR52734.2025.02854) - [OpenAccess](https://openaccess.thecvf.com/content/CVPR2025/html/Shi_FedAWA_Adaptive_Optimization_of_Aggregation_Weights_in_Federated_Learning_Using_CVPR_2025_paper.html) - [Arxiv](https://arxiv.org/abs/2503.15842) - [GITHUB](https://github.com/ChanglongShi/FedAWA) |
| **FedLUAR** | NeurIPS | 2025 | Recycles previous-round updates for a subset of layers, chosen via inverse-magnitude probability sampling, to cut uplink | Layer-wise Update Aggregation with Recycling for Communication-Efficient Federated Learning | [PUB](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7ccae68f81c1bd9104c304a9d8967048-Abstract-Conference.html) - [OpenReview](https://openreview.net/forum?id=t6EPMcudln) - [Arxiv](https://arxiv.org/abs/2503.11146) - [GITHUB](https://github.com/swblaster/FedLUAR) |
| **FedRidge** | arXiv | 2026 | One-shot federated ridge regression via sufficient statistic aggregation | One-Shot Federated Ridge Regression: Exact Recovery via Sufficient Statistic Aggregation | [Arxiv](https://arxiv.org/abs/2601.08216) |
| **FedTrend** | Science China Information Sciences | 2026 | Client-trajectory synthetic data for local transfer plus global-trajectory data for server refinement | Tackling Data Heterogeneity in Federated Time Series Forecasting | [PUB](https://doi.org/10.1007/s11432-025-4553-x) - [Arxiv](https://arxiv.org/abs/2411.15716) |

## qFL — Quantized Federated Learning

The quantization branch of `tFL`. Both implementations quantize one whole client update and apply its unweighted mean at the server; their quantizers remain strategy-specific static methods.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| **FedPAQ** | AISTATS | 2020 | Unweighted periodic averaging with stochastic whole-update-vector quantization | FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization | [PUB](https://proceedings.mlr.press/v108/reisizadeh20a.html) - [Arxiv](https://arxiv.org/abs/1909.13014) |
| **QATFL** | IEEE TMLCN | 2026 | Affine fake-quantization + STE during local training; affine-quantized whole-delta upload | Communication Efficient Federated Learning With Quantization-Aware Training Design | [PUB](https://doi.org/10.1109/TMLCN.2025.3635050) |

## ptFL — Partial-Training Federated Learning

Model-heterogeneous FL where the server owns a full model but each client
trains and communicates only a capacity-matched submodel or parameter subset.
The server retains each extraction mask and applies the aggregation rule
published by the selected method.

Two transports are implemented, and every ptFL strategy inherits one of them.

| | `ptFL` / `ptFL_Client` (physical submodel) | `ptFLUpdate` / `ptFLUpdate_Client` (masked update) |
|---|---|---|
| Downlink | the extracted narrow submodel, plus `capacity` and any `depth_layers` | the full model, plus `trainable_spec` when `_pt_send_spec` is set |
| Client model | a genuinely smaller dense `nn.Module` rebuilt from `capacity` | the full model, with a masked backward pass |
| Uplink | the trained narrow submodel, plus `score` when `_pt_send_score` is set | only the trained coordinates of the update, as `difference[mask]` |
| Index cost | none — the client's model is dense, so the server keeps the mask | none — the server generated the mask and still holds it |

Communication accounting follows the payload, not the Python package: only the
keys named in a package's `__wire__` tuple are measured. Optimizer state,
scheduler state and `personal_model_params` are framework transport for
stateless workers and are excluded, because no ptFL paper transmits them; a
value that a paper does put on the wire is always listed. Downlink is summed
per selected client rather than counted once, so heterogeneous submodel widths
are charged individually.

Two consequences are worth stating because they are easy to misread as
under-counting. `FedLAGC` pays for indices downlink (the client cannot derive a
magnitude-based mask from a submodel) but not uplink (the server generated that
mask), and its correction vector is client-local state in the paper, so the
round-trip that carries it for stateless workers is not charged. `FedPLT` sends
no spec at all: its allocation is a deterministic function of the parameter
counts and the client id, so both sides derive the same one for free. In the
other direction, `HASA`'s one-time `int32` count-vector upload happens before
training and is charged to the first round that reports a cost, because the
paper counts it as HASA's own overhead.

Physical submodel extraction needs a hidden-width axis to cut. The output-only
TSF models (`DLinear`, `DishLinear`, `Linear`, `NLinear`, `RLinear`) have none,
so pairing them with a fractional `capacity` is rejected rather than silently
run at full width: a full-width run reported as partial training is not a result
any of these papers describes. Use `capacity=1.0` for those models, or a
recurrent model (`GRU`, `LSTM`) for real partial training.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| **FedDropout** | arXiv | 2018 | Fresh random fixed-width physical submodels reduce client compute and communication | Expanding the Reach of Federated Learning by Reducing Client Resource Requirements | [Arxiv](https://arxiv.org/abs/1812.07210) - [REF](https://github.com/AIoT-MLSys-Lab/FedRolex) |
| **FedRolex** | NeurIPS | 2022 | Unit-stride rolling physical submodels with selective per-coordinate averaging | FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction | [Arxiv](https://arxiv.org/abs/2212.01548) - [GitHub](https://github.com/AIoT-MLSys-Lab/FedRolex) |
| **FLuID†** | NeurIPS | 2023 | Non-straggler percent weight change marks invariant neurons; stragglers receive physical submodels that drop them | FLuID: Mitigating Stragglers in Federated Learning using Invariant Dropout | [Arxiv](https://arxiv.org/abs/2307.02623) - [GITHUB](https://github.com/iwang05/FLuID) |
| **FedPMT†** | IEEE TNSE | 2025 | Full-model forward pass, contiguous deep-layer back-propagation, partial-update uplink, and contributor-only layer aggregation | Federated Learning with Partial Model Training | [PUB](https://doi.org/10.1109/TNSE.2025.3577910) - [Arxiv](https://arxiv.org/abs/2311.10002) |
| **FedLAGC†** | AAAI | 2026 | Layer-magnitude allocation extracts sparse client submodels; STE and client-local historical correction guide masked updates | FedLAGC: Towards High Performance System-Heterogeneous Federated Learning via Layer-Adaptive Submodel Extraction and Gradient Correction | [PUB](https://doi.org/10.1609/aaai.v40i26.39338) - [GITHUB](https://github.com/huqing2023/FedLAGC26) |
| **FedPLT** | arXiv | 2026 | Full-model forward pass with fixed rotating sublayer updates and sample-weighted per-block aggregation | FedPLT: Scalable, Resource-Efficient, and Heterogeneity-Aware Federated Learning via Partial Layer Training | [Arxiv](https://arxiv.org/abs/2605.02337) |
| **HASA** | IEEE EDGE | 2026 | One-time token-distribution JSD assigns fixed prefix widths under a size-weighted compute budget | HASA: Subnet Allocation for Compute-Constrained Model-Heterogeneous Federated Learning | [Arxiv](https://arxiv.org/abs/2606.07621) |

## peftFL — Parameter-Efficient Federated Learning

The parameter-efficient branch of `tFL`. The shared base installs LoRA adapters, restricts training and payloads to the selected factors, and keeps reusable workers logically stateless. `FedSA_LoRA` also inherits `pFL` because its LoRA-B factors are client-owned state persisted by the server.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| **FedIT†** | ICASSP | 2024 | Trains both LoRA factors and sample-weight averages only those adapters | Towards Building the Federated GPT: Federated Instruction Tuning | [PUB](https://sigport.org/documents/towards-building-federated-gpt-federated-instruction-tuning) - [Arxiv](https://arxiv.org/abs/2305.05644) - [GITHUB](https://github.com/JayZhang42/FederatedGPT-Shepherd) |
| **FFA_LoRA†** | ICLR | 2024 | Freezes the shared random LoRA-A initialization and trains and aggregates only LoRA-B | Improving LoRA in Privacy-preserving Federated Learning | [PUB](https://openreview.net/forum?id=NLPzL6HWNl) - [Arxiv](https://arxiv.org/abs/2403.12313) |
| **FlexLoRA†** | NeurIPS | 2024 | Sample-weight averages clients' scaled full LoRA updates, then redistributes rank-specific SVD factors | Federated Fine-tuning of Large Language Models under Heterogeneous Tasks and Client Resources | [PUB](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1a134b50202088aa8c595cc99b310e5a-Abstract-Conference.html) - [Arxiv](https://arxiv.org/abs/2402.11505) - [GITHUB](https://github.com/alibaba/FederatedScope/tree/FlexLoRA) |
| **FedSA_LoRA†** | ICLR | 2025 | Trains both factors, aggregates shared LoRA-A, and retains each client's LoRA-B on the server | Selective Aggregation for Low-Rank Adaptation in Federated Learning | [PUB](https://openreview.net/forum?id=iX3uESGdsO) - [Arxiv](https://arxiv.org/abs/2410.01463) - [GITHUB](https://github.com/Pengxin-Guo/FedSA-LoRA) |
| **LoRA_FAIR†** | ICCV | 2025 | Sample-weight averages both factors, then optimizes a LoRA-B residual to align the factorized and averaged full updates | LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement | [PUB](https://openaccess.thecvf.com/content/ICCV2025/html/Bian_LoRA-FAIR_Federated_LoRA_Fine-Tuning_with_Aggregation_and_Initialization_Refinement_ICCV_2025_paper.html) - [Arxiv](https://arxiv.org/abs/2411.14961) - [GITHUB](https://github.com/jmbian/LoRA-FAIR) |

## sFL — Security-Aware Federated Learning

Central-server FL with a pluggable threat injection seam between client training and aggregation. Strategies that inherit from `sFL` get the seam for free and can be evaluated under any registered attack (Byzantine model poisoning, sign-flip, etc.) by flipping `--attack` at run time without changing strategy code.

Strategies that inherit from `sFL` implement a specific defense; more families can be added without touching the injection seam.

Use `--attack <name>` and `--malicious_frac <f>` to switch between benign and adversarial evaluation — see [Usage § Adversarial Eval](usage.md#adversarial-eval).

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| **Krum†** | NeurIPS | 2017 | Unweighted Krum/Multi-Krum selection over client model uploads with paper-valid neighbor counts | Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent | [PUB](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) - [Arxiv](https://arxiv.org/abs/1703.02757) - [REF](https://github.com/adap/flower/blob/main/framework/py/flwr/server/strategy/krum.py) |
| **FedMedian†** | ICML | 2018 | Unweighted usual coordinate-wise median of client model uploads | Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | [PUB](https://proceedings.mlr.press/v80/yin18a.html) - [Arxiv](https://arxiv.org/abs/1803.01498) - [REF](https://github.com/adap/flower/blob/main/framework/py/flwr/server/strategy/fedmedian.py) |
| **FedTrimmedAvg†** | ICML | 2018 | Unweighted coordinate-wise beta-trimmed mean of client model uploads | Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | [PUB](https://proceedings.mlr.press/v80/yin18a.html) - [Arxiv](https://arxiv.org/abs/1803.01498) - [REF](https://github.com/adap/flower/blob/main/framework/py/flwr/server/strategy/fedtrimmedavg.py) |

## pFL — Personalized Federated Learning

Personalized FL produces a client-specific model through persistent local state, client-specific aggregation, or adaptation of a learned shared initialization. A method need not retain a single deployable global model (FedAMP does not).

FedProC uses stateless workers, so the server persists and transports logical client-local state in each selected client's package. The update equations and per-client ownership remain the same. Measured communication includes that state unless a strategy's `__wire__` contract narrows accounting to the paper's stated payload.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| **LGFedAvg** | NeurIPS FL Workshop | 2019 | Trains a persistent client-local encoder and sample-weighted shared suffix predictor end-to-end; only the shared suffix is communicated | Think Locally, Act Globally: Federated Learning with Local and Global Representations | [Arxiv](https://arxiv.org/abs/2001.01523) - [GITHUB](https://github.com/pliang279/LG-FedAvg) - [REF](https://github.com/KarhouTam/FL-bench/blob/master/src/server/lgfedavg.py) |
| **APFL** | arXiv | 2020 | Local descent at the parameter mixture αv+(1−α)w with one adaptive-α update per round and uniform global-model averaging | Adaptive Personalized Federated Learning | [Arxiv](https://arxiv.org/abs/2003.13461) - [GITHUB](https://github.com/MLOPTPSU/FedTorch) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverapfl.py) |
| **pFedMe** | NeurIPS | 2020 | K inner proximal steps and an ηλ outer update per fresh mini-batch, followed by uniform averaging and the paper's β server blend | Personalized Federated Learning with Moreau Envelopes | [PUB](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f4f1f13c8289ac1b1ee0ff176b56fc60-Abstract.html) - [Arxiv](https://arxiv.org/abs/2006.08848) - [GITHUB](https://github.com/CharlieDinh/pFedMe) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverpFedMe.py) |
| **CFL** | IEEE TNNLS | 2021 | Recursively complete-linkage clusters all clients by update cosine similarity, then performs sample-weighted FedAvg within each cluster | Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints | [PUB](https://doi.org/10.1109/TNNLS.2020.3015958) - [Arxiv](https://arxiv.org/abs/1910.01991) - [GITHUB](https://github.com/felisat/clustered-federated-learning) - [REF](https://github.com/KarhouTam/FL-bench/blob/master/src/server/cfl.py) |
| **Ditto** | ICML | 2021 | Persistent personalized model optimized with a proximal anchor to the global model; only the independent global update is aggregated | Ditto: Fair and Robust Federated Learning Through Personalization | [PUB](https://proceedings.mlr.press/v139/li21h.html) - [Arxiv](https://arxiv.org/abs/2012.04221) - [GITHUB](https://github.com/litian96/ditto) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverditto.py) |
| **FedAMP** | AAAI | 2021 | Maintains one personalized client/cloud model per client and uses the paper's distance-derived convex attention followed by proximal local training | Personalized Cross-Silo Federated Learning on Non-IID Data | [PUB](https://doi.org/10.1609/aaai.v35i9.16960) - [Arxiv](https://arxiv.org/abs/2007.03797) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serveramp.py) |
| **FedBN** | ICLR | 2021 | Keeps every batch-normalization parameter and running-statistic buffer local while uniformly averaging all non-BN model state | FedBN: Federated Learning on Non-IID Features via Local Batch Normalization | [PUB](https://openreview.net/forum?id=6YEQUn0QICG) - [Arxiv](https://arxiv.org/abs/2102.07623) - [GITHUB](https://github.com/med-air/FedBN) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverbn.py) |
| **pFedHN** | ICML | 2021 | A server-only hypernetwork emits one full personalized model to one sampled client, which returns only its initial-minus-trained update | Personalized Federated Learning using Hypernetworks | [PUB](https://proceedings.mlr.press/v139/shamsian21a.html) - [Arxiv](https://arxiv.org/abs/2103.04628) - [GITHUB](https://github.com/AvivSham/pFedHN) |
| **pFedLA** | CVPR | 2022 | Dedicated per-client hypernetworks learn block-wise weights over incumbent client models; HeurpFedLA retains the top-K self-weighted blocks locally | Layer-Wised Model Aggregation for Personalized Federated Learning | [PUB](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Layer-Wised_Model_Aggregation_for_Personalized_Federated_Learning_CVPR_2022_paper.html) - [Arxiv](https://arxiv.org/abs/2205.03993) - [REF](https://github.com/KarhouTam/pFedLA) |
| **FedALA** | AAAI | 2023 | Preserves lower local layers and learns element-wise old-local/global mixing weights for upper-layer initialization before local training | FedALA: Adaptive Local Aggregation for Personalized Federated Learning | [PUB](https://doi.org/10.1609/aaai.v37i9.26330) - [Arxiv](https://arxiv.org/abs/2212.01197) - [GITHUB](https://github.com/TsingZ0/FedALA) |
| **FedCAC** | ICCV | 2023 | Top-τ sensitivity masks drive full-client uniform global averaging and overlap-thresholded customized collaboration | Bold but Cautious: Unlocking the Potential of Personalized Federated Learning through Cautiously Aggressive Collaboration | [OpenAccess](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Bold_but_Cautious_Unlocking_the_Potential_of_Personalized_Federated_Learning_ICCV_2023_paper.html) - [Arxiv](https://arxiv.org/abs/2309.11103) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/servercac.py) |
| **FML†** | Frontiers of Information Technology & Electronic Engineering | 2023 | Resets each forked meme model/optimizer from the global state, performs detached-target bidirectional KL learning, and uniformly merges every client meme model | Federated mutual learning: a collaborative machine learning method for heterogeneous data, models, and objectives | [PUB](https://doi.org/10.1631/FITEE.2300098) - [Arxiv](https://arxiv.org/abs/2006.16765) - [GITHUB](https://github.com/ZJU-DAI/Federated-Mutual-Learning) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/clients/clientfml.py) |
| **FedSelect†** | CVPR | 2024 | Alternates personalized/global local updates, grows per-weight masks, and averages each remaining global coordinate over its participating clients | FedSelect: Personalized Federated Learning with Customized Selection of Parameters for Fine-Tuning | [PUB](https://openaccess.thecvf.com/content/CVPR2024/html/Tamirisa_FedSelect_Personalized_Federated_Learning_with_Customized_Selection_of_Parameters_for_CVPR_2024_paper.html) - [Arxiv](https://arxiv.org/abs/2404.02478) - [GITHUB](https://github.com/lapisrocks/fedselect) |
| **FedFew†** | CVPR | 2026 | Full-participation STCH-Set jointly updates three server models from every client's model-wise losses and gradients, then deploys each client's minimum-loss model | Few-for-Many Personalized Federated Learning | [OpenAccess](https://openaccess.thecvf.com/content/CVPR2026/html/Guo_Few-for-Many_Personalized_Federated_Learning_CVPR_2026_paper.html) - [Arxiv](https://arxiv.org/abs/2603.11992) - [GITHUB](https://github.com/pgg3/FedFew) |
| **PFMCP** | Expert Systems with Applications | 2026 | Post-FedAvg local decoder and input-dependent global/local MoE gate, followed by client-local dynamically updated conformal intervals | Personalized Federated Learning with Mixture of Experts and Conformal Prediction for Household Energy Forecasting | [PUB](https://doi.org/10.1016/j.eswa.2025.130417) |

## mFL — Meta-Learning Personalized Federated Learning

The meta-learning branch of `pFL`. The server learns a shared initialization, and each client or unseen task personalizes it with local adaptation.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| **PerAvg** | NeurIPS | 2020 | Uniformly averages a shared meta-initialization trained with independent inner/meta mini-batches, then personalizes it by one local gradient step at evaluation | Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach | [PUB](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html) - [Arxiv](https://arxiv.org/abs/2002.07948) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverperavg.py) |
| **AirMetapFL†** | IEEE TCCN | 2026 | Error-feedback sparsification and partial-DFT over-the-air aggregation train a MAML initialization for one-step personalization to new tasks | Pre-Training and Personalized Fine-Tuning via Over-the-Air Federated Meta-Learning: Convergence-Generalization Trade-Offs | [PUB](https://doi.org/10.1109/TCCN.2025.3640114) - [Arxiv](https://arxiv.org/abs/2406.11569) |

## hFL — Heterogeneous Federated Learning

Clients have architecturally different models. Knowledge is transferred via a shared public dataset or distillation rather than direct parameter averaging.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| **FedMD†** | NeurIPS FL Workshop | 2019 | All clients average raw predictions on shared public batches, then alternate public-consensus digestion and private-data revisiting | FedMD: Heterogenous Federated Learning via Model Distillation | [PUB](https://neurips.cc/virtual/2019/15377) - [Arxiv](https://arxiv.org/abs/1910.03581) - [REF](https://github.com/Tzq2doc/FedMD) |
| **FedDF†** | NeurIPS | 2020 | Sample-weight initializes each architecture prototype, then server-side average-output distillation transfers knowledge from every selected model | Ensemble Distillation for Robust Model Fusion in Federated Learning | [PUB](https://proceedings.neurips.cc/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html) - [Arxiv](https://arxiv.org/abs/2006.07242) - [GITHUB](https://github.com/epfml/federated-learning-public-code) |

## dFL — Decentralized Federated Learning

Clients communicate directly with neighbors and aggregate only their local neighborhood. FedProC's server process simulates the peer network and owns each logical node's state so workers remain reusable; internal worker transport is excluded and topology-scaled peer model exchanges are counted.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| **DFedProx§** | MLSys | 2020 | DFedAvg neighbor mixing with the FedProx local proximal objective | Federated Optimization in Heterogeneous Networks | [PUB](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html) - [Arxiv](https://arxiv.org/abs/1812.06127) - [GITHUB](https://github.com/litian96/FedProx) |
| **DFedAvg** | IEEE TPAMI | 2023 | Multiple local SGD steps followed by simultaneous weighted neighbor mixing | Decentralized Federated Averaging | [PUB](https://doi.org/10.1109/TPAMI.2022.3196503) - [Arxiv](https://arxiv.org/abs/2104.11375) |
| **DFedSAM** | ICML | 2023 | Local SAM perturbation followed by one gossip step, or the paper's configurable Q-step MGS variant | Improving the Model Consistency of Decentralized Federated Learning | [PUB](https://proceedings.mlr.press/v202/shi23d.html) - [Arxiv](https://arxiv.org/abs/2302.04083) |
| **DFedAWA§** | CVPR | 2025 | Each node optimizes FedAWA Eq. 3 over its own received neighborhood models | FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors | [OpenAccess](https://openaccess.thecvf.com/content/CVPR2025/html/Shi_FedAWA_Adaptive_Optimization_of_Aggregation_Weights_in_Federated_Learning_Using_CVPR_2025_paper.html) - [Arxiv](https://arxiv.org/abs/2503.15842) - [GITHUB](https://github.com/ChanglongShi/FedAWA) |
| **DFedHPO** | Internet of Things | 2025 | One pre-training local search pass, one neighbor exchange, then CA/FA/MA aggregation per logical node | Consensus-Driven Hyperparameter Optimization for Accelerated Model Convergence in Decentralized Federated Learning | [PUB](https://doi.org/10.1016/j.iot.2024.101476) |

## aFL — Asynchronous Federated Learning

No synchronization barrier; clients run continuously and results are aggregated as they arrive. A server-side buffer collects K results before each aggregation step, decoupling client speed from global update frequency.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| **FedPSA** | Expert Systems with Applications | 2026 | Paper-default sensitivity sketch (`k=16`) + momentum thermometer (`Lq=50`) + softmax aggregation over a five-update async buffer | FedPSA: Modeling Behavioral Staleness in Asynchronous Federated Learning | [PUB](https://doi.org/10.1016/j.eswa.2026.133003) - [Arxiv](https://arxiv.org/abs/2602.15337) |

---

## spFL - Sparse/Pruning Federated Learning

Dynamic sparse training in FL: clients maintain a binary mask that zeroes selected weights each round. A cosine-decay schedule (`delta_T` adjustment rounds up to `T_end`) controls how aggressively the mask evolves.

FedProC keeps dense PyTorch tensors behind boolean masks. The topology equations are implemented, but this generic representation does not reproduce the papers' sparse-kernel compute savings or compressed tensor transport.

| Name | Venue | Year | Description | Paper | URL |
|------|-------|------|-------------|-------|-----|
| **FedDST** | AAAI | 2022 | Client magnitude-prune/gradient-grow followed by coordinate-wise sparse weighted averaging and server magnitude re-pruning | Federated Dynamic Sparse Training: Computing Less, Communicating Less, Yet Learning Better | [PUB](https://doi.org/10.1609/aaai.v36i6.20555) - [Arxiv](https://arxiv.org/abs/2112.09824) - [GITHUB](https://github.com/bibikar/feddst) - [REF](https://github.com/FedPruning/FedPruning/tree/main/api/distributed/feddst) |
| **PruneFL†** | TNNLS | 2022 | Protected high-magnitude core plus greedy squared-gradient/round-time architecture search | Model Pruning Enables Efficient Federated Learning on Edge Devices | [PUB](https://doi.org/10.1109/TNNLS.2022.3166101) - [Arxiv](https://arxiv.org/abs/1909.12326) - [GITHUB](https://github.com/jiangyuang/PruneFL) - [REF](https://github.com/FedPruning/FedPruning/tree/main/api/distributed/prunefl) |
| **FedTiny†** | ICDCS | 2023 | Output-to-input progressive block updates using client top-k inactive gradients and server magnitude prune/grow | Distributed Pruning Towards Tiny Neural Networks in Federated Learning | [PUB](https://doi.org/10.1109/ICDCS57875.2023.00036) - [Arxiv](https://arxiv.org/abs/2212.01977) - [REF](https://github.com/FedPruning/FedPruning/tree/main/api/distributed/fedtinyclean) |
| **FedMef†** | CVPR | 2024 | Squared-L2 budget-aware extrusion with the REX learning-rate floor, top-k gradient growth, and magnitude pruning | FedMef: Towards Memory-efficient Federated Dynamic Pruning | [PUB](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_FedMef_Towards_Memory-efficient_Federated_Dynamic_Pruning_CVPR_2024_paper.html) - [Arxiv](https://arxiv.org/abs/2403.14737) - [REF](https://github.com/FedPruning/FedPruning/tree/main/api/distributed/fedmef) |
| **FedSGC†** | ICLR PML Workshop | 2024 | Congruity-prioritized local prune/grow, sparse residual aggregation for nonparticipants, global repruning, and direction-map feedback | Gradient-Congruity Guided Federated Sparse Training | [PUB](https://iclr.cc/virtual/2024/20633) - [OpenReview](https://openreview.net/forum?id=KHDncjMdjJ) - [Arxiv](https://arxiv.org/abs/2405.01189) - [REF](https://github.com/FedPruning/FedPruning/tree/main/api/distributed/fedsgc) |
| **FedRTS** | NeurIPS | 2025 | Per-weight Beta posteriors combine global/client core votes and inactive top-gradient votes before Thompson top-K selection | FedRTS: Federated Robust Pruning via Combinatorial Thompson Sampling | [PUB](https://doi.org/10.52202/085713-5422) - [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eda9523faa5e7191aee1c2eaff669716-Abstract-Conference.html) - [Arxiv](https://arxiv.org/abs/2501.19122) - [GITHUB](https://github.com/Little0o0/FedRTS) - [REF](https://github.com/FedPruning/FedPruning/tree/main/api/distributed/fedrts) |

---

\* Adapted from classification to regression. Please use with caution.

§ Derived decentralized composition; the cited paper defines the central component.

† TSF adaptation or framework integration applied; see the exact deviation below.

‡ Architecture-driven deviation: FedProC's stateless-client design cannot support a mechanism the paper relies on. Not a bug, not a TSF adaptation — see the Implementation Notes table.

| Strategy | Type | Note |
|---|---|---|
| SimTS† | TSF integration | FedProC treats each local forecasting input window as one SimTS instance and fits the paper's frozen ridge stage directly into the model head; `ridge_alpha` is configurable because client data has no separate validation split for the paper's alpha search. |
| InfoTS† | TSF integration | FedProC treats local forecasting windows as contrastive instances and fits the paper's frozen ridge stage directly into the model head; `ridge_alpha` is configurable because client data has no separate validation split. |
| FedRCL† | TSF adaptation | Pseudo-labels via quantile binning replace class labels; no multi-level contrastive hooks (TSF has no intermediate features). |
| FML† | TSF adaptation | KL divergence is computed over the time dimension instead of classes, and the personal/meme models use the same configured TSF architecture because FedProC has no per-client architecture slot. |
| AirMetapFL† | Implementation note | The paper leaves sparse estimator `E` generic and reports OAMP in experiments; FedProC uses dependency-free iterative hard thresholding over the same partial-DFT measurements. Uplink accounting uses complex64-equivalent analog symbols. |
| FedFew† | FL integration | The mean of the three server models is exposed only to FedProC's generic global metric. Personalized deployment still uses each client's minimum-loss model, and that choice does not feed back into training. |
| FedIT† / FFA_LoRA† / FedSA_LoRA† / LoRA_FAIR† / FlexLoRA† | TSF integration | FedProC applies each paper's LoRA update and ownership rules to configured `Linear` layers in its shared TSF model rather than the papers' LLM tasks. FFA-LoRA implements the core frozen-A method; a privacy guarantee still requires a separately configured DP optimizer. |
| FlexLoRA† | Implementation note | One scalar rank is configured per client; the official FederatedScope implementation can additionally assign ranks per adapter layer. The paper's scaled-update aggregation and rank-truncated SVD redistribution are unchanged. |
| FedSelect† | TSF adaptation | Forecast-loss gradients replace classification/fine-tuning gradients. Reusable workers receive the mask and local model in each server package; no logical personalized state is retained only on a worker. |
| FedLAGC† | TSF integration | The paper's input/output-layer, normalization, bias, layer-importance, STE, correction-vector, sparse transport, and overlap-update rules are applied to the configured TSF module graph instead of its ResNet benchmark. Dense PyTorch kernels do not reproduce sparse-kernel compute savings: the row/parameter masks are applied by a gradient hook after autograd has already produced the dense gradient, so the communication reduction is real but the sparse-backward FLOP reduction is not. |
| FedPLT† | Implementation note | The paper's sublayer selection reduces each client's update and uplink, but FedProC keeps the whole parameter trainable and zeroes inactive rows in a gradient hook, so the dense backward matrix multiply still runs. The reported communication saving is real; the partial-backward compute saving is not reproduced. The paper's optional FedPLT-aware client-selection extension is also out of scope; FedProC uses the framework's own client sampling. |
| ptFL / HASA† / FedLAGC† | Evaluation | FedProC's generalization metric is the global model on every client's test set, and its personalization metric is the client's *own* model on its own test set. Under partial training "the client's model" has two defensible readings that diverge whenever the allocation rotates, so every physical-partial strategy can report both beside the unchanged global metric. `resourcelevel_avg_*_loss` scores the current global model viewed at the client's assigned width, which is what the papers report -- FedLAGC's "global test accuracy under a given client resource level", HASA's "each client is evaluated at its allocated width r_i". A width is a fixed hardware property, so this is defined for every client whether or not it was selected. `personalization_avg_*_loss` scores the submodel each client was last actually sent, which is the model it really holds; with `join_ratio < 1` most clients did not receive one this round, and a rolling allocation like FedRolex's moves every round, so this record goes stale where the resource-level metric does not. Their gap is how far a client's held subnet has drifted from what its width could deliver now. A client that has never been selected has no record and its last-sent metric falls back to the global model, matching `pFL`'s own convention for a client with no personalized weights. Each metric costs a full extra `evaluate_personalized` pass over every incumbent client, per split, per evaluation round, so `--ptfl_local_metrics` selects which to pay for: it takes a comma-separated subset of `resourcelevel,personalization`, or `none` to leave only the global metric, and defaults to `personalization` alone -- the resource-level pass is opt-in because it doubles the cost of the round's evaluation. Disabling `personalization` also stops the server recording each sent submodel, which is a dense state dict per client. FedLAGC opts into the same pair with `theta * M` at its resource level. The masked-update strategies that leave untrained coordinates at the global value (FedPMT, FedPLT, FedOBD) hold the whole model locally, so their local model *is* the global model and neither extra metric is reported. Both run from `_post_eval_hook`, after aggregation and beside the generalization metric, so a subnet metric and the global metric in the same row describe the same server model. This is deliberately not `pFL`'s ordering: `pFL` scores personalized models from `_pre_eval_hook`, before the round trains, because the personal model a client holds at that point is the one it carried out of the previous round. Partial training has no equivalent pre-training reading worth reporting -- a width is a property of the client, and a last-sent subnet is only meaningful against the global model it was cut from. Early stopping and best-model selection still follow the global metric. Cross-paper metric names do not line up with these: FedRolex's "local model accuracy", for instance, is the *server* model on each client's local data, which is FedProC's generalization metric, not its personalization one. |
| FLuID† | Framework integration | FedProC follows the reference implementation's runtime protocol: each client reports its measured `duration`, the server designates the slowest as the straggler from round 2, and the retained ratio comes from the slowest-vs-target speedup through the reference's five-bucket `p_val` ladder, so the configured `capacity` list is unused. A neuron is invariant for a client when every one of its weights satisfies `|w(t) - w(t-1)| <= th*|w(t-1)|`, so a moved zero-valued broadcast weight can never be invariant. Invariance requires a strict majority of non-stragglers as the paper states, so `fluid_majority` defaults to 0.5 and the comparison excludes an even split; the reference's non-strict 75% remains reachable through that option. Drop priority follows the reference's three tiers (previously dropped, then invariant, then random) with `random.sample` inside each tier. Threshold handling follows the paper rather than the reference where the two disagree: each layer carries its own threshold ("FLuID can have a different drop threshold for each layer"), each is seeded from that layer's minimum percent update averaged over rounds 2 and 3, and each is raised every round "until the number of neurons below the threshold is greater than or equal to the number of neurons to be left out of the sub-model". The reference instead keeps one global threshold and adds a fixed 1.0 every third round after round 20 gated by a `stopChange` flag. The paper fixes no step size, so `fluid_threshold_step` exposes a relative one, and the search is bounded because a weight that moved off zero scores an infinite relative change no finite threshold admits. Aggregation is the reference's `aggregate_drop`: a per-coordinate average weighted by each contributor's sample count. FedProC's units are recurrent hidden units rather than the reference's dense-layer neurons. |
| FedOBD† | Framework integration | The paper's building blocks are its benchmark architectures' repeated units, which the reference implementation finds through a hard-coded whitelist of module classes (`Bottleneck`, `TransformerEncoderLayer`, `AlbertTransformer`, BatchNorm/Conv/ReLU runs) plus one block per remaining parameterized submodule. FedProC derives them from the module graph instead, treating each child of a `ModuleList`/`ModuleDict`/`Sequential` as one block and every remaining parameterized layer as a singleton block, because no whitelist covers its TSF models. The `offset` that the paper writes as an unbounded `argmin` is read as minimizing the shifted vector's infinity norm per its Further Optimization paragraph, giving `offset = -(max+min)/2`. The paper's stage budgets are independent: `iterations` is the stage-1 round count `R`, and the `fedobd_stage2_epochs` stage-2 epochs are charged on top of it rather than carved out of it, so the strategy runs `iterations + fedobd_stage2_epochs` outer rounds in total. Stage 2 runs all clients with block dropout off and one aggregation per epoch. |
| FedPMT† | Framework integration | FedProC implements the paper's server-initiated option: the full model and a contiguous deep-layer mask go downlink, only trained-layer updates go uplink, and each layer averages only its contributors. `fedpmt_depths=all` creates one level per parameterized layer; explicit depths such as `1,3,4` reproduce the paper's illustrated masks. Logical client IDs cycle through the configured levels to model the paper's equal capability groups. Depth is read off `named_modules()`, so the suffix is contiguous in module registration order; every model in this repository registers its parameterized modules in forward order, but a model that does not would receive a suffix that is not the forward-deep one. |
| FedMD† | TSF/stateless integration | Raw forecasts and MSE replace class logits and their matching loss. The server persists each logical client model solely because workers are reusable; communication accounting includes only public-batch predictions uplink and consensus downlink. Configured finite public/private epochs replace the paper's train-to-convergence initialization. |
| FedDF† | TSF adaptation | Raw forecasts and MSE replace softmax-logit KL. FedProC implements the paper's heterogeneous Algorithm 3 with one server-owned fused model per architecture prototype; full client models remain the measured payload. |
| MOON† | TSF adaptation | Flattened forecasts replace the paper's projection-head representation because FedProC's TSF models expose no common representation hook. |
| Krum† / FedMedian† / FedTrimmedAvg† | FL integration | The papers aggregate one gradient vector per worker and then take a server gradient step. FedProC follows the linked Flower integration by applying the same translation-equivariant, unweighted rules to locally trained model uploads; the papers' convergence guarantees therefore do not extend to multiple local steps or non-IID client objectives. |
| Caesar† | TSF adaptation | No class labels → KL term dropped. Uplink = COO sparse gradient; downlink = true compressed wire size. Paper defaults (§5.1): `theta_d_max=0.6`, `theta_u_min=0.1`, `theta_u_max=0.6`, `lambda=0.5`. |
| DeComFL‡ | Implementation note | Uplink faithful (dimension-free ZO scalars, `mu=0.001` matches paper). Downlink is **not** dimension-free like the paper: the paper's trick needs stateful clients to replay a model update from a shared random seed; FedProC's clients are stateless (respawned fresh each round) and cannot replay history, so the full model is sent downlink every round instead. `q=2`, `zo_lr=0.01` are TSF-adapted, not paper defaults. |
| PruneFL† | Framework integration | The paper's dedicated-client initial-pruning stage is replaced by the configured initial ERK mask. With no hardware layer timings, the greedy risk/time objective uses equal per-weight coefficients and a configurable `time_constant`. |
| HASA† | TSF integration | The published allocation scores each client by the Jensen-Shannon divergence between its token histogram and the global one. A time series has no vocabulary, so the shared support is the data factory's own integer count statistics: `hasa_count_bins` (default `n_neg,n_zero,n_pos`) names the bins, and the vector is those bins read across every column in sorted order. They partition each column's values exactly (`n_neg + n_zero + n_pos == count`), are identical in number and meaning for every client of a dataset, and are already computed during preprocessing, so no dataset needs regenerating. The divergence, allocation, budget, and wire format are unchanged. |
| FedTiny† | TSF integration | The progressive fine-pruning stage is retained, while the CNN batch-normalization candidate-selection stage is replaced by the configured initial ERK mask because generic TSF models expose no common BN architecture. |
| FedMef† | TSF integration | Budget-aware extrusion and topology adjustment are retained. Scaled activation pruning and NSConv are omitted because they require CNN-specific convolution and activation-cache hooks absent from generic TSF models. |
| FedSGC† | Framework integration | The paper's cumulative client-epoch schedule is folded into each server-requested adjustment round, with `A_epochs` selecting the pre-adjustment local epochs. Forecast-loss gradients replace classification gradients. |
