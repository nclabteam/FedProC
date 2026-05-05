# Strategies
| Name | Venue | Year | Paper | URL |
|------|-------|------| ----- | ----|
| LocalOnly | | | | | 
| Centralized | | | | |
|
| DFL | | | | |
| DFedSAM | ICML | 2023 | Improving the Model Consistency of Decentralized Federated Learning | [Arxiv](https://arxiv.org/abs/2302.04083) - [Slide](https://icml.cc/media/icml-2023/Slides/25139.pdf) |
|
| FedAvg | AISTATS | 2017 | Communication-Efficient Learning of Deep Networks from Decentralized Data | [Arxiv](https://arxiv.org/abs/1602.05629) |
| FedMedian | ICML | 2018 | Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | [Arxiv](https://arxiv.org/abs/1803.01498) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedmedian.py) |
| FedTrimmedAvg | ICML | 2018 | Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | [Arxiv](https://arxiv.org/abs/1803.01498) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedtrimmedavg.py) |
| MOON | CVPR | 2021 | Model-Contrastive Federated Learning | [Arxiv](https://arxiv.org/abs/2103.16257) |
| FedAdam | ICLR | 2021 | Adaptive Federated Optimization | [Arxiv](https://arxiv.org/abs/2003.00295) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedadam.py) |
| FedYogi | ICLR | 2021 | Adaptive Federated Optimization | [Arxiv](https://arxiv.org/abs/2003.00295) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedyogi.py) |
| Krum | | | Byzantine-Tolerant Machine Learning | [Arxiv](https://arxiv.org/abs/1703.02757) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/krum.py) |
| FedAvgM | | | Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification | [Arxiv](https://arxiv.org/abs/1909.06335) - [REF](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavgm.py) |
| FedProx / DFedProx | MLsys | 2020 | Federated Optimization in Heterogeneous Networks | [Arxiv](https://arxiv.org/abs/1812.06127) |
| FML | NeurIPS | 2020 | Federated Mutual Learning | [Arxiv](https://arxiv.org/abs/2006.16765) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/clients/clientfml.py) |
| FedDyn | ICLR | 2021 | Federated Learning Based on Dynamic Regularization | [Arxiv](https://arxiv.org/abs/2111.04263) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/serverdyn.py) |
| FedCross | ICDE | 2024 | FedCross: Towards Accurate Federated Learning via Multi-Model Cross-Aggregation | [IEEE](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10597740) - [Arxiv](https://arxiv.org/abs/2210.08285) - [REF](https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/servers/servercross.py) |
| FedAWA / DFedAWA | CVPR | 2025 | FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors | [CVPR](https://cvpr.thecvf.com/virtual/2025/poster/33857) - [Arxiv](https://arxiv.org/abs/2503.15842) |
| Elastic | CVPR | 2023 | Elastic Aggregation for Federated Optimization | [OpenAccess](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Elastic_Aggregation_for_Federated_Optimization_CVPR_2023_paper.html) - [REF](https://github.com/KarhouTam/FL-bench/blob/master/src/server/elastic.py) |
| FedRCL | CVPR | 2024 | Relaxed Contrastive Learning for Federated Learning | [OpenAccess](https://openaccess.thecvf.com/content/CVPR2024/html/Seo_Relaxed_Contrastive_Learning_for_Federated_Learning_CVPR_2024_paper.html) - [Arxiv](https://arxiv.org/abs/2401.04928) - [GitHub](https://github.com/skynbe/FedRCL) |
|
| FedALA | AAAI | 2023 | FedALA: Adaptive Local Aggregation for Personalized Federated Learning | [Arxiv](https://arxiv.org/abs/2212.01197) |
| FedCAC | ICCV | 2023 | Bold but Cautious: Unlocking the Potential of Personalized Federated Learning through Cautiously Aggressive Collaboration | [OpenAccess](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Bold_but_Cautious_Unlocking_the_Potential_of_Personalized_Federated_Learning_ICCV_2023_paper.html) - [Arxiv](https://arxiv.org/abs/2309.11103) |
| FedTrend |  |  | Tackling Data Heterogeneity in Federated Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2411.15716) |
|
| FedDF* | NeurIPS | 2020 | Ensemble Distillation for Robust Model Fusion in Federated Learning | [Arxiv](https://arxiv.org/abs/2006.07242) |
| FedMD | NeurIPS-W | 2019 | FedMD: Heterogenous Federated Learning via Model Distillation | [Arxiv](https://arxiv.org/abs/1910.03581) |
| FDCR | NeurIPS | 2024 | Parameter Disparities Dissection for Backdoor Defense in Heterogeneous Federated Learning | [Arxiv](https://arxiv.org/abs/2404.10332) |

\* Adapted from classification to regression (original uses softmax + KL-divergence; we use MSE). Please use with caution.
