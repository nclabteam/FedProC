# Models
SSM: State Space Model  
Univariate (Persistent temporal patterns): encompassing trends and seasonal patterns  
Multivariate (Cross-variate information): correlations between different variables  
Auxiliary (eg: static time-varying features, future time-varying features, etc)

| Name | Backbone | Type | Venue | Year | Paper | URL |
|------|----------|------| ----- | ---- | ----- | --- |
| Amplifier | MLP | Multivariate | AAAI | 2025 | Amplifier: Bringing Attention to Neglected Low-Energy Components in Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2501.17216) |
| Linear | MLP | Univariate | AAAI | 2023 | Are Transformers Effective for Time Series Forecasting? | [Arxiv](https://arxiv.org/abs/2205.13504) |
| LinearIC | MLP | Univariate | AAAI | 2023 | Are Transformers Effective for Time Series Forecasting? | [Arxiv](https://arxiv.org/abs/2205.13504) |
| NLinear | MLP | Univariate | AAAI | 2023 | Are Transformers Effective for Time Series Forecasting? | [Arxiv](https://arxiv.org/abs/2205.13504) |
| DLinear | MLP | Univariate | AAAI | 2023 | Are Transformers Effective for Time Series Forecasting? | [Arxiv](https://arxiv.org/abs/2205.13504) |
| DLinearIC | MLP | Univariate | AAAI | 2023 | Are Transformers Effective for Time Series Forecasting? | [Arxiv](https://arxiv.org/abs/2205.13504) |
| DNGLinear | MLP |||| Bridging Simplicity and Sophistication using GLinear: A Novel Architecture for Enhanced Time Series Prediction | [Arxiv](https://arxiv.org/abs/2501.01087) |
| GLinear | MLP |||| Bridging Simplicity and Sophistication using GLinear: A Novel Architecture for Enhanced Time Series Prediction | [Arxiv](https://arxiv.org/abs/2501.01087) |
| FLinear | MLP || NeurIPS | 2024 | Frequency Adaptive Normalization For Non-stationary Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2409.20371) |
| FreTS | MLP | Multivariate | NeurIPS | 2023 | Frequency-domain MLPs are More Effective Learners in Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2311.06184) |
| LightTS | MLP | Multivariate | || Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures | [Arxiv](https://arxiv.org/abs/2207.01186) |
| MTSD | MLP | Univariate ||| MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing | [Arxiv](https://arxiv.org/abs/2302.04501) |
| MTSMatrix | MLP | Multivariate ||| MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing | [Arxiv](https://arxiv.org/abs/2302.04501) |
| MTSMixer | MLP | Multivariate ||| MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing | [Arxiv](https://arxiv.org/abs/2302.04501) |
| PaiFilter | MLP || NeurIPS | 2024 | FilterNet: Harnessing Frequency Filters for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2411.01623) |
| TexFilter | MLP || NeurIPS | 2024 | FilterNet: Harnessing Frequency Filters for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2411.01623) |
| RDLinear | MLP |||| RDLinear: A Novel Time Series Forecasting Model Based on Decomposition with RevIN | [IEEE](https://ieeexplore.ieee.org/document/10650961) |
| RLinear | MLP |||| Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping | [Arxiv](https://arxiv.org/abs/2305.10721) |
| CrossLinear | MLP | Multivariate | KDD | 2025 | CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables | [Arxiv](https://arxiv.org/abs/2505.23116) |
| UMixer | MLP | Multivariate | AAAI | 2024 | U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2401.02236) |
| TSMixer | MLP | Multivariate ||| TSMixer: An All-MLP Architecture for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2303.06053) |
| SWIFT | MLP | Univariate ||| SWIFT: Mapping Sub-series with Wavelet Decomposition Improves Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2501.16178) |
| SparseTSF | MLP | Univariate | ICML | 2024 | SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters | [Arxiv](https://arxiv.org/abs/2405.00946) |
| DishLinear | MLP || AAAI | 2023 | Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2302.14829) |
||||||||
| LSTM | RNN || Neural Computation | 1997 | Long Short-Term Memory | [ACM](https://dl.acm.org/doi/10.1162/neco.1997.9.8.1735) |
| GRU | RNN || EMNLP | 2014 | Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation | [Arxiv](https://arxiv.org/abs/1406.1078) |
| SegRNN | RNN | Univariate ||| SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2308.11200) |
| RWKV4TS | RNN |||| RWKV-TS: Beyond Traditional Recurrent Neural Network for Time Series Tasks | [Arxiv](https://arxiv.org/abs/2401.09093) | 
||||||||
| DSSRNN | SSM |||| DSSRNN: Decomposition-Enhanced State-Space Recurrent Neural Network for Time-Series Analysis | [Arxiv](https://arxiv.org/abs/2412.00994) |
||||||||
| MLCNN | CNN | Multivariate | AAAI | 2020 | Towards Better Forecasting by Fusing Near and Distant Future Visions | [Arxiv](https://arxiv.org/abs/1912.05122) |
| SCINet | CNN | Multivariate | NeurIPS | 2022 | SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction | [Arxiv](https://arxiv.org/abs/2106.09305) |
| ModernTCN | CNN || ICLR | 2024 | ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis | [OpenReview](https://openreview.net/forum?id=vpJMJerXHU) |
| xPatch | CNN | Univariate | AAAI | 2025 | xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition | [Arxiv](https://arxiv.org/abs/2412.17323) |
| TimePoint | CNN | Multivariate | ICML | 2025 | TimePoint: Accelerated Time Series Alignment via Self-Supervised Keypoint and Descriptor Learning | [OpenReview](https://openreview.net/forum?id=bUGdGaNFhi) [Arxiv](https://arxiv.org/abs/2505.23475) |
||||||||
| TimeKAN | KAN || ICLR | 2025 | TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2502.06910) |
| MMK | KAN | | | | Are KANsEffective for Multivariate Time Series Forecasting? | [Arxiv](https://arxiv.org/abs/2408.11306) | 
||||||||
| CARD | Transformer | | ICLR | 2024 | CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2305.12095) | 
||||||||
| Timer | Foundation || ICML | 2024 | Timer: Generative Pre-trained Transformers Are Large Time Series Models | [Arxiv](https://arxiv.org/abs/2402.02368) |
| PAttn | LLM || NeurIPS | 2024| Are Language Models Actually Useful for Time Series Forecasting?| [Arxiv](https://arxiv.org/abs/2406.16964) |
| CALF | LLM | | AAAI | 2025 | CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning | [Arxiv](https://arxiv.org/abs/2403.07300) |
||||||||
| ConvRNN ||| | | Autoregressive Convolutional Recurrent Neural Network for Univariate and Multivariate Time Series Prediction | [Arxiv](https://arxiv.org/abs/1907.04155) |
| S3 ||||| Segment, Shuffle, and Stitch: A Simple Layer for Improving Time-Series Representations | [Arxiv](https://arxiv.org/abs/2405.20082) |
| FAN ||| || FAN: Fourier Analysis Networks | [Arxiv](https://arxiv.org/abs/2410.02675) |
| FiLM ||| NeurIPS | 2022 | FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2205.08897) |
| Leddam ||| ICML | 2024 | Revitalizing Multivariate Time Series Forecasting: Learnable Decomposition with Inter-Series Dependencies and Intra-Series Variations Modeling | [Arxiv](https://arxiv.org/abs/2402.12694) | 
