# Models
SSM: State Space Model  
Univariate (Persistent temporal patterns): encompassing trends and seasonal patterns  
Multivariate (Cross-variate information): correlations between different variables  
Auxiliary (eg: static time-varying features, future time-varying features, etc)

| Name | Backbone | Type | Venue | Year | Paper | URL |
|------|----------|------| ----- | ---- | ----- | --- | --- |
| Amplifier | MLP | Multivariate | AAAI | 2025 | Amplifier: Bringing Attention to Neglected Low-Energy Components in Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2501.17216) - [REF](https://github.com/aikunyi/Amplifier/blob/main/models/Amplifier.py) |
| Linear | MLP | Univariate | AAAI | 2023 | Are Transformers Effective for Time Series Forecasting? | [Arxiv](https://arxiv.org/abs/2205.13504) - [REF](https://github.com/cure-lab/LTSF-Linear/blob/main/models/Linear.py) |
| LinearIC | MLP | Univariate | AAAI | 2023 | Are Transformers Effective for Time Series Forecasting? | [Arxiv](https://arxiv.org/abs/2205.13504) - [REF](https://github.com/cure-lab/LTSF-Linear/blob/main/models/Linear.py) |
| NLinear | MLP | Univariate | AAAI | 2023 | Are Transformers Effective for Time Series Forecasting? | [Arxiv](https://arxiv.org/abs/2205.13504) - [REF](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py) |
| DLinear | MLP | Univariate | AAAI | 2023 | Are Transformers Effective for Time Series Forecasting? | [Arxiv](https://arxiv.org/abs/2205.13504) - [REF](https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py) |
| DLinearIC | MLP | Univariate | AAAI | 2023 | Are Transformers Effective for Time Series Forecasting? | [Arxiv](https://arxiv.org/abs/2205.13504) - [REF](https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py) |
| DNGLinear | MLP |||| Bridging Simplicity and Sophistication using GLinear: A Novel Architecture for Enhanced Time Series Prediction | [Arxiv](https://arxiv.org/abs/2501.01087) - [REF](https://github.com/t-rizvi/GLinear/blob/main/models/WIthout_Normalization/DNGLinear.py) |
| GLinear | MLP |||| Bridging Simplicity and Sophistication using GLinear: A Novel Architecture for Enhanced Time Series Prediction | [Arxiv](https://arxiv.org/abs/2501.01087) - [REF](https://github.com/t-rizvi/GLinear/blob/main/models/WIthout_Normalization/DNGLinear.py) |
| FLinear | MLP || NeurIPS | 2024 | Frequency Adaptive Normalization For Non-stationary Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2409.20371) |
| FreTS | MLP | Multivariate | NeurIPS | 2023 | Frequency-domain MLPs are More Effective Learners in Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2311.06184) - [REF](https://github.com/aikunyi/FilterNet/blob/main/models/FreTS.py) |
| LightTS | MLP | Multivariate | || Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures | [Arxiv](https://arxiv.org/abs/2207.01186) - [REF](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py) |
| MTSD | MLP | Univariate ||| MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing | [Arxiv](https://arxiv.org/abs/2302.04501) - [REF](https://github.com/plumprc/MTS-Mixers/blob/main/models/MTSD.py) |
| MTSMatrix | MLP | Multivariate ||| MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing | [Arxiv](https://arxiv.org/abs/2302.04501) - [REF](https://github.com/plumprc/MTS-Mixers/blob/main/models/MTSMatrix.py) |
| MTSMixer | MLP | Multivariate ||| MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing | [Arxiv](https://arxiv.org/abs/2302.04501) - [REF](https://github.com/plumprc/MTS-Mixers/blob/main/models/MTSMixer.py) |
| PaiFilter | MLP || NeurIPS | 2024 | FilterNet: Harnessing Frequency Filters for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2411.01623) - [REF](https://github.com/aikunyi/FilterNet/blob/main/models/PaiFilter.py) |
| TexFilter | MLP || NeurIPS | 2024 | FilterNet: Harnessing Frequency Filters for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2411.01623) - [REF](https://github.com/aikunyi/FilterNet/blob/main/models/TexFilter.py) |
| RDLinear | MLP |||| RDLinear: A Novel Time Series Forecasting Model Based on Decomposition with RevIN | [IEEE](https://ieeexplore.ieee.org/document/10650961) |
| RLinear | MLP |||| Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping | [Arxiv](https://arxiv.org/abs/2305.10721) - [REF](https://github.com/plumprc/RTSF/blob/main/models/RLinear.py) |
| CrossLinear | MLP | Multivariate | KDD | 2025 | CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables | [Arxiv](https://arxiv.org/abs/2505.23116) - [REF](https://github.com/mumiao2000/CrossLinear/blob/main/models/CrossLinear.py) |
| UMixer | MLP | Multivariate | AAAI | 2024 | U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2401.02236) - [REF](https://github.com/XiangMa-Shaun/U-Mixer/blob/main/models/UMixer.py) |
| TSMixer | MLP | Multivariate ||| TSMixer: An All-MLP Architecture for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2303.06053) - [REF](https://github.com/decisionintelligence/TFB/blob/master/ts_benchmark/baselines/time_series_library/models/TSMixer.py) |
| SWIFT | MLP | Univariate ||| SWIFT: Mapping Sub-series with Wavelet Decomposition Improves Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2501.16178) - [REF](https://github.com/LancelotXWX/SWIFT/blob/main/models/SWIFT_Linear.py) |
| SparseTSF | MLP | Univariate | ICML | 2024 | SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters | [Arxiv](https://arxiv.org/abs/2405.00946) - [REF](https://github.com/lss-1138/SparseTSF/blob/main/models/SparseTSF.py) |
| DishLinear | MLP || AAAI | 2023 | Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2302.14829) |
| FSMLP | MLP |||| FSMLP: Frequency Simplex MLP for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2412.01654) |
| FITS | MLP | Univariate | ICLR | 2024 | FITS: Modeling Time Series with 10k Parameters| [Arxiv](https://arxiv.org/abs/2307.03756) - [REF](https://github.com/VEWOXIC/FITS/blob/main/models/FITS.py) |
| RFITS | MLP | Univariate | ICLR | 2024 | FITS: Modeling Time Series with 10k Parameters| [Arxiv](https://arxiv.org/abs/2307.03756) - [REF](https://github.com/VEWOXIC/FITS/blob/main/models/Real_FITS.py) |
|||||||
|||||||
| LSTM | RNN || Neural Computation | 1997 | Long Short-Term Memory | [ACM](https://dl.acm.org/doi/10.1162/neco.1997.9.8.1735) |
| GRU | RNN || EMNLP | 2014 | Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation | [Arxiv](https://arxiv.org/abs/1406.1078) |
| SegRNN | RNN | Univariate ||| SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2308.11200) |
| RWKV4TS | RNN |||| RWKV-TS: Beyond Traditional Recurrent Neural Network for Time Series Tasks | [Arxiv](https://arxiv.org/abs/2401.09093) - [REF](https://github.com/howard-hou/RWKV-TS/blob/main/Long-term_Forecasting/models/RWKV4TS.py) |
|||||||
|||||||
| DSSRNN | SSM |||| DSSRNN: Decomposition-Enhanced State-Space Recurrent Neural Network for Time-Series Analysis | [Arxiv](https://arxiv.org/abs/2412.00994) - [REF](https://github.com/ahmad-shirazi/DSSRNN/blob/main/models/DSSRNN.py) |
|||||||
|||||||
| MLCNN | CNN | Multivariate | AAAI | 2020 | Towards Better Forecasting by Fusing Near and Distant Future Visions | [Arxiv](https://arxiv.org/abs/1912.05122) - [REF](https://github.com/smallGum/MLCNN-Multivariate-Time-Series/blob/master/models/models.py) |
| SCINet | CNN | Multivariate | NeurIPS | 2022 | SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction | [Arxiv](https://arxiv.org/abs/2106.09305) - [REF](https://github.com/plumprc/MTS-Mixers/blob/main/models/SCINet.py) |
| ModernTCN | CNN || ICLR | 2024 | ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis | [OpenReview](https://openreview.net/forum?id=vpJMJerXHU) - [REF](https://github.com/luodhhh/ModernTCN/blob/main/ModernTCN-Long-term-forecasting/models/ModernTCN.py) |
| xPatch | CNN | Univariate | AAAI | 2025 | xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition | [Arxiv](https://arxiv.org/abs/2412.17323) |
| TimePoint | CNN | Multivariate | ICML | 2025 | TimePoint: Accelerated Time Series Alignment via Self-Supervised Keypoint and Descriptor Learning | [OpenReview](https://openreview.net/forum?id=bUGdGaNFhi) [Arxiv](https://arxiv.org/abs/2505.23475) - [REF](https://github.com/BGU-CS-VIL/TimePoint/blob/main/TimePoint/models/timepoint.py) |
|||||||
|||||||
| TimeKAN | KAN || ICLR | 2025 | TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2502.06910) - [REF](https://github.com/huangst21/TimeKAN/blob/main/models/TimeKAN.py) |
| MMK | KAN | | | | Are KANsEffective for Multivariate Time Series Forecasting? | [Arxiv](https://arxiv.org/abs/2408.11306) - [REF](https://github.com/2448845600/EasyTSF/blob/main/easytsf/model/MMK.py) |
|||||||
|||||||
| CrossFormer | Transformer | Multivariate | ICLR | 2023 | Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting | [OpenReview](https://openreview.net/forum?id=vSVLM2j9eie) - [REF](https://github.com/Thinklab-SJTU/Crossformer/blob/master/cross_models/cross_former.py) |
| PatchTST | Transformer | Univariate | ICLR | 2023 | A Time Series is Worth 64 Words: Long-term Forecasting with Transformers | [Arxiv](https://arxiv.org/abs/2211.14730) - [REF](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/models/PatchTST.py) |
| CARD | Transformer | | ICLR | 2024 | CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2305.12095) - [REF](https://github.com/wxie9/CARD/blob/main/long_term_forecast_l720/models/CARD.py) |
| PAttn | Transformer | Univariate | NeurIPS | 2024| Are Language Models Actually Useful for Time Series Forecasting?| [Arxiv](https://arxiv.org/abs/2406.16964) - [REF](https://github.com/BennyTMT/LLMsForTimeSeries/blob/main/PAttn/models/PAttn.py) |
|||||||
|||||||
| Timer | Foundation || ICML | 2024 | Timer: Generative Pre-trained Transformers Are Large Time Series Models | [Arxiv](https://arxiv.org/abs/2402.02368) - [REF](https://github.com/thuml/Large-Time-Series-Model/blob/main/models/Timer.py) |
| GPT4TS | LLM || NeurIPS | 2023 | One Fits All:Power General Time Series Analysis by Pretrained LM | [Arxiv](https://arxiv.org/abs/2302.11939) - [REF](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/main/Long-term_Forecasting/models/GPT4TS.py) |
| T54TS | LLM | | ICLR | 2024| TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting | [Arxiv](https://arxiv.org/pdf/2310.04948) - [REF](https://github.com/DC-research/TEMPO/blob/main/tempo/models/T5.py) |
| CALF | LLM | | AAAI | 2025 | CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning | [Arxiv](https://arxiv.org/abs/2403.07300) - [REF](https://github.com/Hank0626/CALF/blob/main/models/CALF.py) |
| VisionTS | Foundation | | ICML | 2025 | VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters | [Arxiv](https://arxiv.org/abs/2408.17253) - [REF](https://github.com/Keytoyze/VisionTS/blob/main/visionts/model.py) |
|||||||
|||||||
| ConvRNN ||| | | Autoregressive Convolutional Recurrent Neural Network for Univariate and Multivariate Time Series Prediction | [Arxiv](https://arxiv.org/abs/1907.04155) - [REF](https://github.com/KurochkinAlexey/ConvRNN/blob/master/ConvRNN_SML2010.ipynb) |
| S3 ||||| Segment, Shuffle, and Stitch: A Simple Layer for Improving Time-Series Representations | [Arxiv](https://arxiv.org/abs/2405.20082) - [REF](https://github.com/shivam-grover/S3-TimeSeries/blob/main/S3/S3.py) |
| FAN ||| || FAN: Fourier Analysis Networks | [Arxiv](https://arxiv.org/abs/2410.02675) - [REF](https://github.com/YihongDong/FAN/blob/main/Timeseries_Forecasting/layers/FANLayer.py) |
| FiLM ||| NeurIPS | 2022 | FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2205.08897) - [REF](https://github.com/decisionintelligence/TFB/blob/master/ts_benchmark/baselines/time_series_library/models/FiLM.py) |
| Leddam ||| ICML | 2024 | Revitalizing Multivariate Time Series Forecasting: Learnable Decomposition with Inter-Series Dependencies and Intra-Series Variations Modeling | [Arxiv](https://arxiv.org/abs/2402.12694) - [REF](https://github.com/Levi-Ackman/Leddam/blob/main/models/Leddam.py) |
