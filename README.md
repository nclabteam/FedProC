# FedProC: Federated Prognos Chronos
FedProC is a federated learning framework for time-series forecasting, incorporating various federated aggregation strategies, datasets, and models. It supports multiple forecasting models, datasets with a focus on **personalized federated learning**.


## Implemented Options
### Strategies
| Name | Venue | Year | Paper | URL |
|------|-------|------| ----- | ----|
| LocalOnly | | | | | 
| Centralized | | | | |
| FedAvg | AISTATS | 2017 | Communication-Efficient Learning of Deep Networks from Decentralized Data | [Arxiv](https://arxiv.org/abs/1602.05629) |
| FedMedian | ICML | 2018 | Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | [Arxiv](https://arxiv.org/abs/1803.01498) |
| FedTrimmedAvg | ICML | 2018 | Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | [Arxiv](https://arxiv.org/abs/1803.01498) |
| FedAdam | ICLR | 2021 | Adaptive Federated Optimization | [Arxiv](https://arxiv.org/abs/2003.00295) |
| FedYogi | ICLR | 2021 | Adaptive Federated Optimization |  [Arxiv](https://arxiv.org/abs/2003.00295) |
| Krum | | | Byzantine-Tolerant Machine Learning | [Arxiv](https://arxiv.org/abs/1703.02757) |
| FedAvgM | | | Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification | [Arxiv](https://arxiv.org/abs/1909.06335) |
| FedProx | MLsys | 2020| Federated Optimization in Heterogeneous Networks | [Arxiv](https://arxiv.org/abs/1812.06127) |
| FedALA | AAAI | 2023 | FedALA: Adaptive Local Aggregation for Personalized Federated Learning | [Arxiv](https://arxiv.org/abs/2212.01197) |
| FedCAC | ICCV | 2023 | Bold but Cautious: Unlocking the Potential of Personalized Federated Learning through Cautiously Aggressive Collaboration | [OpenAccess](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Bold_but_Cautious_Unlocking_the_Potential_of_Personalized_Federated_Learning_ICCV_2023_paper.html) - [Arxiv](https://arxiv.org/abs/2309.11103) |
| Elastic | CVPR | 2023 | Elastic Aggregation for Federated Optimization | [OpenAccess](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Elastic_Aggregation_for_Federated_Optimization_CVPR_2023_paper.html) |

### Datasets
| Name | Domain | Granularity | Variates | Clients (max) | Samples | 
| ---- | ------ | ----------- | -------- | ------------- | ------- |
| BaseStation5G |  | 2 minutes | 11 | 3 | 4_192-15_927|
| BeijingAirQuality | Environment | 1 hour | 11 | 12 | |
| COVID19Cases | Healthcare | 1 day | 10 | 55 | |
| CryptoDataDownloadDay | Economic | 1 day | 4 | | |
| CryptoDataDownloadHour | Economic | 1 hour | 4 | | |
| CryptoDataDownloadMinute | Economic | 1 minute | 4 | | |
| ElectricityLoadDiagrams | Energy | 15 minutes | 1 | 370 | 26_304 |
| ETDatasetHour | Energy | 1 hour | 7 | 2 | 14_400 | 
| ETDatasetMinute | Energy | 15 minutes | 7 | 2 | 57_600 |
| MekongSalinity | | | | | |
| PeMSSF | Traffic | 10 minutes | 1 | 963 | 63_345 |
| SolarEnergy | Energy | 1 hour | 1 | 137 | 52_560 |
| Traffic | | | | 862 | 17_544 |
| Weather5K | Environment | 1 hour | 5 | 5_672 ||

**Note**: Number of clients will be decided after spliiting the data since clients with insuffienct data (cannot form at least 10 samples) with be discarded. `Clients (max)` is the maximum number of clients possible.

### Models
SSM: State Space Model  
Univariate (Persistent temporal patterns): encompassing trends and seasonal patterns  
Multivariate (Cross-variate information): correlations between different variables  
Auxiliary (eg: static time-varying features, future time-varying features, etc)

| Name | Backbone | Type | Venue | Year | Paper | URL |
|------|----------|------| ----- | ---- | ----- | --- |
| Amplifier | MLP | Multivariate | AAAI | 2025 | Amplifier: Bringing Attention to Neglected Low-Energy Components in Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2501.17216) |
| ConvRNN ||| | | Autoregressive Convolutional Recurrent Neural Network for Univariate and Multivariate Time Series Prediction | [Arxiv](https://arxiv.org/abs/1907.04155) |
| Linear | MLP | Univariate | AAAI | 2023 | Are Transformers Effective for Time Series Forecasting? | [Arxiv](https://arxiv.org/abs/2205.13504) |
| NLinear | MLP | Univariate | AAAI | 2023 | Are Transformers Effective for Time Series Forecasting? | [Arxiv](https://arxiv.org/abs/2205.13504) |
| DLinear | MLP | Univariate | AAAI | 2023 | Are Transformers Effective for Time Series Forecasting? | [Arxiv](https://arxiv.org/abs/2205.13504) |
| DNGLinear ||||| Bridging Simplicity and Sophistication using GLinear: A Novel Architecture for Enhanced Time Series Prediction | [Arxiv](https://arxiv.org/abs/2501.01087) |
| GLinear ||||| Bridging Simplicity and Sophistication using GLinear: A Novel Architecture for Enhanced Time Series Prediction | [Arxiv](https://arxiv.org/abs/2501.01087) |
| DSSRNN | SSM |||| DSSRNN: Decomposition-Enhanced State-Space Recurrent Neural Network for Time-Series Analysis | [Arxiv](https://arxiv.org/abs/2412.00994) |
| FAN ||| || FAN: Fourier Analysis Networks | [Arxiv](https://arxiv.org/abs/2410.02675) |
| FiLM ||| NeurIPS | 2022 | FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2205.08897) |
| FreTS | MLP | Multivariate | NeurIPS | 2023 | Frequency-domain MLPs are More Effective Learners in Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2311.06184) |
| Leddam ||| ICML | 2024 | Revitalizing Multivariate Time Series Forecasting: Learnable Decomposition with Inter-Series Dependencies and Intra-Series Variations Modeling | [Arxiv](https://arxiv.org/abs/2402.12694) | 
| LightTS | MLP | Multivariate | || Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures | [Arxiv](https://arxiv.org/abs/2207.01186) |
| LSTM |||||||
| MLCNN | CNN | Multivariate | AAAI | 2020 | Towards Better Forecasting by Fusing Near and Distant Future Visions | [Arxiv](https://arxiv.org/abs/1912.05122) |
| MTSD ||||| MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing | [Arxiv](https://arxiv.org/abs/2302.04501) |
| MTSMatrix ||||| MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing | [Arxiv](https://arxiv.org/abs/2302.04501) |
| MTSMixer ||||| MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing | [Arxiv](https://arxiv.org/abs/2302.04501) |
| PaiFilter ||| NeurIPS | 2024 | FilterNet: Harnessing Frequency Filters for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2411.01623) |
| TexFilter ||| NeurIPS | 2024 | FilterNet: Harnessing Frequency Filters for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2411.01623) |
| RLinear ||||| Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping | [Arxiv](https://arxiv.org/abs/2305.10721) |
| SCINet | CNN | Multivariate | NeurIPS | 2022 | SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction | [Arxiv](https://arxiv.org/abs/2106.09305) |
| SegRNN | RNN | Univariate ||| SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2308.11200) |
| SparseTSF | MLP | Univariate | ICML | 2024 | SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters | [Arxiv](https://arxiv.org/abs/2405.00946) |
| SWIFT | MLP | Univariate ||| SWIFT: Mapping Sub-series with Wavelet Decomposition Improves Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2501.16178) |
| TSMixer | MLP | Multivariate ||| TSMixer: An All-MLP Architecture for Time Series Forecasting | [Arxiv](https://arxiv.org/abs/2303.06053) |
| xPatch | CNN | Univariate | AAAI | 2025 | xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition | [Arxiv](https://arxiv.org/abs/2412.17323) |

## Installation

### Linux
```bash
apt install python3-virtualenv
```
```bash
virtualenv venv --python=python3.10
```
```bash
source venv/bin/activate
```
```bash
pip install --upgrade pip
```
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Window
```bash
pip install virtualenv
```
```bash
virtualenv venv --python=python3.10
```
```bash
.\venv\Scripts\activate
```
```bash
pip install --upgrade pip
```
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Usage
```bash
python main.py
```

---

## Customization

## Framework

## Code Formatting
### Linux
```bash
bash code_formatting.sh
```
### Window
```bash
sh code_formatting.sh
```
---