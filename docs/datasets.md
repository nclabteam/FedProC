## Datasets
| Name | Domain | Granularity | Variates | Clients (max) | Samples | 
| ---- | ------ | ----------- | -------- | ------------- | ------- |
| BaseStation5G |  | 2 minutes | 11 | 3 | 4_192-15_927|
| BeijingAirQuality | Environment | 1 hour | 11 | 12 | |
| COVID19Cases | Healthcare | 1 day | 10 | 55 | |
| CitiesILI | | | | | |
| CryptoDataDownloadDay | Economic | 1 day | 4 | | |
| CryptoDataDownloadHour | Economic | 1 hour | 4 | | |
| CryptoDataDownloadMinute | Economic | 1 minute | 4 | | |
| ETTh1 | Energy | 1 hour | 7 | 1 | 14_400 | 
| ETTh2 | Energy | 1 hour | 7 | 1 | 14_400 | 
| ETDatasetHour | Energy | 1 hour | 7 | 2 | 14_400 | 
| ETTm1 | Energy | 15 minutes | 7 | 1 | 57_600 |
| ETTm2 | Energy | 15 minutes | 7 | 1 | 57_600 |
| ETDatasetMinute | Energy | 15 minutes | 7 | 2 | 57_600 |
| Electricity | Energy | 15 minutes | 1 | 321 | 26_304 |
| ElectricityLoadDiagrams | Energy | 15 minutes | 1 | 370 | 140_256 |
| METRLA | | 5 minutes | 1 | 207 | 34_272 |
| MekongSalinity | | | | | |
| PeMS03 | | 5 minutes | 1 | 358 | 26_208 |
| PeMS04 | | 5 minutes | 1 | 307 | 16_992 |
| PeMS07 | | 5 minutes | 1 | 883 | 28_224 |
| PeMS08 | | 5 minutes | 3 | 170 | 17_856 |
| PeMSBAY | | 5 minutes | 1 | 325 | 52_116 |
| PeMSSF | Traffic | 10 minutes | 1 | 963 | 63_345 |
| SolarEnergy | Energy | 1 hour | 1 | 137 | 52_560 |
| StatesILI | | | | | |
| TetouanPowerConsumption | | | | | |
| Traffic | | | | 862 | 17_544 |
| Weather5K | Environment | 1 hour | 5 | 5_672 ||

**Note**: Number of clients will be decided after spliiting the data since clients with insuffienct data (cannot form at least 10 samples) with be discarded. `Clients (max)` is the maximum number of clients possible.