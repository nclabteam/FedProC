# Datasets

## Available Datasets

| Name                     | Domain        | Granularity  | Variates | Clients (max) | Samples       | URL |
|--------------------------|---------------|--------------|----------|---------------|-------------- | --- |
| BaseStation5G            | Communication | 2 minutes    | 11       | 3             | 4_192-15_927  | [Github](https://github.com/vperifan/Federated-Time-Series-Forecasting) |
| BeijingAirQuality        | Environment   | 1 hour       | 11       | 12            |               | [UCI](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) |
| CitiesILI                | Healthcare    | 1 week       | 1        | 122           |               | [Github](https://github.com/emilylaiken/ml-flu-prediction) |
| COVID19Cases             | Healthcare    | 1 day        | 10       | 55            |               | [Github](https://github.com/ashfarhangi/COVID-19) |
| CryptoDataDownloadDay    | Economic      | 1 day        | 4        |               |               | [CDD](https://www.cryptodatadownload.com/data/binance/) |
| CryptoDataDownloadHour   | Economic      | 1 hour       | 4        |               |               | [CDD](https://www.cryptodatadownload.com/data/binance/) |
| CryptoDataDownloadMinute | Economic      | 1 minute     | 4        |               |               | [CDD](https://www.cryptodatadownload.com/data/binance/) |
| ETTh1                    | Energy        | 1 hour       | 7        | 1             | 14_400        | [Github](https://github.com/zhouhaoyi/ETDataset) | 
| ETTh2                    | Energy        | 1 hour       | 7        | 1             | 14_400        | [Github](https://github.com/zhouhaoyi/ETDataset) | 
| ETDatasetHour            | Energy        | 1 hour       | 7        | 2             | 14_400        | [Github](https://github.com/zhouhaoyi/ETDataset) | 
| ETTm1                    | Energy        | 15 minutes   | 7        | 1             | 57_600        | [Github](https://github.com/zhouhaoyi/ETDataset) |
| ETTm2                    | Energy        | 15 minutes   | 7        | 1             | 57_600        | [Github](https://github.com/zhouhaoyi/ETDataset) |
| ETDatasetMinute          | Energy        | 15 minutes   | 7        | 2             | 57_600        | [Github](https://github.com/zhouhaoyi/ETDataset) |
| Electricity              | Energy        | 15 minutes   | 1        | 321           | 26_304        | [Github](https://github.com/laiguokun/multivariate-time-series-data) |
| ElectricityLoadDiagrams  | Energy        | 15 minutes   | 1        | 370           | 140_256       | [UCI](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) |
| METRLA                   | Traffic       | 5 minutes    | 1        | 207           | 34_272        | [Github](https://github.com/liyaguang/DCRNN) |
| MekongSalinity           | Environment   |              |          |               |               | [Springer](https://link.springer.com/chapter/10.1007/978-981-97-5504-2_43) |
| PeMS03                   | Traffic       | 5 minutes    | 1        | 358           | 26_208        | [Github](https://github.com/guoshnBJTU/ASTGNN) |
| PeMS04                   | Traffic       | 5 minutes    | 1        | 307           | 16_992        | [Github](https://github.com/guoshnBJTU/ASTGNN) |
| PeMS07                   | Traffic       | 5 minutes    | 1        | 883           | 28_224        | [Github](https://github.com/guoshnBJTU/ASTGNN) |
| PeMS08                   | Traffic       | 5 minutes    | 3        | 170           | 17_856        | [Github](https://github.com/guoshnBJTU/ASTGNN) |
| PeMSBAY                  | Traffic       | 5 minutes    | 1        | 325           | 52_116        | [Github](https://github.com/liyaguang/DCRNN) |
| PeMSSF                   | Traffic       | 10 minutes   | 1        | 963           | 63_345        | [UCI](https://archive.ics.uci.edu/dataset/204/pems+sf) |
| SolarCSGREGFC            | Energy        | 15 minutes   | 5        | 8             | 20_352-70_176 | [Github](https://github.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis) |
| SolarEnergy              | Energy        | 1 hour       | 1        | 137           | 52_560        | [Github](https://github.com/laiguokun/multivariate-time-series-data) |
| StatesILI                | Healthcare    | 1 week       | 1        | 37            |               | [Github](https://github.com/emilylaiken/ml-flu-prediction) |
| TetouanPowerConsumption  | Energy        | 10 minutes   | 1        | 3             | 52_416        | [UCI](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city) |
| Traffic                  | Traffic       | 1 hour       | 1        | 862           | 17_544        | [Github](https://github.com/laiguokun/multivariate-time-series-data) |
| Weather5K                | Environment   | 1 hour       | 5        | 5_672         |               | [Github](https://github.com/taohan10200/WEATHER-5K) |
| WindCSGREGFC             | Energy        | 15 minutes   | 10       | 6             | 69_999-70_176 | [Github](https://github.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis) |

**Note**: Number of clients will be decided after splitting the data since clients with insufficient data (cannot form at least 10 samples) will be discarded. `Clients (max)` is the maximum number of clients possible.

---

## Usage
### 1. Single Dataset with Single Configuration

To use a dataset in your experiment (default scenario), specify the dataset name with the `--dataset` argument when running your training or analysis scripts.

**Example:**

```bash
python main.py --dataset=ETTh1
```

You can also set other related arguments such as `--input_len`, `--output_len`, and `--batch_size` to control the window size, forecast horizon, and batch size for your experiment.

**Example:**

```bash
python main.py --dataset=SolarEnergy --input_len=168 --output_len=24 --batch_size=64
```

All clients will use the same configuration as specified above.

Refer to the table above for available dataset names and their details.

---

### 2. Single Dataset with Multiple Configurations
Different clients from the same dataset may have different configurations (e.g., different output lengths or channels).  

**Examples:**
  - **PeMS08OutVar1:** 75% of clients have `output_len=96`, 25% have `output_len=720`.  
    ```bash
    python main.py --dataset=PeMS08OutVar1
    ```
    See: `data_factory/PeMS08.py/PeMS08OutVar1`
  - **PeMS08OutVar2:** 50% of clients have `output_len=96`, 50% have `output_len=720`.  
    ```bash
    python main.py --dataset=PeMS08OutVar2
    ```
    See: `data_factory/PeMS08.py/PeMS08OutVar2`
  - **PeMS08OutVar3:** 25% of clients have `output_len=96`, 75% have `output_len=720`.  
    ```bash
    python main.py --dataset=PeMS08OutVar3
    ```
    See: `data_factory/PeMS08.py/PeMS08OutVar3`
  - **Customized2:** 50% of clients have 1 output channel and 1 input channel, 50% have 7 output channels and 7 input channels.  
    ```bash
    python main.py --dataset=Customized2
    ```
    See: `data_factory/Customized.py/Customized2`

---

### 3. Multi-task / Multi-dataset
Merge multiple datasets, each client belongs to one dataset. Useful for multi-task learning or federated learning across different domains.

**Example:**  
  - **Customized1:** Merges ETDatasetHour (2 clients), TetouanPowerConsumption (3 clients), SolarEnergy (137 clients), Electricity (321 clients) for a total of 463 clients.
    ```bash
    python main.py --dataset=Customized1
    ```
    See: `data_factory/Customized.py/Customized1`

---

### 4. Real / Customized (Heterogeneous Configurations)
Merge multiple datasets, each with potentially different configurations per dataset or client.  

**Example:**  
  - **Customized3:** Merges ETDatasetHour (2 clients, `output_len=96`), TetouanPowerConsumption (3 clients, `output_len=192`).
    ```bash
    python main.py --dataset=Customized3
    ```
    See: `data_factory/Customized.py/Customized3`

---

**Note:**  
- For all scenarios, you can further control client configuration using arguments like `--input_len`, `--output_len`, and `--batch_size`.
- Refer to the dataset table above for available dataset names and their details.