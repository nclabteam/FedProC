# Datasets

## Available Datasets

| Name                     | Domain        | Granularity  | Variates | Clients (max) | Samples      | 
|--------------------------|---------------|--------------|----------|---------------|--------------|
| BaseStation5G            | Communication | 2 minutes    | 11       | 3             | 4_192-15_927 |
| BeijingAirQuality        | Environment   | 1 hour       | 11       | 12            |              |
| COVID19Cases             | Healthcare    | 1 day        | 10       | 55            |              |
| CitiesILI                | Healthcare    | 1 week       | 1        | 122           |              |
| CryptoDataDownloadDay    | Economic      | 1 day        | 4        |               |              |
| CryptoDataDownloadHour   | Economic      | 1 hour       | 4        |               |              |
| CryptoDataDownloadMinute | Economic      | 1 minute     | 4        |               |              |
| ETTh1                    | Energy        | 1 hour       | 7        | 1             | 14_400       | 
| ETTh2                    | Energy        | 1 hour       | 7        | 1             | 14_400       | 
| ETDatasetHour            | Energy        | 1 hour       | 7        | 2             | 14_400       | 
| ETTm1                    | Energy        | 15 minutes   | 7        | 1             | 57_600       |
| ETTm2                    | Energy        | 15 minutes   | 7        | 1             | 57_600       |
| ETDatasetMinute          | Energy        | 15 minutes   | 7        | 2             | 57_600       |
| Electricity              | Energy        | 15 minutes   | 1        | 321           | 26_304       |
| ElectricityLoadDiagrams  | Energy        | 15 minutes   | 1        | 370           | 140_256      |
| METRLA                   | Traffic       | 5 minutes    | 1        | 207           | 34_272       |
| MekongSalinity           | Environment   |              |          |               |              |
| PeMS03                   | Traffic       | 5 minutes    | 1        | 358           | 26_208       |
| PeMS04                   | Traffic       | 5 minutes    | 1        | 307           | 16_992       |
| PeMS07                   | Traffic       | 5 minutes    | 1        | 883           | 28_224       |
| PeMS08                   | Traffic       | 5 minutes    | 3        | 170           | 17_856       |
| PeMSBAY                  | Traffic       | 5 minutes    | 1        | 325           | 52_116       |
| PeMSSF                   | Traffic       | 10 minutes   | 1        | 963           | 63_345       |
| SolarEnergy              | Energy        | 1 hour       | 1        | 137           | 52_560       |
| StatesILI                | Healthcare    | 1 week       | 1        | 37            |              |
| TetouanPowerConsumption  | Energy        | 10 minutes   | 1        | 3             | 52_416       |
| Traffic                  | Traffic       | 1 hour       | 1        | 862           | 17_544       |
| Weather5K                | Environment   | 1 hour       | 5        | 5_672         |              |

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