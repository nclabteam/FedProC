## Customization
### Dataset
The customized dataset should be inherited from one of those classes: `BaseDataset`, `CustomDataset`, `CustomOnSingleDataset`.

```
from .base import BaseDataset

class NewDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_raw = os.path.join("datasets", "NewDataset", "raw")
        self.save_path = os.path.join("datasets", "NewDataset")
        self.column_date = "DATE"
        self.column_train = ["feature1", "feature2", "feature3"]
        self.column_target = ["feature1", "feature2", "feature3"]
        self.granularity = 1
        self.granularity_unit = "hour"
```

The `self.path_raw` should be the directory which has raw .csv files. Each file is the raw data for each client:
```
datasets/NewDataset/raw/1.csv
datasets/NewDataset/raw/2.csv
datasets/NewDataset/raw/3.csv
```

If not in this form, you may modify `download` method to download and/or preprocess data into the above format:
```
from .base import BaseDataset

class NewDataset(BaseDataset):
    def download(self):
        # do magic here
```

If you want 2 datasets and want to change configurations one of them:
```
from .base import CustomDataset

class Custom1(CustomDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "Custom1")
        self.sets = [
            {
                "dataset": NewDataset,
                "column_target": ["feature1", "feature3"],
                "column_train": ["feature1", "feature3"],
            },
            {
                "dataset": NewDataset1,
            },
        ]
```

If you want 1 dataset and split them into multiple parts with each part has different configurations. This example shows that `Custom2` dataset has 3/4 `NewDataset` with the `output_len=96` and 1/4 with the `output_len=720`:
```
class Custom2(CustomOnSingleDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "Custom2")
        self.sets = [
            {
                "dataset": NewDataset,
                "output_len": 96,
            },
            {
                "dataset": NewDataset,
                "output_len": 96,
            },
            {
                "dataset": NewDataset,
                "output_len": 96,
            },
            {
                "dataset": NewDataset,
                "output_len": 720,
            },
        ]
```

### Strategy
The customized server class should be inherited from `Server` and client class should be inherited from `Client`. For example, if the `strategy` inherit from `FedAvg`:

```
from .base import Client, Server

class FedNew(Server):
    pass

class FedNew_Client(Client):
    pass
```

If the customized `strategy` is inherited from another `strategy` like `DFL`:

```
from .DFL import DFL, DFL_Client

class FedNew(DFL):
    pass

class FedNew_Client(DFL_Client):
    pass
```

If the customized `strategy` does not change the client side, you may ignore it:

```
from .base import Server

class FedNew(Server):
    pass
```

The customized `strategy` must have the server class name the same as the file name.

```
# file_path: strategies/FedNew.py

class FedNew(Server):
    pass
```

### Specific Hyper-parameters
In each method (`loss`, `optimizer`, `strategy`, etc.), there will be 2 dictionaries `optional` and `compulsory`. 

`optional` is the dictionary storing the configurations that only used for the customized method. For example: 
```
optional = {
    "key1": value1, 
    "key2": value2,
}
```
Those keys must be specified with the same file by making a global function `args_update`. For example:
```
def args_update(parser):
    parser.add_argument(
        "--key1",
        type=int,
        default=None,
        help="what is this parameter",
    )
    parser.add_argument(
        "--key2",
        type=float,
        default=None,
        help="what is this parameter",
    )
```

`compulsory` is the dictionary storing the configurations that must be used in the method. For example, if the customized strategy is a personalized federated learning, the dictionary must have:
```
compulsory = {
  "save_local_model": True
}
```

If this is a customized `strategy`, you can directly use `self.key1` and `self.key2` directly in `Server` class of `Client` class without further specification:
```
class FedNew(Server):
  def server_method(self):
    return self.key1

class FedNew_Client(Client):
  def client_method(self):
    return self.key2
```

If this is a customized method which is inherited from another mehtod having `compulsory` or `optional`:
```
from .FedNew import args_update as FedNew_args_update
from .FedNew import optional as FedNew_optional

optional = {
    **FedNew_optional,
    "key3": value3,
}

def args_update(parser):
    FedNew_args_update(parser)
    parser.add_argument(
        "--key3",
        type=str,
        default=None,
        choices=["a", "b", "c"]
        help="what is this parameter",
    )
```