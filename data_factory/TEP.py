import datetime
import os

import pyreadr

from .base import BaseDataset


class TEP(BaseDataset):
    """Tennessee Eastman Process (Rieth et al., 2017): a single simulation
    run as one federated client.

    The public dataset (Harvard Dataverse, doi:10.7910/DVN/6C3JR1) ships four
    .RData pools (FaultFree/Faulty x Training/Testing), each holding 500
    independent simulation runs that do not share an RNG seed across pools or
    with each other -- concatenating runs (or pools) end-to-end would splice
    together unrelated trajectories, so only one run (FaultFree_Training,
    simulationRun == 1, entirely fault-free) is kept as a single continuous
    52-variable series, following ETDataset's "1 file = 1 client" pattern.

    Samples have no real timestamp, just a 3-minute simulation step index, so
    a synthetic calendar axis is generated like M4.
    """

    process_columns = [f"xmeas_{i}" for i in range(1, 42)] + [
        f"xmv_{i}" for i in range(1, 12)
    ]
    simulation_run = 1
    epoch = datetime.datetime(2000, 1, 1)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join("datasets", "TEP")
        self.path_raw = os.path.join("datasets", "TEP", "raw")
        self.path_temp = os.path.join("datasets", "TEP", "temp")

        self.column_date = "date"
        self.column_target = list(self.process_columns)
        self.column_train = list(self.process_columns)
        self.granularity = 3
        self.granularity_unit = "minute"
        # Synthetic dates are gap-free by construction (same reasoning as M4).
        self.skip_fill_date = True

        self.url = "https://dataverse.harvard.edu/api/access/datafile/3031241"

    def download(self) -> None:
        """Download one fault-free run and add synthetic timestamps."""
        os.makedirs(self.path_raw, exist_ok=True)
        os.makedirs(self.path_temp, exist_ok=True)

        rdata_path = os.path.join(self.path_temp, "TEP_FaultFree_Training.RData")
        self.download_file(url=self.url, save_path=rdata_path)

        result = pyreadr.read_r(rdata_path)
        df = next(iter(result.values()))
        df = df[df["simulationRun"] == self.simulation_run].sort_values("sample")

        dates = [
            self.epoch + datetime.timedelta(minutes=self.granularity * i)
            for i in range(len(df))
        ]
        out = df[self.process_columns].copy()
        out.insert(0, self.column_date, dates)
        out.to_csv(os.path.join(self.path_raw, "run_1.csv"), index=False)
