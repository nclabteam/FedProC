import os
import random

import numpy as np
import torch


class SetSeed:
    def __init__(self, seed: int):
        self.seed = seed

    def _check(self):
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min
        if self.seed is None:
            env_seed = os.environ.get("PL_GLOBAL_SEED")
            if env_seed is None:
                self.seed = random.randint(min_seed_value, max_seed_value)
                print(f"No seed found, seed set to {self.seed}")
            else:
                try:
                    self.seed = int(env_seed)
                except ValueError:
                    self.seed = random.randint(min_seed_value, max_seed_value)
                    print(
                        f"Invalid seed found: {repr(env_seed)}, seed set to {self.seed}"
                    )
        elif not isinstance(self.seed, int):
            self.seed = int(self.seed)

        if not (min_seed_value <= self.seed <= max_seed_value):
            print(
                f"{self.seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}"
            )
            self.seed = random.randint(min_seed_value, max_seed_value)

    def _torch(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # for Multi-GPU, exception safe
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _os(self):
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        os.environ["PL_GLOBAL_SEED"] = str(self.seed)

    def _random(self):
        random.seed(self.seed)

    def _numpy(self):
        np.random.seed(self.seed)

    def set(self):
        self._check()
        self._torch()
        self._os()
        self._random()
        self._numpy()
        print(f"Seed set to {self.seed}")
