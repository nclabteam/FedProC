import argparse
import json
import os

from rich import box
from rich.console import Console
from rich.table import Table
from rich.terminal_theme import MONOKAI

from data_factory import DATASETS
from losses import LOSSES
from models import MODELS
from optimizers import OPTIMIZERS
from scalers import SCALERS
from schedulers import SCHEDULERS
from strategies import STRATEGIES

from .general import increment_path


class CustomHelpFormatter(argparse.HelpFormatter):
    def _get_help_string(self, action):
        help = action.help
        if action.default is not argparse.SUPPRESS:
            help += f" (default: {action.default})"
        return help


class Options:
    def __init__(self, root):
        self.root = root
        self.dup = {}

    def parse_options(self):
        parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
        # general
        parser.add_argument("--seed", type=int, default=941, help="random seed")
        parser.add_argument(
            "--times", type=int, default=1, help="number of times to run the experiment"
        )
        parser.add_argument("--prev", type=int, default=0, help="revious Running times")
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument(
            "--device", type=str, default="cuda", choices=["cpu", "cuda"], help="device"
        )
        parser.add_argument("--device_id", type=str, default="0", help="device id")

        parser.add_argument(
            "--save_local_model",
            action="store_true",
            default=False,
            help="save local model for each client (personalized federated learning)",
        )
        parser.add_argument(
            "--keep_useless_run",
            action="store_true",
            default=False,
            help="delete KeyBoardInterrupt run",
        )

        # save path
        parser.add_argument(
            "--project",
            type=str,
            default=os.path.join(self.root, "runs"),
            help="project name",
        )
        parser.add_argument(
            "--name", type=str, default="exp", help="name of this experiment"
        )
        parser.add_argument("--sep", type=str, default="", help="separator for name")

        # dataset
        parser.add_argument(
            "--input_len",
            type=int,
            default=96,
            help="input length or window size or lookback period or sequence length or history length",
        )
        parser.add_argument(
            "--offset_len",
            type=int,
            default=0,
            help="timestep between input and output",
        )
        parser.add_argument(
            "--output_len",
            type=int,
            default=96,
            help="output length or forecast horizon or prediction length",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="ETDatasetHour",
            help="dataset name",
            choices=DATASETS,
        )
        parser.add_argument("--batch_size", type=int, default=32, help="batch size")
        parser.add_argument(
            "--scaler",
            type=str,
            default="StandardScaler",
            choices=SCALERS,
            help="data normalization method",
        )
        parser.add_argument("--train_ratio", type=float, default=0.8, help="batch size")

        # server
        parser.add_argument(
            "--strategy",
            type=str,
            default="LocalOnly",
            choices=STRATEGIES,
            help="federated learning strategy",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="DLinear",
            choices=MODELS,
            help="model used for training",
        )
        parser.add_argument(
            "--iterations",
            type=int,
            default=10,
            help="number of global rounds in federated learning",
        )
        parser.add_argument(
            "--patience", type=int, default=False, help="Patience for early stopping"
        )
        parser.add_argument(
            "--join_ratio",
            type=float,
            default=1.0,
            help="Ratio of clients per round",
        )
        parser.add_argument(
            "--random_join_ratio",
            type=bool,
            default=False,
            help="Random ratio of clients per round",
        )
        parser.add_argument(
            "--eval_gap", type=int, default=1, help="Rounds gap for evaluation"
        )
        parser.add_argument("--skip_eval_train", action="store_true", default=False)
        parser.add_argument("--last_eval", action="store_true", default=False)

        # client
        parser.add_argument(
            "--optimizer",
            type=str,
            default="Adam",
            help="optimizer name",
            choices=OPTIMIZERS,
        )
        parser.add_argument(
            "--learning_rate", type=float, default=0.0001, help="Local learning rate"
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=1,
            help="Multiple update steps in one iteration",
        )
        parser.add_argument(
            "--loss",
            type=str,
            default="MSE",
            choices=LOSSES,
            help="loss / evaluation function",
        )
        parser.add_argument(
            "--scheduler",
            type=str,
            default="BaseScheduler",
            choices=SCHEDULERS,
            help="learning rate adjustment",
        )

        for dir in ["strategies", "schedulers", "optimizers", "models"]:
            self.apply_args_update(
                parser=parser,
                args_update_functions=getattr(__import__(dir), "args_update_functions"),
            )

        self.args = parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.device_id
        return self

    # Function to apply args_update to a parser
    def apply_args_update(self, parser, args_update_functions):
        existing_args = {action.dest for action in parser._actions}
        for class_name, update_func in args_update_functions.items():
            # Create a temporary parser to capture the arguments
            temp_parser = argparse.ArgumentParser(add_help=False)
            update_func(temp_parser)

            # Add only the new arguments to the main parser
            for action in temp_parser._actions:
                if action.dest not in existing_args:
                    parser._add_action(action)
                    existing_args.add(action.dest)
                else:
                    self.dup[action.dest] = 0

    def _fix_specific_param(self, category, attr_name):
        """
        Dynamically fix specific parameters for a given category (e.g., strategies, schedulers, optimizers, models).

        Args:
            category (str): The category to process (e.g., "strategies", "schedulers", "optimizers", "models").
            attr_name (str): The attribute name in self.args corresponding to the category (e.g., "strategy").
        """
        # Import the module dynamically
        module = __import__(category)

        # Access and update optional parameters
        optional = getattr(module, "optional")
        param_key = getattr(
            self.args, attr_name
        )  # e.g., self.args.strategy for "strategies"
        self.update_if_none(params=optional.get(param_key, {}))

        # Access and update compulsory parameters
        compulsory = getattr(module, "compulsory")
        self.update_args(params=compulsory.get(param_key, {}))

    def update_args(self, params: dict):
        for key, value in params.items():
            self.update_arg(key, value)

    def update_if_none(self, params: dict):
        for key, value in params.items():
            if key not in self.args or getattr(self.args, key) is None:
                self.update_arg(key, value)
                if key in self.dup.keys():
                    self.dup[key] += 1
            else:
                if key in self.dup.keys():
                    if self.dup[key] == 1:
                        raise ValueError(f"Duplicate argument {key} found")

    def update_arg(self, name, value):
        self.args.__dict__[name] = value

    def _fix_save_path(self):
        path = increment_path(
            os.path.join(self.args.project, self.args.name),
            exist_ok=False,
            sep=self.args.sep,
        )
        self.update_arg("save_path", path)
        if not os.path.exists(path):
            os.makedirs(path)

    def _fix_device(self):
        import torch

        if self.args.device == "cuda" and not torch.cuda.is_available():
            print("cuda is not available. Using cpu instead.")
            self.update_arg("device", "cpu")

    @staticmethod
    def _clean_none_args(args):
        return {k: v for k, v in args.items() if v is not None}

    def fix_args(self):
        self._fix_save_path()
        self._fix_device()
        self._fix_specific_param("strategies", "strategy")
        self._fix_specific_param("schedulers", "scheduler")
        self._fix_specific_param("optimizers", "optimizer")
        self._fix_specific_param("models", "model")
        self.args.__dict__["max_epochs"] = self.args.iterations * self.args.epochs
        self.args.__dict__ = self._clean_none_args(args=self.args.__dict__)
        return self

    def save(self):
        with open(os.path.join(self.args.save_path, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)

    def display(self):
        table = Table(title="Experiment Arguments", box=box.ROUNDED)
        table.add_column("Argument", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for arg in vars(self.args):
            table.add_row(arg, str(getattr(self.args, arg)))

        console = Console(record=True)
        console.print(table)
