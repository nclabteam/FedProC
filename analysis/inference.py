"""
Zero-shot Inference Evaluation Module

Evaluates pretrained federated learning models on unseen datasets without fine-tuning.
Provides comprehensive inference metrics including both normalized and denormalized evaluations.

This module enables:
- Evaluating model performance on out-of-distribution datasets
- Computing metrics with and without normalization
- Batch processing multiple experiment runs
- Generating detailed inference statistics across trials
- Comparison of model generalization capabilities
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from losses import evaluation_result
from strategies.base import SharedMethods

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Type Definitions and Constants
# =============================================================================

# Output normalization modes
NORM_MODES: Tuple[str, ...] = ("norm", "denorm", "both")
NormMode = str


class InferenceEvaluator:
    """
    Evaluate pretrained models on zero-shot datasets.

    This class provides functionality to evaluate federated learning models on
    datasets they were not trained on, generating comprehensive inference metrics.

    Attributes:
        runs_dir: Path to the runs directory containing experiments.
        output_dir: Path to output directory for inference results.
        device: Computing device (cuda or cpu).
    """

    def __init__(
        self,
        runs_dir: str | Path = "runs",
        output_dir: str | Path = "analysis/inference",
    ) -> None:
        """
        Initialize the InferenceEvaluator.

        Args:
            runs_dir: Path to the runs directory.
            output_dir: Path to the output directory for inference results.
        """
        self.runs_dir = Path(runs_dir)
        self.output_dir = Path(output_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "InferenceEvaluator initialized: runs_dir=%s, device=%s",
            self.runs_dir,
            self.device,
        )

    def _load_config(self, experiment_name: str) -> Dict[str, Any]:
        """
        Load experiment configuration.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Configuration dictionary.
        """
        config_path = self.runs_dir / experiment_name / "config.json"
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load config from %s: %s", config_path, e)
            return {}

    def _load_data_info(self, path_info: str) -> Dict:
        """
        Load data information from path_info JSON file.

        Args:
            path_info: Path to the data info JSON file.

        Returns:
            Data information dictionary.
        """
        try:
            with open(path_info, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load data info from %s: %s", path_info, e)
            return {}

    def _load_model(self, model_path: Path) -> Optional[torch.nn.Module]:
        """
        Load a pretrained model from checkpoint.

        Args:
            model_path: Path to the model checkpoint.

        Returns:
            Loaded model on the configured device, or None if loading fails.
        """
        try:
            model = torch.load(str(model_path), weights_only=False)
            model = model.to(self.device)
            model.eval()
            logger.debug("Loaded model from %s", model_path)
            return model
        except Exception as e:
            logger.error("Failed to load model from %s: %s", model_path, e)
            return None

    def evaluate_model_on_dataset(
        self,
        model: torch.nn.Module,
        data_info: List[Dict],
        config: Dict[str, Any],
        denormalize: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a model on all clients in a dataset.

        Args:
            model: The pretrained model to evaluate.
            data_info: Data information containing client paths and stats.
            config: Experiment configuration with scaler and batch_size.
            denormalize: Whether to denormalize before computing metrics.

        Returns:
            Dictionary with per-client and aggregated metrics.
        """
        losses = []

        for client_idx, client in enumerate(
            tqdm(
                data_info,
                desc=f"Evaluating on dataset ({'denormalized' if denormalize else 'normalized'})",
            )
        ):
            try:
                # Load scaler
                stats = client["stats"]["train"]
                scaler = getattr(__import__("scalers"), config["scaler"])(stats)

                # Load test data
                testloader = SharedMethods.load_data(
                    file=client["paths"]["test"],
                    sample_ratio=1.0,
                    shuffle=False,
                    scaler=scaler,
                    batch_size=config["batch_size"],
                )

                # Evaluate
                client_losses = []
                with torch.no_grad():
                    for x, y in testloader:
                        x = x.to(self.device)
                        y = y.to(self.device)
                        pred = model(x)

                        if denormalize:
                            pred = torch.tensor(
                                scaler.inverse_transform(pred.cpu().detach().numpy()),
                                dtype=pred.dtype,
                                device=self.device,
                            )
                            y = torch.tensor(
                                scaler.inverse_transform(y.cpu().detach().numpy()),
                                dtype=y.dtype,
                                device=self.device,
                            )

                        loss = evaluation_result(pred, y)
                        client_losses.append(loss)

                # Aggregate client metrics
                if client_losses:
                    client_metrics = {
                        k: np.mean([loss[k] for loss in client_losses])
                        for k in client_losses[0].keys()
                    }
                    losses.append(client_metrics)

            except Exception as e:
                logger.warning(
                    "Failed to evaluate client %d: %s",
                    client_idx,
                    e,
                )
                continue

        # Aggregate metrics across all clients
        if not losses:
            logger.warning("No valid client evaluations")
            return {}

        aggregated = {
            k: np.mean([loss[k] for loss in losses]) for k in losses[0].keys()
        }
        std_metrics = {
            f"{k}_std": np.std([loss[k] for loss in losses]) for k in losses[0].keys()
        }

        return {**aggregated, **std_metrics}

    def evaluate_experiment(
        self,
        experiment_name: str,
        target_dataset: str,
        norm_mode: str = "both",
    ) -> List[Dict]:
        """
        Evaluate all model checkpoints from an experiment on a target dataset.

        Args:
            experiment_name: Name of the experiment.
            target_dataset: Name of the target dataset for evaluation.
            norm_mode: Evaluation mode ('norm', 'denorm', or 'both').

        Returns:
            List of result dictionaries containing evaluation metrics.
        """
        results = []

        # Load configuration and data info
        config = self._load_config(experiment_name)
        if not config:
            logger.error("Could not load config for experiment %s", experiment_name)
            return results

        data_info = self._load_data_info(config["path_info"])
        target_data_info = self._load_data_info(
            config["path_info"].replace(config["dataset"], target_dataset)
        )

        if not target_data_info:
            logger.error(
                "Could not load target dataset info for %s",
                target_dataset,
            )
            return results

        # Process each trial
        exp_path = self.runs_dir / experiment_name
        for trial in sorted(os.listdir(exp_path)):
            trial_path = exp_path / trial / "models"
            if not trial_path.exists():
                continue

            # Process each model checkpoint
            for weights_file in sorted(os.listdir(trial_path)):
                if not weights_file.endswith(".pt"):
                    continue

                model_path = trial_path / weights_file
                model = self._load_model(model_path)
                if model is None:
                    continue

                logger.info("Evaluating %s", model_path)

                # Determine model type
                model_type = "best" if "best" in weights_file else "last"

                # Evaluate on target dataset
                modes_to_eval = (
                    ["norm", "denorm"] if norm_mode == "both" else [norm_mode]
                )

                for mode in modes_to_eval:
                    metrics = self.evaluate_model_on_dataset(
                        model,
                        target_data_info,
                        config,
                        denormalize=(mode == "denorm"),
                    )

                    if metrics:
                        result = {
                            "experiment": experiment_name,
                            "trial": trial,
                            "model_type": model_type,
                            "normalization": mode,
                            "target_dataset": target_dataset,
                            "source_dataset": config["dataset"],
                            "input_len": config["input_len"],
                            "output_len": config["output_len"],
                        }
                        result.update(metrics)
                        results.append(result)

        return results

    def save_results(
        self,
        results: List[Dict],
        experiment_name: str,
        target_dataset: str,
    ) -> Path:
        """
        Save inference results to CSV.

        Args:
            results: List of result dictionaries.
            experiment_name: Name of the source experiment.
            target_dataset: Name of the target evaluation dataset.

        Returns:
            Path to the saved CSV file.
        """
        if not results:
            logger.warning("No results to save")
            return None

        df = pl.DataFrame(results)

        # Generate filename
        filename = f"{experiment_name}_inference_{target_dataset}.csv"
        output_path = self.output_dir / filename

        df.write_csv(str(output_path))
        logger.info("Saved inference results to %s", output_path)

        return output_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained federated learning models on zero-shot datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--runs-dir",
        "-r",
        type=str,
        default="runs",
        help="Path to the runs directory",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="analysis/inference",
        help="Output directory for inference results",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        metavar="EXP",
        help="Experiments to evaluate (e.g., exp1 exp2 exp3)",
    )
    parser.add_argument(
        "--target-dataset",
        type=str,
        required=True,
        metavar="DATASET",
        help="Target dataset for zero-shot evaluation (e.g., ETDatasetHour)",
    )
    parser.add_argument(
        "--norm-mode",
        type=str,
        choices=NORM_MODES,
        default="both",
        help="Evaluation mode: 'norm' (keep normalized), 'denorm' (denormalize output), 'both'",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    evaluator = InferenceEvaluator(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
    )

    if not args.experiments:
        logger.error("No experiments specified. Use --experiments exp1 exp2 ...")
        return

    for experiment in args.experiments:
        logger.info("Processing experiment: %s", experiment)

        results = evaluator.evaluate_experiment(
            experiment,
            args.target_dataset,
            args.norm_mode,
        )

        if results:
            evaluator.save_results(results, experiment, args.target_dataset)
        else:
            logger.warning("No results generated for experiment %s", experiment)


if __name__ == "__main__":
    main()
