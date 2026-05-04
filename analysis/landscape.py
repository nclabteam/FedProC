"""
Loss landscape visualization.

Computes and plots the loss surface around trained model weights.
Supports both traditional FL (server model) and personalized FL (client models).

Input:  runs/expN/ (experiment directory with model checkpoints)
Output: runs/expN/landscape/ (PNG files: 1d, 2d, 3d per model)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.io import load_config, read_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LossLandscape:
    """Compute and visualize loss landscapes around trained model weights."""

    def __init__(
        self,
        experiment_dir: str | Path,
        n_points: int = 51,
        device: Optional[str] = None,
        allow_unsafe_legacy: bool = False,
    ) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.n_points = n_points
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.allow_unsafe_legacy = allow_unsafe_legacy

        # Load config
        self.config = load_config(self.experiment_dir)
        self.save_local_model = self.config.get("save_local_model", False)

    # =========================================================================
    # Model loading
    # =========================================================================

    def _load_model(self, model_path: Path) -> Optional[nn.Module]:
        """Load a model checkpoint."""
        from strategies.base import SharedMethods

        try:
            model = SharedMethods.load_checkpoint_model(
                checkpoint_path=str(model_path),
                device=self.device,
                allow_unsafe_legacy=self.allow_unsafe_legacy,
                verbose=logger,
            )
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error("Failed to load model from %s: %s", model_path, e)
            return None

    # =========================================================================
    # Data loading
    # =========================================================================

    def _load_tensors(
        self, client_info: Dict, split: str = "test"
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Load full dataset as (x, y) tensors. split: 'train' or 'test'."""
        from strategies.base import SharedMethods

        try:
            stats = client_info["stats"]["train"]
            scaler = getattr(__import__("scalers"), self.config["scaler"])(stats)
            loader = SharedMethods.load_data(
                file=client_info["paths"][split],
                sample_ratio=1.0,
                shuffle=False,
                scaler=scaler,
                batch_size=999999,
            )
            for x, y in loader:
                return x.to(self.device), y.to(self.device)
            return None
        except Exception as e:
            logger.error("Failed to load %s data: %s", split, e)
            return None

    def _load_data_info(self) -> List[Dict]:
        """Load data info from path_info."""
        path_info = self.config.get("path_info", "")
        if not path_info:
            return []
        try:
            with open(path_info, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _compute_loss(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> float:
        """Compute loss on full dataset tensors (single forward pass)."""
        import losses as loss_module

        loss_name = self.config.get("loss", "MSE")
        loss_fn = getattr(loss_module, loss_name)()

        model.eval()
        with torch.no_grad():
            return loss_fn(model(x), y).item()

    # =========================================================================
    # Run selection
    # =========================================================================

    def _get_best_run(self) -> Optional[int]:
        """Get the run index with lowest min loss from results.csv."""
        results_path = self.experiment_dir / "results.csv"
        data = read_csv(results_path)
        if data is None:
            return None

        metrics = data.get("metric", [])
        avg_mins = data.get("avg_min", [])

        loss_metric = (
            "personal_avg_test_loss"
            if self.save_local_model
            else "global_avg_test_loss"
        )
        for i, m in enumerate(metrics):
            if m == loss_metric:
                # We need per-run min, not avg. Fall back to first run.
                break

        # Find run with lowest loss by checking each run's server.csv
        best_run = None
        best_loss = float("inf")
        for entry in sorted(self.experiment_dir.iterdir()):
            if not entry.is_dir() or not entry.name.isdigit():
                continue
            server_csv = entry / "results" / "server.csv"
            server_data = read_csv(server_csv)
            if server_data is None:
                continue
            loss_col = server_data.get(loss_metric, [])
            if loss_col:
                vals = [float(v) for v in loss_col if float(v) != 9_999_999.0]
                if vals:
                    min_loss = min(vals)
                    if min_loss < best_loss:
                        best_loss = min_loss
                        best_run = int(entry.name)
        return best_run

    def _get_run_dirs(
        self, run: Optional[int] = None, all_runs: bool = False
    ) -> List[Path]:
        """Get run directories to process."""
        if all_runs:
            return sorted(
                e
                for e in self.experiment_dir.iterdir()
                if e.is_dir() and e.name.isdigit()
            )

        if run is not None:
            run_dir = self.experiment_dir / str(run)
            return [run_dir] if run_dir.exists() else []

        # Default: best run
        best = self._get_best_run()
        if best is not None:
            return [self.experiment_dir / str(best)]

        # Fallback: first run
        for entry in sorted(self.experiment_dir.iterdir()):
            if entry.is_dir() and entry.name.isdigit():
                return [entry]
        return []

    # =========================================================================
    # Weight perturbation
    # =========================================================================

    def _get_flat_params(self, model: nn.Module) -> torch.Tensor:
        """Flatten all model parameters into a single vector."""
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    def _set_flat_params(self, model: nn.Module, flat_params: torch.Tensor) -> None:
        """Set model parameters from a flat vector."""
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset : offset + numel].view(p.shape))
            offset += numel

    def _random_direction(self, model: nn.Module) -> torch.Tensor:
        """Generate a random direction, filter-normalized per layer."""
        direction = []
        for p in model.parameters():
            d = torch.randn_like(p.data)
            # Filter-wise normalization
            norm_d = d / (d.norm() + 1e-10) * p.data.norm()
            direction.append(norm_d.view(-1))
        return torch.cat(direction)

    # =========================================================================
    # Landscape computation
    # =========================================================================

    def compute_1d(
        self,
        model: nn.Module,
        test_x: torch.Tensor,
        test_y: torch.Tensor,
        train_x: Optional[torch.Tensor] = None,
        train_y: Optional[torch.Tensor] = None,
        alpha_range: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compute 1D loss landscape along one random direction.

        Returns (alphas, test_losses, train_losses_or_None).
        """
        original = self._get_flat_params(model)
        d = self._random_direction(model)
        alphas = np.linspace(-alpha_range, alpha_range, self.n_points)
        test_losses = []
        train_losses = [] if train_x is not None else None

        for alpha in alphas:
            self._set_flat_params(model, original + alpha * d)
            test_losses.append(self._compute_loss(model, test_x, test_y))
            if train_x is not None:
                train_losses.append(self._compute_loss(model, train_x, train_y))

        self._set_flat_params(model, original)
        return (
            alphas,
            np.array(test_losses),
            np.array(train_losses) if train_losses else None,
        )

    def compute_2d(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha_range: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 2D loss landscape along two random directions."""
        original = self._get_flat_params(model)
        d1 = self._random_direction(model)
        d2 = self._random_direction(model)
        alphas = np.linspace(-alpha_range, alpha_range, self.n_points)
        betas = np.linspace(-alpha_range, alpha_range, self.n_points)
        losses = np.zeros((self.n_points, self.n_points))

        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                self._set_flat_params(model, original + alpha * d1 + beta * d2)
                losses[i, j] = self._compute_loss(model, x, y)

        self._set_flat_params(model, original)
        return alphas, betas, losses

    # =========================================================================
    # Plotting
    # =========================================================================

    def plot_1d(
        self,
        alphas: np.ndarray,
        test_losses: np.ndarray,
        train_losses: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot 1D loss landscape with train/test lines."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(alphas, test_losses, linewidth=2, label="Test")
        if train_losses is not None:
            ax.plot(alphas, train_losses, linewidth=2, label="Train", linestyle="--")
        # Mark origin (alpha=0)
        mid = len(alphas) // 2
        ax.plot(alphas[mid], test_losses[mid], "k+", markersize=12, markeredgewidth=2)
        ax.set_xlabel("Weight perturbation", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Saved: %s", save_path)
        plt.close(fig)

    def plot_2d(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        losses: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot 2D loss landscape as contour lines with labels."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")
        ax.set_facecolor("white")
        cs = ax.contour(alphas, betas, losses, levels=20, cmap="viridis")
        ax.clabel(cs, fmt="%.3f", fontsize=8)
        ax.set_xlabel("Alpha", fontsize=13)
        ax.set_ylabel("Beta", fontsize=13)
        ax.tick_params(labelsize=10)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Saved: %s", save_path)
        plt.close(fig)

    def plot_3d(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        losses: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot 3D loss landscape as surface — paper style."""
        import matplotlib.pyplot as plt
        from matplotlib import cm

        A, B = np.meshgrid(alphas, betas)
        fig = plt.figure(figsize=(10, 8), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            A,
            B,
            losses,
            cmap=cm.RdYlBu_r,
            edgecolor="#cccccc",
            linewidth=0.3,
            alpha=0.9,
        )
        # Low oblique camera
        ax.view_init(elev=25, azim=-135)
        # Remove all axes, ticks, frame
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            logger.info("Saved: %s", save_path)
        plt.close(fig)

    # =========================================================================
    # Main pipeline
    # =========================================================================

    def run(
        self,
        mode: str = "3d",
        checkpoint: str = "best",
        run: Optional[int] = None,
        all_runs: bool = False,
        client: Optional[int] = None,
        all_clients: bool = False,
        alpha_range: float = 1.0,
    ) -> None:
        """Generate loss landscape plots.

        Args:
            mode: "1d", "2d", "3d", or "all"
        """
        run_dirs = self._get_run_dirs(run, all_runs)
        if not run_dirs:
            logger.error(
                "No run directories found (experiment may be compacted — landscape needs model checkpoints)"
            )
            return

        data_info = self._load_data_info()
        if not data_info:
            logger.error("No data info found")
            return

        # Use first client's data for landscape computation
        # (loss landscape is about weight perturbation, not data variation)
        test_data = self._load_tensors(data_info[0], split="test")
        train_data = self._load_tensors(data_info[0], split="train")
        if test_data is None:
            logger.error("Failed to load data")
            return
        test_x, test_y = test_data
        train_x, train_y = train_data if train_data else (None, None)

        landscape_dir = self.experiment_dir / "landscape"
        landscape_dir.mkdir(parents=True, exist_ok=True)

        need_1d = mode in ("1d", "all")
        need_2d = mode in ("2d", "all")
        need_3d = mode in ("3d", "all")

        for run_dir in run_dirs:
            run_idx = run_dir.name
            models_dir = run_dir / "models"

            if not models_dir.exists():
                logger.warning("No models dir for run %s", run_idx)
                continue

            # Determine which models to process
            model_files = self._get_model_files(
                models_dir,
                checkpoint=checkpoint,
                client=client,
                all_clients=all_clients,
            )

            for model_name, model_path in model_files:
                model = self._load_model(model_path)
                if model is None:
                    continue

                logger.info("Computing landscape for run %s / %s", run_idx, model_name)
                suffix = f"run_{run_idx}_{model_name}"

                if need_1d:
                    alphas, test_losses_1d, train_losses_1d = self.compute_1d(
                        model,
                        test_x,
                        test_y,
                        train_x,
                        train_y,
                        alpha_range,
                    )
                    self.plot_1d(
                        alphas,
                        test_losses_1d,
                        train_losses_1d,
                        landscape_dir / f"{suffix}_1d.png",
                    )

                if need_2d or need_3d:
                    alphas_2d, betas_2d, losses_2d = self.compute_2d(
                        model, test_x, test_y, alpha_range
                    )
                    if need_2d:
                        self.plot_2d(
                            alphas_2d,
                            betas_2d,
                            losses_2d,
                            landscape_dir / f"{suffix}_2d.png",
                        )
                    if need_3d:
                        self.plot_3d(
                            alphas_2d,
                            betas_2d,
                            losses_2d,
                            landscape_dir / f"{suffix}_3d.png",
                        )

                # Free GPU memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _get_model_files(
        self,
        models_dir: Path,
        checkpoint: str = "best",
        client: Optional[int] = None,
        all_clients: bool = False,
    ) -> List[Tuple[str, Path]]:
        """Get model files to process based on FL mode."""
        files = []
        suffix = f"_{checkpoint}.pt"  # _best.pt or _last.pt

        if not self.save_local_model:
            # Traditional FL: server model only
            path = models_dir / f"server{suffix}"
            if path.exists():
                files.append(("server", path))
        else:
            # Personalized FL: client models
            if all_clients:
                for f in sorted(models_dir.glob(f"client_*{suffix}")):
                    name = f.stem.replace(f"_{checkpoint}", "")
                    files.append((name, f))
            elif client is not None:
                path = models_dir / f"client_{client:03d}{suffix}"
                if path.exists():
                    files.append((f"client_{client:03d}", path))
            else:
                # Default: first client
                for f in sorted(models_dir.glob(f"client_*{suffix}")):
                    name = f.stem.replace(f"_{checkpoint}", "")
                    files.append((name, f))
                    break

        return files


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Loss landscape visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment", "-e", type=str, required=True, help="Experiment directory"
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="3d",
        choices=["1d", "2d", "3d", "all"],
        help="Plot mode",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Model checkpoint to use",
    )
    parser.add_argument("--run", type=int, default=None, help="Specific run index")
    parser.add_argument("--all", action="store_true", dest="all_runs", help="All runs")
    parser.add_argument(
        "--client",
        type=int,
        default=None,
        help="Specific client index (personalized FL)",
    )
    parser.add_argument(
        "--all-clients", action="store_true", help="All clients (personalized FL)"
    )
    parser.add_argument("--n-points", type=int, default=51, help="Grid resolution")
    parser.add_argument(
        "--alpha-range", type=float, default=1.0, help="Perturbation range"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--allow-unsafe-legacy", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    landscape = LossLandscape(
        experiment_dir=args.experiment,
        n_points=args.n_points,
        device=args.device,
        allow_unsafe_legacy=args.allow_unsafe_legacy,
    )
    landscape.run(
        mode=args.mode,
        checkpoint=args.checkpoint,
        run=args.run,
        all_runs=args.all_runs,
        client=args.client,
        all_clients=args.all_clients,
        alpha_range=args.alpha_range,
    )


if __name__ == "__main__":
    main()
