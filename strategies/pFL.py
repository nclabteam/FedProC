import numpy as np

from .tFL import tFL, tFL_Client


class pFL(tFL):
    """Personalized FL server.

    Extends tFL by evaluating each client's personalized model (global model
    overlaid with stored personal params) before each round's local training.
    Strategies that maintain per-client personal parameters should inherit from
    pFL and set ``personal_params_name`` on their client class.
    """

    def _pre_eval_hook(self, dataset_type: str) -> None:
        incumbent = [i for i in range(self.num_clients) if not self.is_new[i]]
        losses = self.trainer.evaluate_personalized(
            incumbent,
            self.public_model_params,
            self.clients_personal_model_params,
            dataset_type,
            self.current_iter,
        )
        metric = f"personal_avg_{dataset_type}_loss"
        self.metrics[metric].append(float(np.mean(losses)))
        self.logger.info(
            f"Personalization {dataset_type.capitalize()} Loss: "
            f"{self.metrics[metric][-1]:.4f}"
        )

    def early_stopping(self) -> bool:
        metric = self.metrics["personal_avg_test_loss"]
        if not self.patience or len(metric) < self.patience:
            return False
        if min(metric) not in metric[-self.patience:]:
            self.logger.info("Early stopping activated.")
            return True
        return False


class pFL_Client(tFL_Client):
    """Passthrough — same as tFL_Client; named subclass kept as the
    discovery anchor for ``<Strategy>_Client`` resolution and as the shared
    base for personalized-FL client classes."""
