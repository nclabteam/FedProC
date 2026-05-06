import gc

from .tFL import tFL, tFL_Client


class pFL(tFL):
    """
    Personalized Federated Learning.

    Extends tFL by saving client models and tracking personalization metrics.
    All save_local_model logic lives here — tFL has none.
    """

    def _pre_eval_hook(self, dataset_type: str) -> None:
        self.evaluate_personalization_loss(dataset_type)

    def save_models(self, save_type: str) -> None:
        if save_type not in ["last", "best"]:
            raise ValueError("save_type must be 'last' or 'best'")

        should_save = True
        if save_type == "best":
            metric_key = "personal_avg_test_loss"
            if metric_key not in self.metrics or not self.metrics[metric_key]:
                should_save = False
            else:
                metric_values = self.metrics[metric_key]
                if metric_values[-1] != min(metric_values):
                    should_save = False

        if not should_save:
            return

        # Save server/global model
        if not self.exclude_server_model_processes:
            self.save_model(
                model=self.model,
                path=self.model_path,
                name=self.name,
                postfix=save_type,
                configs=self.configs,
                metadata={"save_type": save_type, "owner": "server"},
                verbose=self.logger,
            )

        # Save client/local models
        for client in self.clients:
            client.save_model(
                model=client.model,
                path=client.model_path,
                name=client.name,
                postfix=save_type,
                configs=client.configs,
                metadata={"save_type": save_type, "owner": client.name},
                verbose=client.logger,
            )

    def early_stopping(self) -> bool:
        metric = self.metrics["personal_avg_test_loss"]
        if not self.patience or len(metric) < self.patience:
            return False
        best_so_far = min(metric)
        if best_so_far not in metric[-self.patience :]:
            self.logger.info("Early stopping activated.")
            return True
        return False

    def get_model_info(self) -> None:
        import torch.nn as nn

        if not self.exclude_server_model_processes and isinstance(
            self.model, nn.Module
        ):
            dl = self.clients[0].load_train_data()
            self.summarize_model(dataloader=dl)
            del dl
            gc.collect()
        for client in self.clients:
            if isinstance(client.model, nn.Module):
                dl = client.load_train_data()
                client.summarize_model(dataloader=dl)
                del dl
                gc.collect()


class pFL_Client(tFL_Client):
    """Passthrough — same as tFL_Client."""
