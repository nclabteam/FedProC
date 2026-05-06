import copy

import numpy as np

from schedulers.BaseScheduler import BaseScheduler

from .dFL import dFL, dFL_Client


class DFedHPO(dFL):
    """Run decentralized HPO before the normal DFL training loop."""

    optional = {
        "trials": 10,
        "eval_epochs": 3,
        "aggregator": "FA",
        "top_k": 3,
        "lr_min": 1e-5,
        "lr_max": 1e-1,
    }

    compulsory = {
        "scheduler": "BaseScheduler",
    }

    @classmethod
    def args_update(cls, parser):
        super().args_update(parser)
        parser.add_argument("--trials", type=int, default=None)
        parser.add_argument("--eval_epochs", type=int, default=None)
        parser.add_argument(
            "--aggregator", type=str, default=None, choices=["CA", "FA", "MA"]
        )
        parser.add_argument("--top_k", type=int, default=None)
        parser.add_argument("--lr_min", type=float, default=None)
        parser.add_argument("--lr_max", type=float, default=None)

    def train(self):
        self.run_hpo()
        super().train()

    def run_hpo(self):
        self.logger.info("--- DFed-HPO: starting HPO phase ---")
        for client in self.clients:
            client.run_local_hpo()

        snapshots = {client.id: list(client.hpo_candidates) for client in self.clients}
        for client in self.clients:
            for neighbor_id in client.neighbors:
                client.hpo_candidates.extend(snapshots[neighbor_id])

        for client in self.clients:
            client.aggregate_hpo()

        for client in self.clients:
            client.apply_optimal_config()
            self.logger.info(
                f"Client {client.id}: lr={client.optimal_config['lr']:.6f}"
            )
        self.logger.info("--- DFed-HPO: HPO phase complete ---")


class DFedHPO_Client(dFL_Client):
    def run_local_hpo(self):
        self.hpo_candidates = []
        initial_state = copy.deepcopy(self.model.state_dict())

        for _ in range(self.trials):
            config = self._sample_config()
            loss = self._evaluate_config(config, initial_state)
            self.hpo_candidates.append((config, loss))

        self.model.load_state_dict(initial_state)
        self.logger.info(f"HPO: evaluated {self.trials} configs")

    def _sample_config(self):
        log_lr = np.random.uniform(np.log(self.lr_min), np.log(self.lr_max))
        return {"lr": float(np.exp(log_lr))}

    def _evaluate_config(self, config, initial_state):
        self.model.load_state_dict(initial_state)

        original_lr = self.configs.learning_rate
        self.configs.learning_rate = config["lr"]
        obj = self._get_objective_function("optimizers", self.configs.optimizer)
        temp_optimizer = obj(params=self.model.parameters(), configs=self.configs)
        self.configs.learning_rate = original_lr

        temp_scheduler = BaseScheduler(optimizer=temp_optimizer, configs=self.configs)

        train_loader = self.load_train_data()
        for _ in range(self.eval_epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=train_loader,
                optimizer=temp_optimizer,
                criterion=self.loss,
                scheduler=temp_scheduler,
                device=self.device,
                offload_after=False,
            )

        losses = self.calculate_loss(
            model=self.model,
            dataloader=self.load_test_data(),
            criterion=self.loss,
            device=self.device,
        )
        return float(np.mean(losses))

    def aggregate_hpo(self):
        if self.aggregator == "CA":
            self.optimal_config = self._consensus_aggregator()
        elif self.aggregator == "FA":
            self.optimal_config = self._fusion_aggregator()
        elif self.aggregator == "MA":
            self.optimal_config = self._metaregress_aggregator()
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

    def _consensus_aggregator(self):
        return sorted(self.hpo_candidates, key=lambda item: item[1])[0][0]

    def _fusion_aggregator(self):
        sorted_candidates = sorted(self.hpo_candidates, key=lambda item: item[1])
        top_k = sorted_candidates[: self.top_k]
        return {"lr": float(np.mean([candidate[0]["lr"] for candidate in top_k]))}

    def _metaregress_aggregator(self):
        if len(self.hpo_candidates) < 3:
            return self._fusion_aggregator()

        x = np.array([np.log(candidate[0]["lr"]) for candidate in self.hpo_candidates])
        y = np.array([candidate[1] for candidate in self.hpo_candidates])
        degree = min(2, len(x) - 1)
        poly = np.poly1d(np.polyfit(x, y, deg=degree))

        lr_grid = np.linspace(np.log(self.lr_min), np.log(self.lr_max), 200)
        top_k_idx = np.argsort(poly(lr_grid))[: self.top_k]
        top_k_lrs = np.exp(lr_grid[top_k_idx])

        initial_state = copy.deepcopy(self.model.state_dict())
        best_loss = float("inf")
        best_config = {"lr": float(top_k_lrs[0])}
        for lr_value in top_k_lrs:
            config = {"lr": float(lr_value)}
            loss = self._evaluate_config(config, initial_state)
            if loss < best_loss:
                best_loss = loss
                best_config = config

        self.model.load_state_dict(initial_state)
        return best_config

    def apply_optimal_config(self):
        lr = self.optimal_config["lr"]
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.logger.info(f"HPO: applied lr={lr:.6f}")
