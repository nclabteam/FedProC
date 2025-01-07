import copy
import json
import logging
import math
import os
import random
import sys
import time
from argparse import Namespace
from collections import defaultdict

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class SharedMethods:
    def __init__(self):
        self.default_value = -1.0

    def save_results(self):
        pl_df = pl.DataFrame(self.metrics)
        path = os.path.join(self.result_path, self.name.lower().strip() + ".csv")
        pl_df.write_csv(path)
        self.logger.info(f"Results saved to {path}")

    def save_model(self):
        path = os.path.join(self.model_path, self.name.lower().strip() + ".pt")
        torch.save(self.model, path)
        self.logger.info(f"Model saved to {path}")

    def fix_results(self, default=-1.0):
        max_length = max(len(lst) for lst in self.metrics.values())
        for key in self.metrics.keys():
            if len(self.metrics[key]) < max_length:
                self.metrics[key].extend(
                    [default] * (max_length - len(self.metrics[key]))
                )

    def make_logger(self, name, path):
        """
        Creates a logger with a unique name and path.
        """
        log_path = os.path.join(path, f"{name.lower().strip()}.log")

        # Create a unique logger name using the instance id
        logger_name = f"{name}_{self.times}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create file and stream handlers
        file_handler = logging.FileHandler(log_path)
        stream_handler = logging.StreamHandler()

        # Set logging format
        formatter = logging.Formatter(f"%(asctime)s ~ {name} ~ %(message)s")
        # formatter = logging.Formatter(
        #     f"%(asctime)s ~ %(levelname)s ~ %(lineno)-4.4d ~ {name} ~ %(message)s"
        # )
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.info(f"Logger created at {log_path}")

    def set_configs(self, configs, **kwargs):
        if isinstance(configs, Namespace):
            for key, value in vars(configs).items():
                setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.configs = configs

    def get_loss(self):
        self.loss = getattr(__import__("losses"), self.loss)()

    def get_model(self):
        self.model = getattr(__import__("models"), self.model)(self.configs).to(
            self.device
        )

    def get_optimizer(self):
        self.optimizer = getattr(__import__("optimizers"), self.optimizer)(
            params=self.model.parameters(), configs=self.configs
        )

    def get_scheduler(self):
        self.scheduler = getattr(__import__("schedulers"), self.scheduler)(
            optimizer=self.optimizer, configs=self.configs
        )

    def mkdir(self):
        self.save_path = os.path.join(self.save_path, str(self.times))
        self.model_path = os.path.join(self.save_path, "models")
        self.model_info_path = os.path.join(self.save_path, "models_info")
        self.log_path = os.path.join(self.save_path, "logs")
        self.result_path = os.path.join(self.save_path, "results")
        for dir in [
            self.save_path,
            self.model_path,
            self.log_path,
            self.model_info_path,
            self.result_path,
        ]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def load_data(self, path):
        x = np.load(path + "_x.npy")
        y = np.load(path + "_y.npy")
        x = self.scaler.transform(x)
        y = self.scaler.transform(y)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return torch.utils.data.TensorDataset(x, y)

    def load_data_with_ratio(self, file, sample_ratio=1.0, shuffle=False):
        """
        General method to load and subsample data.

        Args:
            file (str): Path to the data file.
            sample_ratio (float): Ratio of data to load (between 0 and 1).
            shuffle (bool): Whether to shuffle the data (True for training, False otherwise).
        """
        assert 0 <= sample_ratio <= 1, "sample_ratio must be between 0 and 1"

        dataset = self.load_data(file)

        # Apply subsampling if necessary
        if sample_ratio < 1.0:
            subset_size = int(len(dataset) * sample_ratio)
            indices = torch.randperm(len(dataset))[:subset_size]
            dataset = torch.utils.data.Subset(dataset, indices)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle
        )

        return dataloader

    def calculate_loss(self, model, dataloader, criterion):
        losses = []
        model.eval()
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            losses.append(loss.item())
        return losses

    def save_model(self, postfix=""):
        path = os.path.join(
            self.model_path, "_".join([self.name.lower().strip(), postfix]) + ".pt"
        )
        torch.save(self.model, path)
        self.logger.info(f"Model saved to {path}")

    def get_total_model_size(self, model):
        total_size = 0

        # Include parameters (weights)
        for param in model.parameters():
            total_size += param.element_size() * param.nelement()

        # Include buffers (e.g., batch norm statistics)
        for buffer in model.buffers():
            total_size += buffer.element_size() * buffer.nelement()

        return total_size

    def get_tensor_size(self, tensor):
        return tensor.element_size() * tensor.nelement()

    def get_dataset_size(self, dataset):
        # Calculate total dataset size by summing over all data
        total_size = 0
        for data in dataset:
            for item in data:
                if isinstance(item, torch.Tensor):
                    total_size += item.element_size() * item.nelement()
        return total_size

    def get_size(self, obj):
        if isinstance(obj, torch.Tensor):
            return self.get_tensor_size(obj) / pow(1024, 2)  # Size in MB
        if isinstance(obj, nn.Module):
            return self.get_total_model_size(obj) / pow(1024, 2)  # Size in MB
        if isinstance(obj, DataLoader):
            # Estimate DataLoader size by considering the dataset size and batch size
            return self.get_dataset_size(obj.dataset) / pow(1024, 2)  # Size in MB
        return sys.getsizeof(obj) / pow(1024, 2)  # Size in MB

    def summarize_model(self, dataloader):
        getattr(__import__("utils"), "ModelSummarizer")(
            model=self.model,
            dataloader=dataloader,
            save_path=os.path.join(
                self.model_info_path, f"{self.name.lower().strip()}.svg"
            ),
        ).execute()

    def train_one_epoch(
        self, model, dataloader, optimizer, criterion, scheduler, device
    ):
        model.train()
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()


class Server(SharedMethods):
    def __init__(self, configs, times):
        super().__init__()
        self.set_configs(configs=configs, times=times)
        self.mkdir()

        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients

        self.name = "  SERVER  "

        self.metrics = {
            "time_per_iter": [],
            "global_avg_val_loss": [],
            "personal_avg_val_loss": [],
            "global_avg_train_loss": [],
            "personal_avg_train_loss": [],
            "global_avg_test_loss": [],
            "personal_avg_test_loss": [],
            "send_mb": [],
        }

        self.make_logger(name=self.name, path=self.log_path)
        self.get_model()
        self.set_clients()
        self.get_model_info()

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        self.selected_clients = list(
            np.random.choice(self.clients, self.current_num_join_clients, replace=False)
        )

    def variables_to_be_sent(self):
        return {"model": self.model}

    def send_to_clients(self):
        b = 0
        to_be_sent = self.variables_to_be_sent()
        for client in self.clients:
            s = time.time()
            for value in to_be_sent.values():
                b += self.get_size(value)
            client.initialize_local(to_be_sent["model"])
            client.metrics["send_time"].append(2 * (time.time() - s))
        self.metrics["send_mb"].append(b)

    def set_clients(self):
        module_name = self.__module__
        class_name = self.__class__.__name__ + "_Client"
        try:
            client_object = getattr(
                __import__(module_name, fromlist=[class_name]), class_name
            )
        except:
            client_object = Client
        self.clients = [
            client_object(self.configs, id, self.model, self.times)
            for id in range(self.num_clients)
        ]

    def save_results(self):
        super().save_results()
        for client in self.clients:
            client.save_results()

    def save_lastest_models(self):
        super().save_model(postfix="last")
        if not self.save_local_model:
            return
        for client in self.clients:
            client.save_model(postfix="last")

    def save_best_model(self):
        metric = (
            self.metrics["personal_avg_val_loss"]
            if self.save_local_model
            else self.metrics["global_avg_val_loss"]
        )
        if metric[-1] != min(metric):
            return
        super().save_model(postfix="best")
        if not self.save_local_model:
            return
        for client in self.clients:
            client.save_model(postfix="best")

    def early_stopping(self):
        metric = (
            self.metrics["personal_avg_val_loss"]
            if self.save_local_model
            else self.metrics["global_avg_val_loss"]
        )
        if self.patience is None or len(metric) < self.patience:
            return False

        min_val = min(metric[-self.patience :])
        latest_val = metric[-1]

        if abs(latest_val - min_val) > 0.0001:
            patience_val = self.patience - len(metric) + metric.index(min(metric))
            self.logger.info(f"Patience: {patience_val:.4f}")
            return False

        return True

    def evaluate_generalization_trainset(self):
        losses = [
            np.mean(
                self.calculate_loss(self.model, client.load_train_data(), client.loss)
            )
            for client in self.clients
        ]
        self.metrics["global_avg_train_loss"].append(np.mean(losses))
        self.logger.info(
            f"Generalization Training Loss: {self.metrics['global_avg_train_loss'][-1]:.4f}"
        )

    def evaluate_generalization_valset(self):
        losses = [
            np.mean(
                self.calculate_loss(self.model, client.load_valid_data(), client.loss)
            )
            for client in self.clients
        ]
        self.metrics["global_avg_val_loss"].append(np.mean(losses))
        self.logger.info(
            f"Generalization Validation Loss: {self.metrics['global_avg_val_loss'][-1]:.4f}"
        )

    def evaluate_personalization_trainset(self):
        losses = [client.get_train_loss() for client in self.clients]
        self.metrics["personal_avg_train_loss"].append(np.mean(losses))
        self.logger.info(
            f"Personalization Training Loss: {self.metrics['personal_avg_train_loss'][-1]:.4f}"
        )

    def evaluate_personalization_valset(self):
        losses = [client.get_valid_loss() for client in self.clients]
        self.metrics["personal_avg_val_loss"].append(np.mean(losses))
        self.logger.info(
            f"Personalization Validation Loss: {self.metrics['personal_avg_val_loss'][-1]:.4f}"
        )

    def evaluate_personalization_testset(self):
        losses = [client.get_test_loss() for client in self.clients]
        self.metrics["personal_avg_test_loss"].append(np.mean(losses))
        self.logger.info(
            f"Personalization Test Loss: {self.metrics['personal_avg_test_loss'][-1]:.4f}"
        )

    def evaluate_generalization_testset(self):
        losses = [
            np.mean(
                self.calculate_loss(self.model, client.load_test_data(), client.loss)
            )
            for client in self.clients
        ]
        self.metrics["global_avg_test_loss"].append(np.mean(losses))
        self.logger.info(
            f"Generalization Test Loss: {self.metrics['global_avg_test_loss'][-1]:.4f}"
        )

    def evaluate(self):
        # print('Evaluating')
        if self.current_iter % self.eval_gap == 0:
            self.logger.info("")
            self.logger.info(
                f"-------------Round number: {str(self.current_iter).zfill(4)}-------------"
            )
            self.evaluate_generalization_trainset()
            self.evaluate_personalization_trainset()
            self.evaluate_generalization_valset()
            self.evaluate_personalization_valset()
            self.evaluate_generalization_testset()
            self.evaluate_personalization_testset()

    def train_clients(self):
        for client in self.selected_clients:
            client.train()

    def fix_results(self):
        super().fix_results()
        for client in self.clients:
            client.fix_results(self.default_value)

    def receive_from_clients(self):
        self.client_data = defaultdict(list)

        for client in self.selected_clients:
            try:
                received_data = client.send_to_server()

                for key, value in received_data.items():
                    self.client_data[key].append(value)

            except Exception as e:
                print(f"Failed to receive data from client {client.id}: {e}")

    def calculate_aggregation_weights(self):
        self.weights = torch.tensor(self.client_data["train_samples"]).to(
            self.device
        ) / sum(self.client_data["train_samples"])

    def reset_model(self, model):
        result = copy.deepcopy(model)
        for param in result.parameters():
            param.data.zero_()
        return result

    def aggregate_models(self):
        self.model = self.reset_model(self.model)
        for client, weight in zip(self.client_data["model"], self.weights):
            for global_param, local_param in zip(
                self.model.parameters(), client.parameters()
            ):
                global_param.data.add_(local_param.data, alpha=weight)

    def _last_eval(self, model, dataloader, scaler=None):
        metrics = {}
        func = getattr(__import__("losses"), "evaluation_result")
        model.eval()
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            outputs = model(batch_x)
            if scaler is not None:
                outputs = torch.tensor(
                    scaler.inverse_transform(outputs.cpu().detach().numpy())
                )
                batch_y = torch.tensor(
                    scaler.inverse_transform(batch_y.cpu().detach().numpy())
                )
            for key, value in func(y_pred=outputs, y_true=batch_y).items():
                if key in metrics:
                    metrics[key].append(value)  # Extend the existing list in a
                else:
                    metrics[key] = [value]  # If key is not in a, add it
        for key, value in metrics.items():
            metrics[key] = np.mean(value)
        return metrics

    def evaluate_all_metrics(self):
        merged_testset = DataLoader(
            torch.utils.data.ConcatDataset(
                [client.load_test_data().dataset for client in self.clients]
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )

        results = [
            dict(
                {"entity": self.name, "denorm": False, "type": "last"},
                **self._last_eval(
                    model=torch.load(
                        os.path.join(
                            self.model_path, self.name.lower().strip() + "_last.pt"
                        )
                    ),
                    dataloader=merged_testset,
                ),
            ),
            dict(
                {"entity": self.name, "denorm": False, "type": "best"},
                **self._last_eval(
                    model=torch.load(
                        os.path.join(
                            self.model_path, self.name.lower().strip() + "_best.pt"
                        )
                    ),
                    dataloader=merged_testset,
                ),
            ),
        ]
        if self.save_local_model:
            for client in self.clients:
                for scaler in [None, client.scaler]:
                    results.extend(
                        [
                            dict(
                                {
                                    "entity": client.name,
                                    "denorm": scaler is not None,
                                    "type": "last",
                                },
                                **self._last_eval(
                                    model=torch.load(
                                        os.path.join(
                                            client.model_path,
                                            client.name.lower().strip() + "_last.pt",
                                        )
                                    ),
                                    dataloader=client.load_test_data(),
                                    scaler=scaler,
                                ),
                            ),
                            dict(
                                {
                                    "entity": client.name,
                                    "denorm": scaler is not None,
                                    "type": "best",
                                },
                                **self._last_eval(
                                    model=torch.load(
                                        os.path.join(
                                            client.model_path,
                                            client.name.lower().strip() + "_best.pt",
                                        )
                                    ),
                                    dataloader=client.load_test_data(),
                                    scaler=scaler,
                                ),
                            ),
                        ]
                    )
        results = pl.from_dicts(results)
        print(results)
        path = os.path.join(self.result_path, "all.csv")
        results.write_csv(path)
        self.logger.info(f"Results saved to {path}")

    def get_model_info(self):
        self.summarize_model(dataloader=self.clients[0].load_train_data())
        if self.save_local_model:
            for client in self.clients:
                client.summarize_model(dataloader=client.load_train_data())

    def train(self):
        for i in range(self.iterations):
            s_t = time.time()
            self.current_iter = i
            self.select_clients()
            self.send_to_clients()
            self.evaluate()
            self.train_clients()
            self.receive_from_clients()
            self.calculate_aggregation_weights()
            self.aggregate_models()
            self.save_best_model()
            self.metrics["time_per_iter"].append(time.time() - s_t)
            self.logger.info(f'Time cost: {self.metrics["time_per_iter"][-1]:.4f}s')
            self.fix_results()
            if self.early_stopping():
                self.logger.info("Early stopping activated.")
                break
        self.logger.info("")
        self.logger.info("-" * 50)
        self.save_lastest_models()
        self.save_results()
        self.evaluate_all_metrics()


class Client(SharedMethods):
    def __init__(self, configs: dict, id: int, model: nn.Module, times: int):
        super().__init__()
        self.set_configs(configs=configs, id=id, times=times)
        self.mkdir()
        self.model = copy.deepcopy(model)
        self.metrics = {
            "train_time": [],
            "send_time": [],
            "val_loss": [],
            "train_loss": [],
            "test_loss": [],
            "send_mb": [],
        }

        self.get_loss()
        self.get_optimizer()
        self.get_scheduler()
        self.name = f"CLIENT_{str(self.id).zfill(3)}"
        self.make_logger(name=self.name, path=self.log_path)
        self.train_file = os.path.join(self.dataset_path, "train/", str(self.id))
        self.valid_file = os.path.join(self.dataset_path, "valid/", str(self.id))
        self.test_file = os.path.join(self.dataset_path, "test/", str(self.id))
        self.stats = json.load(open(self.path_info))["clients"][id]["stats"]["train"]
        self.get_scaler()

    def variables_to_be_sent(self):
        model = self.model if not self.return_diff else self.diff
        return {"model": model, "train_samples": self.train_samples}

    def send_to_server(self):
        to_be_sent = self.variables_to_be_sent()
        b = 0
        for value in to_be_sent.values():
            b += self.get_size(value)
        self.metrics["send_mb"].append(b)
        return to_be_sent

    def get_scaler(self):
        self.scaler = getattr(__import__("scalers"), self.scaler)(self.stats)

    def load_train_data(self, sample_ratio=1.0, shuffle=False):
        trainloader = self.load_data_with_ratio(
            file=self.train_file, sample_ratio=sample_ratio, shuffle=shuffle
        )
        self.train_samples = len(trainloader.dataset)
        return trainloader

    def load_test_data(self, sample_ratio=1.0, shuffle=False):
        testloader = self.load_data_with_ratio(
            file=self.test_file, sample_ratio=sample_ratio, shuffle=shuffle
        )
        self.test_samples = len(testloader.dataset)
        return testloader

    def load_valid_data(self, sample_ratio=1.0, shuffle=False):
        validloader = self.load_data_with_ratio(
            file=self.valid_file, sample_ratio=sample_ratio, shuffle=shuffle
        )
        self.valid_samples = len(validloader.dataset)
        return validloader

    def initialize_local(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def train(self):
        train_loader = self.load_train_data()
        start_time = time.time()
        if self.return_diff:
            self.snapshot = copy.deepcopy(self.model)
        for _ in range(self.epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=train_loader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
            )
        if self.return_diff:
            diff = {
                key: param_old - param_new
                for (key, param_old), param_new in zip(
                    self.snapshot.named_parameters(), self.model.parameters()
                )
            }
            self.diff = copy.deepcopy(self.model)
            self.diff.load_state_dict(diff)
        self.metrics["train_time"].append(time.time() - start_time)

    def get_valid_loss(self):
        losses = self.calculate_loss(self.model, self.load_valid_data(), self.loss)
        losses = np.mean(losses)
        self.metrics["val_loss"].append(losses)
        return losses

    def get_train_loss(self):
        losses = self.calculate_loss(self.model, self.load_train_data(), self.loss)
        losses = np.mean(losses)
        self.metrics["train_loss"].append(losses)
        return losses

    def get_test_loss(self):
        losses = self.calculate_loss(self.model, self.load_test_data(), self.loss)
        losses = np.mean(losses)
        self.metrics["test_loss"].append(losses)
        return losses
