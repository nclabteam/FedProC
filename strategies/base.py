import copy
import json
import logging
import os
import sys
import time
from argparse import Namespace

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class SharedMethods:
    def __init__(self):
        self.default_value = 9_999_999

    def save_results(self):
        pl_df = pl.DataFrame(self.metrics)
        path = os.path.join(self.result_path, self.name.lower().strip() + ".csv")
        pl_df.write_csv(path)
        self.logger.info(f"Results saved to {path}")

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

    def initialize_loss(self):
        self.loss = getattr(__import__("losses"), self.loss)()

    def initialize_model(self):
        self.model = getattr(__import__("models"), self.model)(self.configs).to(
            self.device
        )

    def initialize_optimizer(self):
        self.optimizer = getattr(__import__("optimizers"), self.optimizer)(
            params=self.model.parameters(), configs=self.configs
        )

    def initialize_scheduler(self):
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

    @staticmethod
    def load_data(file, sample_ratio=1.0, batch_size=32, shuffle=False, scaler=None):
        """
        General method to load and subsample data.

        Args:
            file (str): Path to the data file.
            sample_ratio (float): Ratio of data to load (between 0 and 1).
            shuffle (bool): Whether to shuffle the data (True for training, False otherwise).
            batch_size (int): Batch size for the DataLoader.
            scaler
        """
        assert 0 <= sample_ratio <= 1, "sample_ratio must be between 0 and 1"

        data = np.load(file)
        x = data["x"]
        y = data["y"]
        x = scaler.transform(x)
        y = scaler.transform(y)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x, y)

        # Apply subsampling if necessary
        if sample_ratio < 1.0:
            subset_size = int(len(dataset) * sample_ratio)
            indices = torch.randperm(len(dataset))[:subset_size]
            dataset = torch.utils.data.Subset(dataset, indices)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        return dataloader

    @staticmethod
    def calculate_loss(model, dataloader, criterion, device="cpu"):
        losses = []
        model.eval()
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            losses.append(loss.item())
        return losses

    def save_model(self, postfix="", extention="pt", verbose=True):
        path = os.path.join(
            self.model_path,
            f"{'_'.join([self.name.lower().strip(), postfix])}.{extention}",
        )
        torch.save(obj=self.model, f=path)

        # Display a message if verbose is set to True
        if verbose:
            self.logger.info(f"Model saved to {path}")

    @staticmethod
    def reset_model(model):
        result = copy.deepcopy(model)
        for param in result.parameters():
            param.data.zero_()
        return result

    @staticmethod
    def get_total_model_size(model):
        total_size = 0

        # Include parameters (weights)
        for param in model.parameters():
            total_size += param.element_size() * param.nelement()

        # Include buffers (e.g., batch norm statistics)
        for buffer in model.buffers():
            total_size += buffer.element_size() * buffer.nelement()

        return total_size

    @staticmethod
    def get_tensor_size(tensor):
        return tensor.element_size() * tensor.nelement()

    @staticmethod
    def get_dataset_size(dataset):
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

    @staticmethod
    def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, device):
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

    def update_model_params(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()


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
            "global_avg_valid_loss": [],
            "personal_avg_valid_loss": [],
            "global_avg_train_loss": [],
            "personal_avg_train_loss": [],
            "global_avg_test_loss": [],
            "personal_avg_test_loss": [],
            "send_mb": [],
        }

        self.make_logger(name=self.name, path=self.log_path)
        self.initialize_model()
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
        for idx, client in enumerate(self.clients):
            s = time.time()
            c = {}
            for key, value in to_be_sent.items():
                if isinstance(value, list):
                    value = value[idx]
                b += self.get_size(value)
                c[key] = value
            client.receive_from_server(c)
            client.metrics["send_time"].append(2 * (time.time() - s))
        self.metrics["send_mb"].append(b)

    def receive_from_clients(self):
        self.client_data = []

        for client in self.selected_clients:
            try:
                self.client_data.append(client.send_to_server())

            except Exception as e:
                print(f"Failed to receive data from client {client.id}: {e}")

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
            client_object(configs=self.configs, id=id, times=self.times)
            for id in range(self.num_clients)
        ]

    def save_results(self):
        super().save_results()
        for client in self.clients:
            client.save_results()

    def save_lastest_models(self):
        self.save_model(postfix="last")
        if not self.save_local_model:
            return
        for client in self.clients:
            client.save_model(postfix="last")

    def save_best_model(self):
        metric = (
            self.metrics["personal_avg_valid_loss"]
            if self.save_local_model
            else self.metrics["global_avg_valid_loss"]
        )
        if metric[-1] != min(metric):
            return
        self.save_model(postfix="best")
        if not self.save_local_model:
            return
        for client in self.clients:
            client.save_model(postfix="best")

    def early_stopping(self):
        metric = (
            self.metrics["personal_avg_valid_loss"]
            if self.save_local_model
            else self.metrics["global_avg_valid_loss"]
        )
        if not self.patience or len(metric) < self.patience:
            return False

        min_val = min(metric[-self.patience :])
        latest_val = metric[-1]

        if abs(latest_val - min_val) > 0.0001:
            patience_val = self.patience - len(metric) + metric.index(min(metric))
            self.logger.info(f"Patience: {patience_val:.4f}")
            return False

        return True

    def evaluate_generalization_loss(self, dataset_type):
        """
        Generalized function to evaluate loss for a given dataset type and loss method.

        Parameters:
        - dataset_type: str, type of dataset ('train', 'valid', 'test')
        """
        losses = [
            np.mean(
                self.calculate_loss(
                    model=self.model,
                    dataloader=getattr(client, f"load_{dataset_type}_data")(),
                    criterion=client.loss,
                    device=client.device,
                )
            )
            for client in self.clients
        ]

        metric_name = f"global_avg_{dataset_type}_loss"
        self.metrics[metric_name].append(np.mean(losses))
        self.logger.info(
            f"Generalization {dataset_type.capitalize()} Loss: {self.metrics[metric_name][-1]:.4f}"
        )

    def evaluate_personalization_loss(self, dataset_type):
        """
        Generalized function to evaluate personalization loss for a given dataset type.

        Parameters:
        - dataset_type: str, type of dataset ('train', 'valid', 'test')
        """
        losses = [
            getattr(client, f"get_{dataset_type}_loss")() for client in self.clients
        ]

        metric_name = f"personal_avg_{dataset_type}_loss"
        self.metrics[metric_name].append(np.mean(losses))
        self.logger.info(
            f"Personalization {dataset_type.capitalize()} Loss: {self.metrics[metric_name][-1]:.4f}"
        )

    def evaluate(self):
        self.logger.info("")
        self.logger.info(
            f"-------------Round number: {str(self.current_iter).zfill(4)}-------------"
        )
        if self.current_iter % self.eval_gap == 0:
            for dataset_type in ["train", "valid", "test"]:
                # Generalization loss evaluation
                self.evaluate_generalization_loss(dataset_type)
                # Personalization loss evaluation
                self.evaluate_personalization_loss(dataset_type)

    def train_clients(self):
        for client in self.selected_clients:
            client.train()

    def fix_results(self):
        super().fix_results(default=self.default_value)
        for client in self.clients:
            client.fix_results(default=self.default_value)

    def calculate_aggregation_weights(self):
        ts = [client["train_samples"] for client in self.client_data]
        self.weights = torch.tensor(ts).to(self.device) / sum(ts)

    def aggregate_models(self):
        self.model = self.reset_model(self.model)
        for client, weight in zip(self.client_data, self.weights):
            for global_param, local_param in zip(
                self.model.parameters(), client["model"].parameters()
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
                        ),
                        weights_only=False,
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
                        ),
                        weights_only=False,
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
                                        ),
                                        weights_only=False,
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
                                        ),
                                        weights_only=False,
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
    def __init__(self, configs: dict, id: int, times: int):
        super().__init__()
        self.set_configs(configs=configs, id=id, times=times)
        self.mkdir()
        self.initialize_model()
        self.initialize_loss()
        self.initialize_optimizer()
        self.initialize_scheduler()
        self.initialize_data_paths()
        self.initialize_stats()
        self.initialize_scaler()

        self.name = f"CLIENT_{str(self.id).zfill(3)}"
        self.make_logger(name=self.name, path=self.log_path)

        self.metrics = {
            "train_time": [],
            "send_time": [],
            "val_loss": [],
            "train_loss": [],
            "test_loss": [],
            "send_mb": [],
        }

    def initialize_data_paths(self):
        self.train_file = os.path.join(self.dataset_path, "train", f"{self.id}.npz")
        self.valid_file = os.path.join(self.dataset_path, "valid", f"{self.id}.npz")
        self.test_file = os.path.join(self.dataset_path, "test", f"{self.id}.npz")

    def initialize_scaler(self):
        self.scaler = getattr(__import__("scalers"), self.scaler)(self.stats)

    def initialize_stats(self):
        self.stats = json.load(fp=open(self.path_info))["clients"][self.id]["stats"][
            "train"
        ]

    def variables_to_be_sent(self):
        return {"model": self.model, "train_samples": self.train_samples}

    def send_to_server(self):
        to_be_sent = self.variables_to_be_sent()
        b = 0
        for value in to_be_sent.values():
            b += self.get_size(value)
        self.metrics["send_mb"].append(b)
        return to_be_sent

    def load_train_data(self, sample_ratio=1.0, shuffle=False):
        trainloader = self.load_data(
            file=self.train_file,
            sample_ratio=sample_ratio,
            shuffle=shuffle,
            scaler=self.scaler,
            batch_size=self.batch_size,
        )
        self.train_samples = len(trainloader.dataset)
        return trainloader

    def load_test_data(self, sample_ratio=1.0, shuffle=False):
        testloader = self.load_data(
            file=self.test_file,
            sample_ratio=sample_ratio,
            shuffle=shuffle,
            scaler=self.scaler,
            batch_size=self.batch_size,
        )
        self.test_samples = len(testloader.dataset)
        return testloader

    def load_valid_data(self, sample_ratio=1.0, shuffle=False):
        validloader = self.load_data(
            file=self.valid_file,
            sample_ratio=sample_ratio,
            shuffle=shuffle,
            scaler=self.scaler,
            batch_size=self.batch_size,
        )
        self.valid_samples = len(validloader.dataset)
        return validloader

    def train(self):
        train_loader = self.load_train_data()
        start_time = time.time()
        for _ in range(self.epochs):
            self.train_one_epoch(
                model=self.model,
                dataloader=train_loader,
                optimizer=self.optimizer,
                criterion=self.loss,
                scheduler=self.scheduler,
                device=self.device,
            )
        self.metrics["train_time"].append(time.time() - start_time)

    def get_valid_loss(self):
        losses = self.calculate_loss(
            model=self.model,
            dataloader=self.load_valid_data(),
            criterion=self.loss,
            device=self.device,
        )
        losses = np.mean(losses)
        self.metrics["val_loss"].append(losses)
        return losses

    def get_train_loss(self):
        losses = self.calculate_loss(
            model=self.model,
            dataloader=self.load_train_data(),
            criterion=self.loss,
            device=self.device,
        )
        losses = np.mean(losses)
        self.metrics["train_loss"].append(losses)
        return losses

    def get_test_loss(self):
        losses = self.calculate_loss(
            model=self.model,
            dataloader=self.load_test_data(),
            criterion=self.loss,
            device=self.device,
        )
        losses = np.mean(losses)
        self.metrics["test_loss"].append(losses)
        return losses

    def receive_from_server(self, data):
        self.update_model_params(data["model"])
