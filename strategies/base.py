import copy
import json
import logging
import multiprocessing
import os
import sys
import time
from argparse import Namespace
from collections import deque
from functools import partial

import numpy as np
import polars as pl
import ray
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset


class SharedMethods:
    def __init__(self):
        self.default_value = 9_999_999.0

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
        self.model = getattr(__import__("models"), self.model)(configs=self.configs).to(
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
        dataset = TensorDataset(x, y)

        # Apply subsampling if necessary
        if sample_ratio < 1.0:
            subset_size = int(len(dataset) * sample_ratio)
            indices = torch.randperm(len(dataset))[:subset_size]
            dataset = Subset(dataset, indices)

        dataloader = DataLoader(
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
    def get_size(obj):
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement() / (1024**2)  # Size in MB
        if isinstance(obj, torch.nn.Module):
            total_size = sum(
                param.element_size() * param.nelement() for param in obj.parameters()
            )
            total_size += sum(
                buffer.element_size() * buffer.nelement() for buffer in obj.buffers()
            )  # Include buffers
            return total_size / (1024**2)  # Size in MB
        if isinstance(obj, DataLoader):
            total_size = sum(
                sum(
                    item.element_size() * item.nelement()
                    for item in data
                    if isinstance(item, torch.Tensor)
                )
                for data in obj.dataset
            )
            return total_size / (1024**2)  # Size in MB
        return sys.getsizeof(obj) / (1024**2)  # Size in MB

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

    @staticmethod
    def update_model_params(old, new):
        """Update the parameters of old_model with those from new_model."""
        for old_param, new_param in zip(old.parameters(), new.parameters()):
            old_param.data.copy_(new_param.data)

    @staticmethod
    def update_optimizer_params(old, new):
        """Update the parameters and hyperparameters of old_optimizer with those from new_optimizer."""
        for old_group, new_group in zip(old.param_groups, new.param_groups):
            # Update all hyperparameters dynamically
            for key in new_group.keys():
                if key != "params":  # Skip updating "params" directly
                    old_group[key] = new_group[key]

            # Update the model parameters inside param_groups
            for old_param, new_param in zip(old_group["params"], new_group["params"]):
                old_param.data.copy_(new_param.data)


class Server(SharedMethods):
    def __init__(self, configs, times):
        super().__init__()
        self.set_configs(configs=configs, times=times)
        self.mkdir()

        self.num_join_clients = max(1, int(self.num_clients * self.join_ratio))
        self.current_num_join_clients = self.num_join_clients

        self.num_gpus = len(self.device_id.split(","))
        configs.parallel = True if self.num_gpus > 1 else False
        self.parallel = configs.parallel
        ray.init(
            num_gpus=self.num_gpus,
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
            log_to_driver=False,
        )

        self.name = "  SERVER  "

        self.metrics = {
            "time_per_iter": [],
            "global_avg_train_loss": [],
            "personal_avg_train_loss": [],
            "global_avg_test_loss": [],
            "personal_avg_test_loss": [],
            "send_mb": [],
        }

        self.make_logger(name=self.name, path=self.log_path)
        self.initialize_clients()
        self.configs.__dict__["input_channels"] = self.clients[0].input_channels
        self.input_channels = self.clients[0].input_channels
        self.configs.__dict__["output_channels"] = self.clients[0].output_channels
        self.output_channels = self.clients[0].output_channels
        self.initialize_model()
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

    def initialize_clients(self):
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
            self.metrics["personal_avg_test_loss"]
            if self.save_local_model
            else self.metrics["global_avg_test_loss"]
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
            self.metrics["personal_avg_test_loss"]
            if self.save_local_model
            else self.metrics["global_avg_test_loss"]
        )

        if not self.patience or len(metric) < self.patience:
            return False

        # Find the minimum loss so far
        best_so_far = min(metric)

        # Check if the minimum value is in the last `self.patience` rounds
        if best_so_far not in metric[-self.patience :]:
            self.logger.info("Early stopping activated.")
            return True

        return False

    def _compute_generalization_loss(self, client, dataset_type, model):
        """
        Compute the loss for a single client. This will be executed in parallel.

        Parameters:
        - client: Client object
        - dataset_type: str, type of dataset ('train', 'valid', 'test')
        - model: model being evaluated
        """
        return np.mean(
            self.calculate_loss(
                model=model,
                dataloader=getattr(client, f"load_{dataset_type}_data")(),
                criterion=client.loss,
                device=client.device,
            )
        )

    def evaluate_generalization_loss(self, dataset_type):
        """
        Generalized function to evaluate loss for a given dataset type and loss method.

        Parameters:
        - dataset_type: str, type of dataset ('train', 'valid', 'test')
        """
        if 1 < self.num_workers < self.num_clients:
            # Create a pool of processes
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                # Prepare the partial function with the necessary arguments
                func = partial(
                    self._compute_generalization_loss,
                    dataset_type=dataset_type,
                    model=self.model,
                )

                # Run the function in parallel for each client
                losses = pool.map(func, self.clients)

        else:
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

    def _compute_personalization_loss(self, client, dataset_type):
        """
        Compute the loss for a single client.

        Parameters:
        - client: Client object
        - dataset_type: str, type of dataset ('train', 'test')
        """
        return getattr(client, f"get_{dataset_type}_loss")()

    def evaluate_personalization_loss(self, dataset_type):
        """
        Generalized function to evaluate personalization loss for a given dataset type using multiprocessing.

        Parameters:
        - dataset_type: str, type of dataset ('train', 'test')
        """
        if 1 < self.num_workers < self.num_clients:
            # Use multiprocessing Pool to parallelize the loss computation
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                losses = pool.starmap(
                    self._compute_personalization_loss,
                    [(client, dataset_type) for client in self.clients],
                )
        else:
            # Non-parallel version
            losses = [
                getattr(client, f"get_{dataset_type}_loss")() for client in self.clients
            ]

        # Calculate and log the average personalization loss
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
            for dataset_type in ["train", "test"]:
                if dataset_type == "train" and self.skip_eval_train:
                    continue
                # Generalization loss evaluation
                self.evaluate_generalization_loss(dataset_type)
                # Personalization loss evaluation
                if not self.save_local_model:
                    continue
                self.evaluate_personalization_loss(dataset_type)

    def train_clients(self):
        if self.parallel:
            i = 0
            futures = []
            idle_workers = deque(range(self.num_workers))
            job_map = {}
            client_packages = {}

            while i < len(self.selected_clients) or len(futures) > 0:
                while i < len(self.selected_clients) and len(idle_workers) > 0:
                    worker_id = idle_workers.popleft()
                    client = self.selected_clients[i]

                    # Parallelize the `train()` method using Ray
                    future = ray.remote(num_gpus=self.num_gpus / self.num_workers)(
                        lambda cl: cl.train()
                    ).remote(client)

                    job_map[future] = (client, worker_id)
                    futures.append(future)
                    i += 1

                if len(futures) > 0:
                    all_finished, futures = ray.wait(futures)
                    for finished in all_finished:
                        client, worker_id = job_map[finished]
                        client_package = ray.get(finished)
                        idle_workers.append(worker_id)
                        client_packages[client] = client_package

                        # Update client model & optimizer (since clients are normal Python objects)
                        client.update_model_params(
                            old=client.model, new=client_package["model"]
                        )
                        client.update_optimizer_params(
                            old=client.optimizer, new=client_package["optimizer"]
                        )
                        client.metrics["train_time"].append(
                            client_package["train_time"]
                        )
                        client.train_samples = client_package["train_samples"]

        else:
            [client.train() for client in self.selected_clients]  # Serial execution

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
        if not self.last_eval:
            return

        merged_testset = DataLoader(
            ConcatDataset([client.load_test_data().dataset for client in self.clients]),
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

    def post_process(self):
        self.logger.info("")
        self.logger.info("-" * 50)
        self.save_lastest_models()
        self.save_results()
        self.evaluate_all_metrics()

    def train(self):
        for i in range(self.iterations):
            s_t = time.time()
            self.current_iter = i
            self.select_clients()
            self.send_to_clients()
            self.train_clients()
            self.receive_from_clients()
            self.calculate_aggregation_weights()
            self.aggregate_models()
            self.evaluate()
            self.save_best_model()
            self.metrics["time_per_iter"].append(time.time() - s_t)
            self.logger.info(f'Time cost: {self.metrics["time_per_iter"][-1]:.4f}s')
            self.fix_results()
            if self.early_stopping():
                break
        self.post_process()


class Client(SharedMethods):
    def __init__(self, configs: dict, id: int, times: int):
        super().__init__()
        self.set_configs(configs=configs, id=id, times=times)
        self.mkdir()
        self.initialize_private_info()
        self.initialize_model()
        self.initialize_loss()
        self.initialize_optimizer()
        self.initialize_scheduler()
        self.initialize_scaler()

        self.name = f"CLIENT_{str(self.id).zfill(3)}"
        self.make_logger(name=self.name, path=self.log_path)

        self.metrics = {
            "train_time": [],
            "send_time": [],
            "train_loss": [],
            "test_loss": [],
            "send_mb": [],
            "lr": [],
        }

    def initialize_private_info(self):
        self.private_data = json.load(fp=open(self.path_info))[self.id]
        if self.private_data["client"] != self.id:
            raise ValueError("Client ID mismatch")
        self.train_file = self.private_data["paths"]["train"]
        self.test_file = self.private_data["paths"]["test"]
        self.stats = self.private_data["stats"]["train"]
        self.configs.__dict__["input_channels"] = self.private_data["input_channels"]
        self.input_channels = self.private_data["input_channels"]
        self.configs.__dict__["output_channels"] = self.private_data["output_channels"]
        self.output_channels = self.private_data["output_channels"]

    def initialize_scaler(self):
        self.scaler = getattr(__import__("scalers"), self.scaler)(self.stats)

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
        train_time = time.time() - start_time
        if self.parallel:
            return {
                "id": self.id,
                "model": self.model,
                "optimizer": self.optimizer,
                "train_time": train_time,
                "train_samples": self.train_samples,
            }
        self.metrics["train_time"].append(train_time)
        self.metrics["lr"].append(self.scheduler.get_last_lr()[0])

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
        self.update_model_params(old=self.model, new=data["model"])
