import copy

from topologies import TOPOLOGIES

from .base import Client, Server
from .DFL import DFL, DFL_Client

optional = {
    "mu": 0.01,
    "topology": "FullyConnected",
}

DFedProx_compulsory = {
    "save_local_model": True,
    "exclude_server_model_processes": True,
}


def args_update(parser):
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--topology", type=str, default=None, choices=TOPOLOGIES)


class FedProx(Server):
    pass


class FedProx_Client(Client):
    def receive_from_server(self, data):
        self.snapshot = copy.deepcopy(data["model"]).to("cpu")
        self.update_model_params(old=self.model, new=data["model"])

    def train_one_epoch(
        self,
        model,
        dataloader,
        optimizer,
        criterion,
        scheduler,
        device,
        offload_after=True,
    ):
        model.to(device)
        self._move_optimizer_state_to_param_devices(optimizer)
        global_params = copy.deepcopy(list(self.snapshot.to(device).parameters()))
        model.train()
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            for w, w_t in zip(model.parameters(), global_params):
                if w.requires_grad:
                    w.grad.data += self.mu * (w.data - w_t.data)
            optimizer.step()
        scheduler.step()
        if offload_after:
            model.to("cpu")
        del global_params


class DFedProx(DFL):
    """Decentralized FedProx: DFL aggregation plus local proximal training."""


class DFedProx_Client(DFL_Client):
    def aggregate_models(self):
        super().aggregate_models()
        self.snapshot = copy.deepcopy(self.model).to("cpu")

    def train_one_epoch(self, *args, **kwargs):
        if not hasattr(self, "snapshot"):
            model = args[0] if args else kwargs["model"]
            self.snapshot = copy.deepcopy(model).to("cpu")
        return FedProx_Client.train_one_epoch(self, *args, **kwargs)
