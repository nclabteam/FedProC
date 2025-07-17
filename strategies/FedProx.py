import copy

from .base import Client, Server

optional = {
    "mu": 0.01,
}


def args_update(parser):
    parser.add_argument("--mu", type=float, default=None)


class FedProx(Server):
    pass


class FedProx_Client(Client):
    def train_one_epoch(
        self, model, dataloader, optimizer, criterion, scheduler, device
    ):
        model.to(device)
        global_params = copy.deepcopy(list(model.parameters()))
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
        model.to("cpu")
