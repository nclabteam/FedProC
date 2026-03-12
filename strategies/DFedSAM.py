import math

from topologies import TOPOLOGIES

from .DFL import DFL, DFL_Client

optional = {
    "topology": "FullyConnected",
    "use_mgs": True,
    "mgs_steps": 2,
    "rho": 0.05,
}

compulsory = {
    "save_local_model": True,
    "exclude_server_model_processes": True,
}


def args_update(parser):
    parser.add_argument("--topology", type=str, default=None, choices=TOPOLOGIES)
    parser.add_argument("--rho", type=float, default=0.05, help="SAM radius")
    parser.add_argument(
        "--use_mgs",
        action="store_true",
        help="Enable Multiple Gossip Steps (MGS) during aggregation",
    )
    parser.add_argument(
        "--mgs_steps",
        type=int,
        default=2,
        help="Number of gossip/consensus steps when --use_mgs is enabled (>=1)",
    )


class DFedSAM(DFL):
    """DFedSAM / DFedSAM-MGS server orchestrator.

    Inherits DFL topology-based communication. Overrides aggregate_models
    to support Q gossip steps (MGS variant) for improved model consistency.
    """

    def aggregate_models(self, *args, **kwargs):
        """One gossip step (DFedSAM) or Q gossip steps (DFedSAM-MGS).

        Each gossip step = receive neighbor models + weighted average.
        DFedSAM: Q = 1 (single aggregation pass).
        DFedSAM-MGS: Q >= 2 (repeat receive + aggregate Q times).
        """
        q = 1
        if self.use_mgs:
            q = max(1, int(self.mgs_steps))

        # First gossip step.
        super().aggregate_models(*args, **kwargs)

        # Additional gossip steps for MGS:
        # re-exchange current models among neighbors and re-average.
        for _ in range(q - 1):
            self.receive_from_clients()
            self.calculate_aggregation_weights()
            super().aggregate_models(*args, **kwargs)


class DFedSAM_Client(DFL_Client):
    """Local SAM training for DFedSAM.

    Each mini-batch:
      1. Compute gradient g = nabla F(y; xi)
      2. Perturbation delta = rho * g / ||g||_2
      3. Compute perturbed gradient g_tilde = nabla F(y + delta; xi)
      4. Update y <- y - eta * g_tilde
    """

    def _grad_norm(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            total_norm += p.grad.detach().float().norm(2).item() ** 2
        return math.sqrt(total_norm)

    def _add_perturbation(self, model, rho, grad_norm):
        eps = {}
        scale = rho / (grad_norm + 1e-12)
        for p in model.parameters():
            if p.grad is None:
                eps[p] = None
                continue
            e_w = p.grad.detach() * scale
            p.data.add_(e_w)
            eps[p] = e_w
        return eps

    def _remove_perturbation(self, model, eps):
        for p in model.parameters():
            e_w = eps.get(p, None)
            if e_w is None:
                continue
            p.data.sub_(e_w)

    def train_one_epoch(
        self, model, dataloader, optimizer, criterion, scheduler, device
    ):
        """SAM training loop: two forward-backward passes per batch."""
        rho = self.rho
        model.to(device)
        model.train()
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            # Step 1: first forward/backward → compute g
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            grad_norm = self._grad_norm(model)

            # Step 2: perturb weights by delta = rho * g / ||g||
            eps = self._add_perturbation(model, rho, grad_norm)

            # Step 3: second forward/backward at (y + delta) → compute g_tilde
            optimizer.zero_grad()
            loss2 = criterion(model(batch_x), batch_y)
            loss2.backward()

            # Step 4: restore weights and step with perturbed gradient
            self._remove_perturbation(model, eps)
            optimizer.step()
        model.to("cpu")
        scheduler.step()
