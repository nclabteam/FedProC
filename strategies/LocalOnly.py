from .nFL import nFL, nFL_Client


class LocalOnly(nFL):
    def evaluate_generalization_loss(self, *args, **kwargs):
        pass

    def _pre_eval_hook(self, dataset_type: str) -> None:
        self.evaluate_personalization_loss(dataset_type)


class LocalOnly_Client(nFL_Client):
    pass
