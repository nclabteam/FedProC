from .nFL import nFL, nFL_Client


class LocalOnly(nFL):
    def evaluate_generalization_loss(self, *args, **kwargs):
        pass


class LocalOnly_Client(nFL_Client):
    pass
