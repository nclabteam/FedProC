import torch

from losses.sMdAPE import sMdAPE


def test_smdape_uses_symmetric_absolute_percentage_error():
    prediction = torch.tensor([1.0, 2.0, 5.0])
    target = torch.tensor([1.0, 4.0, 3.0])

    result = sMdAPE()(prediction, target)

    assert torch.isclose(result, torch.tensor(50.0))
