from argparse import Namespace
from collections import OrderedDict

import pytest
import torch

from models.GRU import GRU
from models.Linear import Linear
from models.LSTM import LSTM
from strategies.FedDropout import FedDropout
from strategies.FedRolex import FedRolex
from strategies.ptFL import ptFL


def _recurrent_configs(model: str) -> Namespace:
    return Namespace(
        model=model,
        hidden_size=8,
        num_layers=2,
        input_len=5,
        output_len=2,
        input_channels=3,
        output_channels=3,
    )


@pytest.mark.parametrize(("model_name", "model_class"), [("LSTM", LSTM), ("GRU", GRU)])
def test_vertical_recurrent_plan_strict_load_and_forward(model_name, model_class):
    configs = _recurrent_configs(model_name)
    full_model = model_class(configs)
    selected = {
        "cells.0": torch.tensor([0, 2, 4, 6]),
        "cells.1": torch.tensor([1, 3, 5, 7]),
    }

    plan = ptFL._pt_build_recurrent_plan(
        full_model,
        capacity=0.5,
        selector=lambda group, _full, _retained: selected[group],
    )
    input_gate = "W_ii" if model_name == "LSTM" else "W_ir"
    recurrent_gate = "W_hi" if model_name == "LSTM" else "W_hr"
    gate_bias = "b_i" if model_name == "LSTM" else "b_r"

    assert plan.retained_widths == (4, 4)
    assert torch.equal(plan.manifest[f"cells.0.{input_gate}"][0], selected["cells.0"])
    assert torch.equal(
        plan.manifest[f"cells.0.{input_gate}"][1],
        torch.arange(configs.input_channels),
    )
    assert torch.equal(plan.manifest[f"cells.1.{input_gate}"][0], selected["cells.1"])
    assert torch.equal(plan.manifest[f"cells.1.{input_gate}"][1], selected["cells.0"])
    assert torch.equal(
        plan.manifest[f"cells.1.{recurrent_gate}"][0], selected["cells.1"]
    )
    assert torch.equal(
        plan.manifest[f"cells.1.{recurrent_gate}"][1], selected["cells.1"]
    )
    assert torch.equal(plan.manifest[f"cells.1.{gate_bias}"][0], selected["cells.1"])
    assert torch.equal(
        plan.manifest["fc_pred.weight"][0],
        torch.arange(configs.output_len * configs.input_channels),
    )
    assert torch.equal(plan.manifest["fc_pred.weight"][1], selected["cells.1"])

    full_parameters = OrderedDict(full_model.named_parameters())
    submodel_parameters = ptFL._pt_extract_parameters(full_parameters, plan.manifest)
    narrow_model = ptFL._pt_build_client_model(configs, capacity=0.5)
    narrow_model.load_state_dict(submodel_parameters, strict=True)

    assert narrow_model.hidden_size == 4
    assert narrow_model(
        torch.randn(2, configs.input_len, configs.input_channels)
    ).shape == (
        2,
        configs.output_len,
        configs.output_channels,
    )


def test_fedrolex_is_unit_stride_cyclic_and_client_independent():
    server = FedRolex.__new__(FedRolex)
    server.current_iter = 6

    expected = torch.tensor([6, 7, 0, 1])
    assert torch.equal(server._pt_select_indices("cells.0", 8, 4, 0), expected)
    assert torch.equal(server._pt_select_indices("cells.1", 8, 4, 99), expected)


def test_feddropout_is_seeded_unique_and_fresh_by_client_and_round():
    server = FedDropout.__new__(FedDropout)
    server.seed = 17
    server.times = 2
    server.current_iter = 3

    first = server._pt_select_indices("cells.0", 64, 16, 4)
    repeated = server._pt_select_indices("cells.0", 64, 16, 4)
    other_client = server._pt_select_indices("cells.0", 64, 16, 5)
    server.current_iter = 4
    next_round = server._pt_select_indices("cells.0", 64, 16, 4)

    assert torch.equal(first, repeated)
    assert first.unique().numel() == 16
    assert not torch.equal(first, other_client)
    assert not torch.equal(first, next_round)


def test_output_only_model_stays_full_width_at_fractional_capacity():
    configs = Namespace(model="Linear", input_len=8, output_len=4)
    full_model = Linear(configs)
    plan = ptFL._pt_build_plan(
        configs.model,
        full_model,
        capacity=0.25,
        selector=lambda *_: pytest.fail(
            "output-only plan must not select hidden units"
        ),
    )

    assert plan.is_degenerate
    full_parameters = OrderedDict(full_model.named_parameters())
    extracted = ptFL._pt_extract_parameters(full_parameters, plan.manifest)
    rebuilt = ptFL._pt_build_client_model(configs, capacity=0.25)
    rebuilt.load_state_dict(extracted, strict=True)
    assert sum(p.numel() for p in rebuilt.parameters()) == sum(
        p.numel() for p in full_model.parameters()
    )


def test_selective_aggregation_averages_only_exact_updated_coordinates():
    server = ptFL.__new__(ptFL)
    server.public_model_params = OrderedDict(
        weight=torch.arange(9, dtype=torch.float32).reshape(3, 3)
    )
    server._pt_pending_manifests = {
        0: OrderedDict(weight=(torch.tensor([0, 1]), torch.tensor([0, 1]))),
        1: OrderedDict(weight=(torch.tensor([1, 2]), torch.tensor([1, 2]))),
    }
    server._commit_global = lambda new_params: setattr(
        server, "public_model_params", OrderedDict(new_params)
    )
    packages = OrderedDict(
        {
            0: {
                "regular_model_params": OrderedDict(
                    weight=torch.tensor([[10.0, 20.0], [30.0, 40.0]])
                )
            },
            1: {
                "regular_model_params": OrderedDict(
                    weight=torch.tensor([[50.0, 60.0], [70.0, 80.0]])
                )
            },
        }
    )

    server.aggregate_client_updates(packages=packages)

    assert torch.equal(
        server.public_model_params["weight"],
        torch.tensor([[10.0, 20.0, 2.0], [30.0, 45.0, 60.0], [6.0, 70.0, 80.0]]),
    )
    assert server._pt_pending_manifests == {}


def test_server_package_keeps_manifest_private_and_transmits_dense_submodel():
    configs = _recurrent_configs("LSTM")
    server = FedRolex.__new__(FedRolex)
    server.configs = configs
    server.model = LSTM(configs)
    server.public_model_params = OrderedDict(
        (name, parameter.detach().clone())
        for name, parameter in server.model.named_parameters()
    )
    server._pt_capacities = (0.5,)
    server._pt_pending_manifests = {}
    server._pt_warned_degenerate = False
    server.current_iter = 1

    package = server.package(client_id=7)

    assert package["__wire__"] == ("regular_model_params", "capacity")
    assert "manifest" not in package
    assert "active_indices" not in package
    assert 7 in server._pt_pending_manifests
    assert package["regular_model_params"]["cells.1.W_ii"].shape == (4, 4)
    assert package["regular_model_params"]["fc_pred.weight"].shape == (6, 4)
