import logging
import math
import random
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from models.GRU import GRU
from models.Linear import Linear
from models.LSTM import LSTM
from strategies.FedDropout import FedDropout
from strategies.FedLAGC import FedLAGC, FedLAGC_Client, FedLAGCShared
from strategies.FedOBD import REPR, FedOBD, FedOBDShared, FedOBD_Client
from strategies.FedPLT import FedPLTShared
from strategies.FedPMT import FedPMT
from strategies.FedRolex import FedRolex
from strategies.FLuID import FLuID, FLuIDShared, FLuID_Client
from strategies.HASA import HASA, HASA_Client, HASAShared
from strategies.ptFL import ptFL, ptFLLocalMetric, ptFLUpdate, ptFLUpdate_Client
from strategies.tFL import tFL, tFL_Client


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
def test_vertical_recurrent_plan_strict_load_and_forward(
    model_name: str, model_class: type[nn.Module]
) -> None:
    configs = _recurrent_configs(model_name)
    full_model = model_class(configs)
    selected = {
        "cells.0": torch.tensor([0, 2, 4, 6]),
        "cells.1": torch.tensor([1, 3, 5, 7]),
    }

    plan = ptFL._pt_build_recurrent_plan(
        model=full_model,
        capacity=0.5,
        selector=lambda group_name, full_width, retained: selected[group_name],
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
    submodel_parameters = ptFL._pt_extract_parameters(
        parameters=full_parameters,
        manifest=plan.manifest,
    )
    narrow_model = ptFL._pt_build_client_model(configs=configs, capacity=0.5)
    narrow_model.load_state_dict(state_dict=submodel_parameters, strict=True)

    assert narrow_model.hidden_size == 4
    assert narrow_model(
        torch.randn(2, configs.input_len, configs.input_channels)
    ).shape == (
        2,
        configs.output_len,
        configs.output_channels,
    )


def test_fedrolex_is_unit_stride_cyclic_and_client_independent() -> None:
    server = FedRolex.__new__(FedRolex)
    server.current_iter = 6

    expected = torch.tensor([6, 7, 0, 1])
    assert torch.equal(
        server._pt_select_indices(
            group_name="cells.0",
            full_width=8,
            retained=4,
            client_id=0,
        ),
        expected,
    )
    assert torch.equal(
        server._pt_select_indices(
            group_name="cells.1",
            full_width=8,
            retained=4,
            client_id=99,
        ),
        expected,
    )


def test_feddropout_is_seeded_unique_and_fresh_by_client_and_round() -> None:
    server = FedDropout.__new__(FedDropout)
    server.seed = 17
    server.times = 2
    server.current_iter = 3

    first = server._pt_select_indices(
        group_name="cells.0", full_width=64, retained=16, client_id=4
    )
    repeated = server._pt_select_indices(
        group_name="cells.0", full_width=64, retained=16, client_id=4
    )
    other_client = server._pt_select_indices(
        group_name="cells.0", full_width=64, retained=16, client_id=5
    )
    server.current_iter = 4
    next_round = server._pt_select_indices(
        group_name="cells.0", full_width=64, retained=16, client_id=4
    )

    assert torch.equal(first, repeated)
    assert first.unique().numel() == 16
    assert not torch.equal(first, other_client)
    assert not torch.equal(first, next_round)


def test_output_only_model_rejects_fractional_capacity() -> None:
    configs = Namespace(model="Linear", input_len=8, output_len=4)
    full_model = Linear(configs)
    selector = lambda *_: pytest.fail("output-only plan must not select hidden units")

    with pytest.raises(ValueError, match="no hidden-width axis"):
        ptFL._pt_build_plan(
            model_name=configs.model,
            model=full_model,
            capacity=0.25,
            selector=selector,
        )

    # Full capacity is still a legal (if trivial) pairing.
    plan = ptFL._pt_build_plan(
        model_name=configs.model,
        model=full_model,
        capacity=1.0,
        selector=selector,
    )
    assert plan.is_degenerate
    full_parameters = OrderedDict(full_model.named_parameters())
    extracted = ptFL._pt_extract_parameters(
        parameters=full_parameters,
        manifest=plan.manifest,
    )
    rebuilt = ptFL._pt_build_client_model(configs=configs, capacity=1.0)
    rebuilt.load_state_dict(state_dict=extracted, strict=True)
    assert sum(p.numel() for p in rebuilt.parameters()) == sum(
        p.numel() for p in full_model.parameters()
    )


def test_selective_aggregation_averages_only_exact_updated_coordinates() -> None:
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


def test_ptfl_shared_paper_math_and_aggregation_contracts(tmp_path: Path) -> None:
    assert FedPMT._fedpmt_parse_depths(raw="all", layer_count=4) == (1, 2, 3, 4)
    assert FedPMT._fedpmt_parse_depths(raw="1,3,4", layer_count=4) == (1, 3, 4)
    fedpmt = FedPMT.__new__(FedPMT)
    fedpmt._fedpmt_groups = ("fc1", "fc2", "fc3", "fc4")
    fedpmt._fedpmt_depths = (1, 3, 4)
    assert tuple(fedpmt._pt_update_spec(client_id=0)) == ("fc4",)
    assert tuple(fedpmt._pt_update_spec(client_id=1)) == ("fc2", "fc3", "fc4")
    assert tuple(fedpmt._pt_update_spec(client_id=2)) == (
        "fc1",
        "fc2",
        "fc3",
        "fc4",
    )

    assert torch.equal(
        FedLAGCShared._fedlagc_allocate_counts(
            mean_importance=torch.tensor([1.0, 3.0]),
            parameter_counts=torch.tensor([4, 4]),
            budget=4,
        ),
        torch.tensor([1, 3]),
    )
    assert FedLAGCShared._fedlagc_correction_active(
        current_iter=249, iterations=1000
    )
    assert not FedLAGCShared._fedlagc_correction_active(
        current_iter=250, iterations=1000
    )
    # h(t) = 1 for t < T / 4 exactly, so a non-divisible T keeps the last
    # round below the real-valued boundary (10 / 4 = 2.5 keeps t = 2).
    assert FedLAGCShared._fedlagc_correction_active(current_iter=2, iterations=10)
    assert not FedLAGCShared._fedlagc_correction_active(current_iter=3, iterations=10)

    sparse_model = nn.Linear(3, 2)
    sparse_parameters = OrderedDict(sparse_model.named_parameters())
    sparse_masks = OrderedDict(
        weight=torch.tensor(
            [[True, False, True], [False, True, False]], dtype=torch.bool
        ),
        bias=torch.ones(2, dtype=torch.bool),
    )
    sparse_state = FedLAGCShared._fedlagc_compress(
        parameters=sparse_parameters,
        masks=sparse_masks,
    )
    expanded_state, expanded_masks = FedLAGCShared._fedlagc_expand(
        model=sparse_model,
        sparse=sparse_state,
    )
    assert sparse_state["weight"][0].dtype == torch.int32
    for name, parameter in sparse_parameters.items():
        assert torch.equal(expanded_masks[name], sparse_masks[name])
        assert torch.equal(
            expanded_state[name][sparse_masks[name]],
            parameter.detach()[sparse_masks[name]],
        )
        assert not bool(expanded_state[name][~sparse_masks[name]].any())

    fedlagc_client = FedLAGC_Client.__new__(FedLAGC_Client)
    fedlagc_client.model = nn.Linear(2, 1, bias=False)
    fedlagc_client.model.weight.data.copy_(torch.tensor([[1.0, 0.0]]))
    fedlagc_client._pt_trainable_mask = OrderedDict(
        weight=torch.tensor([[True, False]])
    )
    fedlagc_client._fedlagc_parameters = dict(
        fedlagc_client.model.named_parameters()
    )
    fedlagc_client._fedlagc_threshold = {"weight": torch.tensor(0.5)}
    fedlagc_client._fedlagc_lambda = OrderedDict(
        weight=torch.tensor([[0.2, 0.3]])
    )
    fedlagc_client._fedlagc_use_correction = True
    corrected = fedlagc_client._pt_mask_gradient(
        name="weight", gradient=torch.ones(1, 2)
    )
    assert torch.allclose(corrected, torch.tensor([[1.2444445, 0.0]]))

    assert FedPLTShared._fedplt_layer_ratios(
        parameter_counts=(8, 2), training_ratio=0.5
    ) == pytest.approx((0.375, 1.0))
    assert FedPLTShared._fedplt_rotating_blocks(
        retained_counts=(2, 1, 2), sublayers=4
    ) == ((0, 1), (2,), (3, 0))

    counts = torch.tensor([[9, 1], [1, 9], [5, 5]])
    scores = HASAShared._hasa_jsd_scores(counts=counts, alpha=1.0)
    widths = HASAShared._hasa_allocate(
        scores=scores,
        sample_sizes=torch.ones(3),
        caps=torch.full((3,), 0.8),
        minimum=0.2,
        maximum=0.8,
        budget=0.5,
    )

    assert torch.allclose(widths, torch.tensor([0.65, 0.65, 0.2], dtype=widths.dtype))

    server = HASA.__new__(HASA)
    server.hasa_aggregation = "full"
    server.public_model_params = OrderedDict(weight=torch.tensor([10.0, 20.0]))
    server._pt_pending_manifests = {
        0: OrderedDict(weight=(torch.tensor([0]),)),
        1: OrderedDict(weight=(torch.tensor([1]),)),
    }
    server._hasa_sample_sizes = {0: 1.0, 1: 3.0}
    server._commit_global = lambda new_params: setattr(
        server, "public_model_params", OrderedDict(new_params)
    )
    server.aggregate_client_updates(
        packages=OrderedDict(
            {
                0: {"regular_model_params": OrderedDict(weight=torch.tensor([14.0]))},
                1: {"regular_model_params": OrderedDict(weight=torch.tensor([30.0]))},
            }
        )
    )

    assert torch.equal(server.public_model_params["weight"], torch.tensor([11.0, 27.5]))

    profile_path = tmp_path / "hasa-profile.npz"
    np.savez(profile_path, x=np.zeros((4, 1), dtype=np.float32))
    client = HASA_Client.__new__(HASA_Client)
    client.hasa_count_bins = "n_neg,n_zero,n_pos"
    # Columns are read in sorted order so every client builds the same support.
    client.stats = {
        "B": {"n_neg": 0, "n_zero": 3, "n_pos": 1, "count": 4},
        "A": {"n_neg": 2, "n_zero": 5, "n_pos": 7, "count": 14},
    }
    client._load_private = lambda client_id: setattr(
        client, "train_file", str(profile_path)
    )
    profile = client.hasa_profile(client_id=2)

    assert profile["__wire__"] == ("value_counts", "sample_size")
    assert profile["value_counts"].dtype == torch.int32
    assert profile["value_counts"].tolist() == [2, 5, 7, 0, 3, 1]
    assert profile["sample_size"] == 4

    client.stats = {"A": {"n_neg": 2, "n_zero": 5}}
    with pytest.raises(KeyError, match="n_pos"):
        client.hasa_profile(client_id=2)


def test_server_package_keeps_manifest_private_and_transmits_dense_submodel() -> None:
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
    server._pt_local_metrics = ("personalization",)
    server._pt_last_submodel = {}
    server.client_scheduler_states = {7: {"last_epoch": 1}}
    server.current_iter = 1

    package = server.package(client_id=7)

    assert package["__wire__"] == ("regular_model_params", "capacity")
    assert "depth_layers" not in package
    assert package["scheduler_state"] == {"last_epoch": 1}
    assert "manifest" not in package
    assert "active_indices" not in package
    assert 7 in server._pt_pending_manifests
    assert package["regular_model_params"]["cells.1.W_ii"].shape == (4, 4)
    assert package["regular_model_params"]["fc_pred.weight"].shape == (6, 4)


def test_masked_update_transport_sends_and_aggregates_only_selected_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = ptFLUpdate.__new__(ptFLUpdate)
    server.model = nn.Linear(2, 2)
    server.public_model_params = OrderedDict(
        weight=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        bias=torch.tensor([5.0, 6.0]),
    )
    server._pt_pending_update_masks = {}
    server._pt_update_spec = lambda client_id: OrderedDict({"": (0,)})
    server.clients_personal_model_params = {3: {}}
    server.client_optimizer_states = {3: {}}
    server.client_scheduler_states = {3: {}}
    server.current_iter = 2
    server._pt_weighted_aggregation = False
    server._commit_global = lambda new_params: setattr(
        server, "public_model_params", OrderedDict(new_params)
    )

    downlink = server.package(client_id=3)
    assert downlink["__wire__"] == ("regular_model_params", "trainable_spec")

    server.aggregate_client_updates(
        packages=OrderedDict(
            {
                3: {
                    "model_params_diff": OrderedDict(
                        weight=torch.tensor([0.25, 0.5]),
                        bias=torch.tensor([1.0]),
                    )
                }
            }
        )
    )

    assert torch.equal(
        server.public_model_params["weight"],
        torch.tensor([[0.75, 1.5], [3.0, 4.0]]),
    )
    assert torch.equal(server.public_model_params["bias"], torch.tensor([4.0, 6.0]))

    client = ptFLUpdate_Client.__new__(ptFLUpdate_Client)
    client.model = nn.Linear(2, 2)
    client.optimizer = torch.optim.SGD(
        client.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.1
    )
    for parameter in client.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    client.optimizer.step()
    client._pt_trainable_mask = OrderedDict(
        weight=torch.tensor([[True, True], [False, False]]),
        bias=torch.tensor([True, False]),
    )
    client._pt_install_optimizer_guard()
    selected_weight = client.model.weight[0].detach().clone()
    frozen_weight = client.model.weight[1].detach().clone()
    frozen_bias = client.model.bias[1].detach().clone()

    for parameter in client.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    client.optimizer.step()

    assert torch.equal(client.model.weight[1], frozen_weight)
    assert torch.equal(client.model.bias[1], frozen_bias)
    assert not torch.equal(client.model.weight[0], selected_weight)
    assert torch.equal(
        client.optimizer.state[client.model.weight]["momentum_buffer"][1],
        torch.zeros(2),
    )
    client._pt_optimizer_handle.remove()

    dispatch_client = ptFLUpdate_Client.__new__(ptFLUpdate_Client)
    dispatch_client.model = nn.Linear(2, 2)
    expected_mask = OrderedDict(
        weight=torch.tensor([[True, True], [False, False]]),
        bias=torch.tensor([False, False]),
    )
    resolved_packages = []
    dispatch_client._pt_resolve_update_mask = lambda package: (
        resolved_packages.append(package) or expected_mask
    )
    dispatch_client._pt_install_optimizer_guard = lambda: None
    monkeypatch.setattr(tFL_Client, "set_parameters", lambda self, package: None)

    marker = {"mask": "strategy-specific"}
    dispatch_client.set_parameters(package=marker)

    assert resolved_packages == [marker]
    assert dispatch_client._pt_trainable_mask is expected_mask
    for handle in dispatch_client._pt_gradient_handles:
        handle.remove()


def _configs() -> Namespace:
    return _recurrent_configs("LSTM")


def _params(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(
        (name, parameter.detach().clone())
        for name, parameter in model.named_parameters()
    )


# --------------------------------------------------------------------------
# FLuID
# --------------------------------------------------------------------------


def test_fluid_neuron_score_is_the_tightest_relative_change_bound() -> None:
    model = LSTM(_configs())
    previous = _params(model)
    for tensor in previous.values():
        tensor.fill_(1.0)
    current = OrderedDict((name, tensor.clone()) for name, tensor in previous.items())
    # Unit 3 of layer 0 moves one input-gate weight by 40%; unit 5 by 10%.
    current["cells.0.W_ii"][3, 0] = 1.4
    current["cells.0.W_ii"][5, 0] = 1.1

    scores = FLuIDShared._fluid_neuron_scores(
        model=model, previous=previous, current=current
    )

    layer0 = scores["cells.0"]
    assert layer0[3].item() == pytest.approx(0.4)
    assert layer0[5].item() == pytest.approx(0.1)
    assert layer0[[0, 1, 2, 4, 6, 7]].abs().max().item() == pytest.approx(0.0)
    assert torch.allclose(scores["cells.1"], torch.zeros(8, dtype=torch.float64))


def test_fluid_neuron_score_spans_every_gate_block_of_a_unit() -> None:
    model = LSTM(_configs())
    previous = _params(model)
    for tensor in previous.values():
        tensor.fill_(2.0)
    current = OrderedDict((name, tensor.clone()) for name, tensor in previous.items())
    # The recurrent forget-gate row and the bias entry of unit 2 both move; the
    # neuron score must be the larger of the two.
    current["cells.1.W_hf"][2, 1] = 2.5
    current["cells.1.b_o"][2] = 3.0

    scores = FLuIDShared._fluid_neuron_scores(
        model=model, previous=previous, current=current
    )

    assert scores["cells.1"][2].item() == pytest.approx(0.5)


def test_fluid_zero_valued_broadcast_weight_admits_no_threshold() -> None:
    model = LSTM(_configs())
    previous = _params(model)
    for tensor in previous.values():
        tensor.fill_(1.0)
    previous["cells.0.b_i"].fill_(0.0)
    current = OrderedDict((name, tensor.clone()) for name, tensor in previous.items())
    current["cells.0.b_i"][1] = 5.0

    scores = FLuIDShared._fluid_neuron_scores(
        model=model, previous=previous, current=current
    )

    # ``|dw| <= th * |w|`` can never hold for a moved zero weight, so unit 1 is
    # never invariant; the untouched zero weights bound nothing.
    assert scores["cells.0"][1].item() == math.inf
    assert scores["cells.0"][[0, 2, 3, 4, 5, 6, 7]].max().item() == pytest.approx(0.0)


def _fluid_server(model=None, threshold: float = 30.0) -> FLuID:
    server = FLuID.__new__(FLuID)
    server.model = LSTM(_configs()) if model is None else model
    server.fluid_initial_threshold = threshold
    server.fluid_threshold_step = 0.1
    server.fluid_majority = 0.5
    server._fluid_straggler = {}
    server._fluid_just_updated = False
    server._fluid_p_val = 0.95
    server._fluid_thresholds = {}
    server._fluid_unchanged = {}
    server._fluid_def_drop = {}
    server._fluid_prev_drop = {}
    server._fluid_rng = random.Random(0)
    server.current_iter = 0
    return server


def test_fluid_only_a_detected_straggler_gets_a_submodel() -> None:
    server = _fluid_server()
    server._fluid_p_val = 0.75

    # No straggler has been detected yet, so nobody is cut.
    server.current_iter = 5
    assert server._pt_capacity_for_client(client_id=1) == 1.0

    server._fluid_straggler = {1: 9.0}
    # The official strategy starts dropping only after round two.
    server.current_iter = 1
    assert server._pt_capacity_for_client(client_id=1) == 1.0
    server.current_iter = 2
    assert server._pt_capacity_for_client(client_id=1) == 0.75
    assert server._pt_capacity_for_client(client_id=0) == 1.0


def test_fluid_straggler_and_p_val_come_from_measured_durations() -> None:
    server = _fluid_server()
    server.current_iter = 1
    packages = OrderedDict(
        (cid, {"duration": duration}) for cid, duration in ((0, 1.0), (1, 1.6), (2, 2.0))
    )

    server._fluid_update_straggler(packages=packages)

    # Client 2 is slowest; the ladder reads the next-slowest ratio 1.6 / 2.0.
    assert set(server._fluid_straggler) == {2}
    assert server._fluid_p_val == 0.85


def test_fluid_missing_duration_is_an_error() -> None:
    server = _fluid_server()
    server.current_iter = 1
    with pytest.raises(KeyError, match="measured duration"):
        server._fluid_update_straggler(
            packages=OrderedDict({0: {"duration": 1.0}, 1: {}})
        )


def test_fluid_p_val_ladder_matches_the_reference_buckets() -> None:
    server = _fluid_server()
    for ratio, expected in (
        (0.95, 0.95),
        (0.90, 0.95),
        (0.85, 0.85),
        (0.75, 0.75),
        (0.65, 0.65),
        (0.10, 0.5),
    ):
        server._fluid_set_p_val(ratio)
        assert server._fluid_p_val == expected


def test_fluid_invariance_needs_a_strict_majority_of_clients() -> None:
    server = _fluid_server()
    server._fluid_thresholds = {"cells.0": 1.0, "cells.1": 1.0}
    server._fluid_p_val = 1.0  # nothing to drop, so no threshold climbing
    parameters = dict(server.model.named_parameters())
    axes = FLuIDShared._fluid_unit_axes(model=server.model)

    def source(quiet):
        # A unit is quiet only when every axis tagged to it is quiet, so the
        # change has to be written on each tagged axis, not just axis 0.
        out = {
            name: torch.full_like(parameters[name], 9.0)
            for _hidden, tagged in axes.values()
            for name, _axis in tagged
        }
        for _hidden, tagged in axes.values():
            for name, axis in tagged:
                for unit in quiet:
                    out[name].select(axis, unit).fill_(0.5)
        return out

    # Unit 0 is quiet on 3 of 4 clients, unit 1 on exactly 2 of 4 -- a tie is
    # not a majority, so only unit 0 qualifies.
    sources = [source({0, 1}), source({0, 1}), source({0}), source(set())]
    server._fluid_find_stable(sources=sources)

    unchanged = server._fluid_unchanged["cells.0"]
    assert 0 in unchanged
    assert 1 not in unchanged


def test_fluid_definite_drops_are_units_invariant_across_two_cuts() -> None:
    server = _fluid_server()
    server._fluid_thresholds = {"cells.0": 1e9, "cells.1": 1e9}
    server._fluid_prev_drop = {"cells.0": [2, 5]}
    sources = [
        {
            name: torch.zeros_like(parameter)
            for name, parameter in server.model.named_parameters()
        }
    ]
    server._fluid_find_stable(sources=sources)

    assert server._fluid_def_drop["cells.0"] == [2, 5]


def test_fluid_drop_priority_prefers_definite_then_invariant() -> None:
    server = _fluid_server()
    server._fluid_def_drop = {"cells.0": [1, 4]}
    server._fluid_unchanged = {"cells.0": [1, 3, 4, 6]}

    keep = server._pt_select_indices(
        group_name="cells.0", full_width=8, retained=6, client_id=1
    )
    # Two units must go and both definite drops qualify, so exactly those go.
    assert keep.tolist() == [0, 2, 3, 5, 6, 7]

    keep = server._pt_select_indices(
        group_name="cells.0", full_width=8, retained=4, client_id=1
    )
    # Four must go: every one of them comes from the invariant set.
    assert set(keep.tolist()).isdisjoint({1, 4})
    assert set(range(8)) - set(keep.tolist()) <= {1, 3, 4, 6}


def test_fluid_each_layer_seeds_its_own_threshold() -> None:
    server = _fluid_server()
    scores = [
        OrderedDict(
            {
                "cells.0": torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64),
                "cells.1": torch.tensor([0.6, 0.8, 1.0], dtype=torch.float64),
            }
        )
    ]

    server.current_iter = 1  # round 2 sets each layer outright
    server._fluid_find_min(scores=scores)
    assert server._fluid_thresholds == pytest.approx({"cells.0": 0.2, "cells.1": 0.6})

    scores[0]["cells.0"] = torch.tensor([0.4, 0.5, 0.9], dtype=torch.float64)
    server.current_iter = 2  # round 3 averages with what round 2 set
    server._fluid_find_min(scores=scores)
    assert server._fluid_thresholds["cells.0"] == pytest.approx(0.3)
    assert server._fluid_thresholds["cells.1"] == pytest.approx(0.6)

    server.current_iter = 5  # seeding only happens in the initial few rounds
    server._fluid_find_min(scores=scores)
    assert server._fluid_thresholds["cells.0"] == pytest.approx(0.3)


def test_fluid_threshold_climbs_until_enough_units_are_invariant() -> None:
    server = _fluid_server(threshold=0.01)
    server._fluid_p_val = 0.5  # 8 hidden units -> 4 must be droppable
    parameters = dict(server.model.named_parameters())
    axes = FLuIDShared._fluid_unit_axes(model=server.model)
    source = {
        name: torch.full_like(parameters[name], 0.02)
        for _hidden, tagged in axes.values()
        for name, _axis in tagged
    }

    server._fluid_find_stable(sources=[source])

    # 0.01 admits nothing; the first raise to 0.011 still does not reach 0.02,
    # so the search climbs until it clears every unit at once.
    assert len(server._fluid_unchanged["cells.0"]) >= 4
    assert server._fluid_thresholds["cells.0"] >= 0.02
    assert server._fluid_thresholds["cells.0"] < 0.02 * 1.1


def test_fluid_threshold_search_is_bounded_on_unreachable_units() -> None:
    server = _fluid_server(threshold=1.0)
    server._fluid_p_val = 0.5
    parameters = dict(server.model.named_parameters())
    axes = FLuIDShared._fluid_unit_axes(model=server.model)
    # A weight that moved off zero scores an infinite relative change, so no
    # finite threshold ever admits it; the search must still terminate.
    source = {
        name: torch.full_like(parameters[name], float("inf"))
        for _hidden, tagged in axes.values()
        for name, _axis in tagged
    }

    server._fluid_find_stable(sources=[source])

    assert server._fluid_unchanged["cells.0"] == []


def test_fluid_aggregation_is_sample_weighted() -> None:
    # ``aggregate_drop`` divides each coordinate by the examples that trained it.
    assert FLuID._pt_send_score is True
    assert FLuID_Client._pt_send_score is True


def test_fluid_client_reports_its_training_duration() -> None:
    client = FLuID_Client.__new__(FLuID_Client)
    client.set_parameters = lambda package: None
    client.fit = lambda: None
    client.package = lambda: {
        "regular_model_params": OrderedDict(),
        "__wire__": ("regular_model_params", "score"),
    }

    out = FLuID_Client.train(client, package={})

    assert out["duration"] >= 0.0
    assert out["__wire__"] == ("regular_model_params", "score", "duration")


def test_fluid_indices_rebuild_a_loadable_narrow_model() -> None:
    configs = _configs()
    model = LSTM(configs)
    server = _fluid_server(model=model)
    server._fluid_unchanged = {
        "cells.0": list(range(8)),
        "cells.1": list(range(8)),
    }
    plan = FLuID._pt_build_recurrent_plan(
        model=model,
        capacity=0.5,
        selector=lambda group_name, full_width, retained: server._pt_select_indices(
            group_name=group_name,
            full_width=full_width,
            retained=retained,
            client_id=1,
        ),
    )
    submodel = FLuID._pt_extract_parameters(
        parameters=_params(model), manifest=plan.manifest
    )
    narrow = FLuID._pt_build_client_model(configs=configs, capacity=0.5)
    narrow.load_state_dict(state_dict=submodel, strict=True)
    narrow(torch.zeros(2, configs.input_len, configs.input_channels))


# --------------------------------------------------------------------------
# FedOBD
# --------------------------------------------------------------------------


def test_obd_blocks_are_the_repeated_cells_and_the_head() -> None:
    model = LSTM(_configs())
    blocks = FedOBDShared._obd_blocks(model=model)

    assert list(blocks) == ["cells.0", "cells.1", "fc_pred"]
    covered = [name for names in blocks.values() for name in names]
    assert sorted(covered) == sorted(name for name, _ in model.named_parameters())
    assert blocks["fc_pred"] == ("fc_pred.weight", "fc_pred.bias")


def test_obd_mean_block_difference_is_the_norm_over_the_parameter_count() -> None:
    previous = OrderedDict({"a": torch.zeros(4), "b": torch.zeros(4)})
    current = OrderedDict({"a": torch.full((4,), 3.0), "b": torch.full((4,), 4.0)})

    value = FedOBDShared._obd_mean_block_difference(
        previous=previous, current=current, names=("a", "b")
    )

    assert value == pytest.approx(math.sqrt(4 * 9 + 4 * 16) / 8)


def test_obd_retains_high_difference_blocks_within_the_budget() -> None:
    blocks = OrderedDict(
        {"big": ("big",), "mid": ("mid",), "small": ("small",)}
    )
    previous = OrderedDict(
        {"big": torch.zeros(60), "mid": torch.zeros(30), "small": torch.zeros(10)}
    )
    current = OrderedDict(
        {
            "big": torch.full((60,), 0.1),
            "mid": torch.full((30,), 1.0),
            "small": torch.full((10,), 0.5),
        }
    )

    retained = FedOBDShared._obd_retained_blocks(
        blocks=blocks, previous=previous, current=current, dropout_rate=0.3
    )

    # Budget is 70 of 100 parameters. "mid" has the largest MBD and is taken
    # first; "big" (60) no longer fits, so the loop continues and still admits
    # "small" (10) rather than stopping.
    assert retained == ("mid", "small")


def test_obd_dropout_rate_zero_retains_every_block() -> None:
    model = LSTM(_configs())
    previous = _params(model)
    current = OrderedDict(
        (name, tensor + 0.01) for name, tensor in previous.items()
    )
    blocks = FedOBDShared._obd_blocks(model=model)

    retained = FedOBDShared._obd_retained_blocks(
        blocks=blocks, previous=previous, current=current, dropout_rate=0.0
    )

    assert set(retained) == set(blocks)


def test_adq_level_count_matches_the_closed_form_solution() -> None:
    tensor = torch.tensor([-1.0, 3.0])
    weight = 0.001
    reconstructed, levels = FedOBDShared._adq(tensor=tensor, weight=weight)

    # offset = -(3 + -1)/2 = -1, so v' = [-2, 2] and d = 2.
    expected = int(math.floor(math.sqrt(math.log(4.0) * REPR / weight * 2.0)))
    assert levels == expected
    assert torch.allclose(reconstructed, tensor, atol=1e-4)


def test_adq_error_is_bounded_by_half_a_quantization_step() -> None:
    torch.manual_seed(0)
    tensor = torch.randn(500) * 0.05
    reconstructed, levels = FedOBDShared._adq(tensor=tensor, weight=0.001)

    shifted = tensor + (-(tensor.max() + tensor.min()) / 2)
    normalizer = float(shifted.abs().max())
    assert (reconstructed - tensor).abs().max().item() <= normalizer / levels / 2 + 1e-9


def test_adq_carries_a_constant_tensor_exactly() -> None:
    tensor = torch.full((6,), -0.75)
    reconstructed, _ = FedOBDShared._adq(tensor=tensor, weight=0.001)
    assert torch.allclose(reconstructed, tensor)


def test_nnadq_wire_size_counts_levels_signs_and_scalars() -> None:
    tensors = OrderedDict({"a": torch.randn(100), "b": torch.randn(20)})
    _, megabytes = FedOBDShared._nnadq(tensors=tensors, weight=0.001)

    expected_bits = 0.0
    for tensor in tensors.values():
        _, levels = FedOBDShared._adq(tensor=tensor, weight=0.001)
        expected_bits += tensor.numel() * (math.ceil(math.log2(levels + 1)) + 1)
        expected_bits += 3 * REPR
    assert megabytes == pytest.approx(expected_bits / 8 / (1024**2))


def _obd_server(iterations: int = 12, stage2: int = 2) -> FedOBD:
    server = FedOBD.__new__(FedOBD)
    server.fedobd_stage2_epochs = stage2
    server.fedobd_epochs = 5
    server.fedobd_dropout_rate = 0.3
    server.fedobd_weight = 0.001
    server._obd_stage1_rounds = iterations
    server.iterations = iterations + stage2
    return server


def test_obd_stage_boundary_follows_the_round_budget() -> None:
    server = _obd_server(iterations=12, stage2=2)
    # The paper runs R stage-1 rounds and then E2 stage-2 epochs on top.
    assert server.iterations == 14
    server.current_iter = 11
    assert server._obd_stage() == 1
    server.current_iter = 12
    assert server._obd_stage() == 2


def test_obd_second_stage_disables_dropout_and_trains_one_epoch() -> None:
    server = _obd_server(iterations=4, stage2=1)
    server.public_model_params = OrderedDict({"w": torch.randn(10)})
    server.clients_personal_model_params = {0: {}}
    server.client_optimizer_states = {0: {}}
    server.client_scheduler_states = {0: {}}
    server._obd_broadcast = OrderedDict()
    server._obd_broadcast_iter = None
    server._obd_downlink_mb = 0.0

    server.current_iter = 0
    stage1 = server.package(client_id=0)
    assert stage1["fedobd_epochs"] == 5
    assert stage1["fedobd_dropout_rate"] == pytest.approx(0.3)

    # Stage 1 keeps all four configured rounds; stage 2 starts after them.
    server.current_iter = 4
    stage2 = server.package(client_id=0)
    assert stage2["fedobd_epochs"] == 1
    assert stage2["fedobd_dropout_rate"] == pytest.approx(0.0)


def test_obd_aggregation_keeps_dropped_blocks_at_the_broadcast_value() -> None:
    server = _obd_server()
    server._obd_blocks_map = OrderedDict(
        {"kept": ("kept.w",), "dropped": ("dropped.w",)}
    )
    server._obd_broadcast = OrderedDict(
        {"kept.w": torch.zeros(4), "dropped.w": torch.full((4,), 7.0)}
    )
    committed: dict[str, torch.Tensor] = {}
    server._commit_global = lambda new_params: committed.update(new_params)

    packages = OrderedDict(
        {
            0: {
                "fedobd_retained": ("kept",),
                "fedobd_update": {"kept.w": torch.full((4,), 2.0)},
                "score": 3.0,
            },
            1: {
                "fedobd_retained": ("kept",),
                "fedobd_update": {"kept.w": torch.full((4,), 6.0)},
                "score": 1.0,
            },
        }
    )
    server.aggregate_client_updates(packages=packages)

    # Sample-weighted mean of 2 and 6 with weights 3 and 1.
    assert torch.allclose(committed["kept.w"], torch.full((4,), 3.0))
    assert torch.allclose(committed["dropped.w"], torch.full((4,), 7.0))


def test_obd_aggregation_rejects_an_unknown_block() -> None:
    server = _obd_server()
    server._obd_blocks_map = OrderedDict({"kept": ("kept.w",)})
    server._obd_broadcast = OrderedDict({"kept.w": torch.zeros(2)})
    packages = OrderedDict(
        {0: {"fedobd_retained": ("ghost",), "fedobd_update": {}, "score": 1.0}}
    )
    with pytest.raises(KeyError):
        server.aggregate_client_updates(packages=packages)


def test_obd_uplink_is_measured_from_the_quantized_payload() -> None:
    server = _obd_server()
    server._obd_downlink_mb = 0.25
    server.selected_clients = [0, 1]
    packages = OrderedDict(
        {0: {"fedobd_uplink_mb": 0.1}, 1: {"fedobd_uplink_mb": 0.2}}
    )
    uplink, downlink = server._compute_send_mb(packages=packages)

    assert uplink == {0: 0.1, 1: 0.2}
    assert downlink == pytest.approx(0.5)


def test_obd_client_uploads_only_retained_block_differences(monkeypatch) -> None:
    configs = _configs()
    model = LSTM(configs)
    client = FedOBD_Client.__new__(FedOBD_Client)
    client.model = model
    client._obd_blocks_map = FedOBDShared._obd_blocks(model=model)
    client._obd_dropout_rate = 0.5
    client._obd_weight = 0.001
    client._obd_previous = _params(model)
    current = OrderedDict(
        (name, tensor.clone()) for name, tensor in client._obd_previous.items()
    )
    # Only the head moves, so it must win the ranking and fit the budget.
    current["fc_pred.weight"] += 1.0
    monkeypatch.setattr(
        tFL_Client,
        "package",
        lambda self: {
            "regular_model_params": current,
            "__wire__": ("regular_model_params", "score"),
        },
    )
    result = client.package()

    assert "fc_pred" in result["fedobd_retained"]
    assert set(result["fedobd_update"]) <= set(
        name
        for block in result["fedobd_retained"]
        for name in client._obd_blocks_map[block]
    )
    assert result["regular_model_params"] == {}
    assert result["__wire__"] == ("fedobd_update", "fedobd_retained")
    assert result["fedobd_uplink_mb"] > 0.0


class _RecordingTrainer:
    """Capture the parameters each client is evaluated against."""

    def __init__(self) -> None:
        self.seen: dict[int, OrderedDict[str, torch.Tensor]] = {}
        self.calls: list[list[int]] = []
        self.maps: list[dict[int, OrderedDict[str, torch.Tensor]]] = []

    def evaluate(self, ids, global_params, dataset_type, current_iter):
        del dataset_type, current_iter
        self.calls.append(list(ids))
        for client_id in ids:
            self.seen[client_id] = global_params
        return [float(client_id) for client_id in ids]

    def evaluate_personalized(
        self, ids, global_params, personal_map, dataset_type, current_iter
    ):
        del global_params, dataset_type, current_iter
        self.calls.append(list(ids))
        self.maps.append(dict(personal_map))
        for client_id in ids:
            self.seen[client_id] = personal_map[client_id]
        return [float(client_id) for client_id in ids]


def _hasa_local_metric_server() -> HASA:
    configs = _recurrent_configs("LSTM")
    server = HASA.__new__(HASA)
    server.configs = configs
    server.model = LSTM(configs)
    server.public_model_params = OrderedDict(
        (name, parameter.detach().clone())
        for name, parameter in server.model.named_parameters()
    )
    server._hasa_capacities = {0: 0.25, 1: 0.75}
    server.current_iter = 3
    server.trainer = _RecordingTrainer()
    return server


def test_local_metric_reports_the_resource_level_and_the_last_sent_subnet() -> None:
    server = _hasa_local_metric_server()
    server._hasa_capacities = {0: 0.25, 1: 0.75, 2: 0.25}
    server.num_clients = 3
    server.is_new = [False, False, False]
    server.metrics = {}
    server.logger = logging.getLogger("test_local_metric_record")
    server.ptfl_local_metrics = "resourcelevel,personalization"
    server._pt_init_local_metric()
    server._pt_pending_manifests = {}
    server.client_scheduler_states = {0: {}, 1: {}, 2: {}}
    server.current_iter = 0

    # Only clients 0 and 1 are selected this round; client 2 gets nothing.
    for client_id in (0, 1):
        server.package(client_id=client_id)
    server._post_eval_hook(dataset_type="test")

    resource_level, personalization = server.trainer.maps
    # floor(0.25 * 8) = 2 and floor(0.75 * 8) = 6 hidden units. Every client
    # has a resource level, selected this round or not.
    for scored in (resource_level[0], resource_level[2], personalization[0]):
        assert scored["cells.0.W_hi"].shape == (2, 2)
    assert resource_level[1]["cells.0.W_hi"].shape == (6, 6)
    assert personalization[1]["cells.0.W_hi"].shape == (6, 6)
    # Client 2 was never sent a submodel, so the last-sent metric falls back to
    # the global model rather than to one this round would have assigned it.
    assert personalization[2] == {}
    # Two batched calls; the Trainer fans out per client within each.
    assert server.trainer.calls == [[0, 1, 2], [0, 1, 2]]
    assert server.metrics["resourcelevel_avg_test_loss"] == [1.0]
    assert server.metrics["personalization_avg_test_loss"] == [1.0]


def test_generalization_still_reports_the_full_global_model() -> None:
    server = _hasa_local_metric_server()
    server.num_clients = 2
    server.is_new = [False, False]
    server.metrics = {"generalization_avg_test_loss": []}
    server.logger = logging.getLogger("test_generalization")
    server._best_global_loss = float("inf")
    server._round_client_data = {}

    tFL.evaluate_generalization(server, dataset_type="test")

    # One call for both clients, at full width -- the global metric never sees
    # a submodel.
    assert server.trainer.seen[0] is server.public_model_params
    assert server.trainer.seen[1] is server.public_model_params


def test_fedlagc_records_its_resource_level_submodel_when_packaging() -> None:
    server = FedLAGC.__new__(FedLAGC)
    # A single layer would be entirely critical (input + output), leaving
    # nothing prunable; the middle layer is what the capacity actually cuts.
    server.model = nn.Sequential(
        nn.Linear(4, 8, bias=False),
        nn.Linear(8, 8, bias=False),
        nn.Linear(8, 4, bias=False),
    )
    server.public_model_params = OrderedDict(
        (name, torch.ones_like(parameter))
        for name, parameter in server.model.named_parameters()
    )
    # Below 0.75 the critical input and output layers alone exhaust the budget.
    server._fedlagc_capacities = (1.0, 0.75)
    server._pt_pending_update_masks = {}
    server._pt_local_metrics = ("resourcelevel", "personalization")
    server._pt_last_submodel = {}
    server.clients_personal_model_params = {0: {}, 1: {}}
    server.client_optimizer_states = {0: {}, 1: {}}
    server.client_scheduler_states = {0: {}, 1: {}}
    server.current_iter = 0
    server.iterations = 4

    for client_id in (0, 1):
        server.package(client_id=client_id)

    full = server._pt_last_submodel[0]["1.weight"]
    half = server._pt_last_submodel[1]["1.weight"]
    # theta * M: the restricted client sees zeros outside its own mask.
    assert bool((full != 0).all())
    assert int((half != 0).sum()) < full.numel()
    assert bool(((half == 0) | (half == full)).all())
    # The correction payload keeps its own slot, untouched by the record.
    assert "fedlagc_correction" not in server._pt_last_submodel[0]
    # The resource-level metric rebuilds the same theta * M from whatever the
    # global model holds now, for any client, selected this round or not.
    assert bool(
        (server._pt_resource_level_params(client_id=1)["1.weight"] == half).all()
    )


def test_fedlagc_layer_importance_counts_the_retained_bias() -> None:
    model = nn.Sequential(
        nn.Linear(4, 8, bias=False),
        nn.Linear(8, 8),
        nn.Linear(8, 4, bias=False),
    )
    with torch.no_grad():
        model[1].weight.fill_(1.0)
        # A large bias lifts layer 1's mean only if the bias is counted at all.
        model[1].bias.fill_(100.0)

    parameters = dict(model.named_parameters())
    _critical, prunable = FedLAGC._fedlagc_layout(model=model)
    assert set(prunable) == {"1"}
    weights_only = parameters["1.weight"].abs().mean().item()
    with_bias = (
        torch.cat(
            [parameters["1.weight"].flatten(), parameters["1.bias"].flatten()]
        )
        .abs()
        .mean()
        .item()
    )
    assert with_bias > weights_only

    # The paper averages over "the total number of parameters in the l-th
    # layer", so the retained bias belongs in S_l even though it is never cut.
    mask = FedLAGC._fedlagc_mask(model=model, capacity=0.75)
    assert bool(mask["1.bias"].all())
    assert not bool(mask["1.weight"].all())


def test_local_metrics_option_selects_which_passes_run() -> None:
    parse = ptFLLocalMetric._pt_parse_local_metrics
    assert parse(raw="personalization") == ("personalization",)
    # Reported in a fixed order however they were listed.
    assert parse(raw="personalization,resourcelevel") == (
        "resourcelevel",
        "personalization",
    )
    assert parse(raw="none") == ()
    assert parse(raw="") == ()
    with pytest.raises(ValueError, match="unknown ptfl_local_metrics"):
        parse(raw="resourcelevel,typo")

    server = _hasa_local_metric_server()
    server.num_clients = 2
    server.is_new = [False, False]
    server.metrics = {}
    server.logger = logging.getLogger("test_local_metric_off")
    server.ptfl_local_metrics = "none"
    server._pt_init_local_metric()

    server._post_eval_hook(dataset_type="test")

    # No metric keys registered and no evaluation pass issued.
    assert server.metrics == {}
    assert server.trainer.calls == []
