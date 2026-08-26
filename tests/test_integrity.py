"""General paper-integrity regression checks."""

import ast
import math
from collections import OrderedDict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from strategies.DeComFL import DeComFL
from strategies.FedAdam import FedAdam
from strategies.FedAvgM import FedAvgM
from strategies.FedLUAR import FedLUAR
from strategies.FedNova import FedNova_Client
from strategies.FedPAQ import FedPAQ, FedPAQ_Client, FedPAQShared
from strategies.FedSPA import FedSPA, FedSPA_Client
from strategies.FedYogi import FedYogi
from strategies.QATFL import QATFL, QATFL_Client, QATFLShared
from strategies.qFL import qFL, qFL_Client
from strategies.SCAFFOLD import SCAFFOLD, SCAFFOLD_Client
from strategies.tFL import tFL, tFL_Client


def _server(strategy, size=1):
    server = object.__new__(strategy)
    server.model = torch.nn.Linear(size, 1, bias=False)
    server.model.weight.data.zero_()
    server.public_model_params = OrderedDict(
        (name, value.detach().clone())
        for name, value in server.model.named_parameters()
    )
    return server


def test_package_extraction_is_single_pass() -> None:
    class CountingPackages(OrderedDict):
        values_calls = 0

        def values(self) -> Any:
            self.values_calls += 1
            return super().values()

    packages = CountingPackages(
        {
            0: {"regular_model_params": {"weight": torch.tensor([1.0])}, "score": 2},
            1: {"regular_model_params": {"weight": torch.tensor([3.0])}, "score": 4},
        }
    )
    models, scores = tFL.extract_models_and_scores(packages=packages)
    assert packages.values_calls == 1
    assert [model["weight"].item() for model in models] == [1.0, 3.0]
    assert scores == [2.0, 4.0]


def test_recycling_candidates_are_trainable_matrices() -> None:
    model = torch.nn.Sequential(
        OrderedDict(
            dense=torch.nn.Linear(2, 2),
            norm=torch.nn.BatchNorm1d(2),
        )
    )
    assert FedLUAR.recyclable_layer_names(
        model=model,
        public_names=model.state_dict(),
    ) == ["dense.weight"]

    server = object.__new__(FedLUAR)
    server._luar_candidate_layers = ["first.weight", "second.weight"]
    server._luar_scores = OrderedDict.fromkeys(server._luar_candidate_layers, 0.0)
    server._luar_first_round = False
    server._luar_recycle_layers = []
    server._luar_prev_params = {
        name: torch.zeros(1) for name in server._luar_candidate_layers
    }
    server._compute_metric(
        agg_delta={name: torch.ones(1) for name in server._luar_candidate_layers}
    )
    assert server._luar_scores["first.weight"] > 0
    server._luar_scores = OrderedDict.fromkeys(server._luar_candidate_layers, 0.0)
    server.luar_num_recycle_layers = 1
    server.seed = 1
    server.current_iter = 0
    server._update_layer_selection()
    assert len(server._luar_recycle_layers) == 1


def test_decomfl_uses_uniform_scalars_and_fresh_round_seeds() -> None:
    server = object.__new__(DeComFL)
    server.public_model_params = OrderedDict(
        first=torch.zeros(1),
        second=torch.zeros(1),
    )
    server.zo_lr = 1.0
    server._perturbation_seeds = [10, 20]
    perturbations = {
        10: torch.tensor([1.0, 0.0]),
        20: torch.tensor([0.0, 1.0]),
    }
    server.generate_perturbation = lambda *, dim, seed, device: perturbations[seed]
    server._commit_global = lambda *, new_params: setattr(
        server,
        "public_model_params",
        new_params,
    )

    server.aggregate_client_updates(
        packages=OrderedDict(
            {
                0: {"zo_g_scalars": [1.0, 3.0], "score": 99},
                1: {"zo_g_scalars": [3.0, 5.0], "score": 1},
            }
        )
    )

    assert torch.equal(server.public_model_params["first"], torch.tensor([-2.0]))
    assert torch.equal(server.public_model_params["second"], torch.tensor([-4.0]))
    assert server._perturbation_seeds == []


def test_classic_tfl_paper_contracts():
    name = "weight"
    assert all(issubclass(strategy, qFL) for strategy in (FedPAQ, QATFL))
    assert all(
        issubclass(client, qFL_Client) for client in (FedPAQ_Client, QATFL_Client)
    )

    avgm = _server(FedAvgM)
    avgm.server_momentum = 0.0
    avgm.server_learning_rate = 0.25
    avgm.momentum_vector = None
    avgm.aggregate_client_updates(
        OrderedDict(
            {0: {"score": 1, "regular_model_params": {name: torch.tensor([[2.0]])}}}
        )
    )
    assert torch.allclose(avgm.public_model_params[name], torch.tensor([[0.5]]))

    scaffold = object.__new__(SCAFFOLD)
    scaffold.global_c = [torch.tensor([0.0])]
    with patch.object(
        tFL,
        "package",
        return_value={"__wire__": ("regular_model_params",)},
    ):
        assert "global_c" in SCAFFOLD.package(scaffold, 0)["__wire__"]
    scaffold_client = object.__new__(SCAFFOLD_Client)
    scaffold_client.client_c = [torch.tensor([0.0])]
    scaffold_client._delta_c = [torch.tensor([1.0])]
    with patch.object(
        tFL_Client,
        "package",
        return_value={
            "__wire__": ("regular_model_params", "score"),
            "personal_model_params": {},
        },
    ):
        assert "delta_c" in SCAFFOLD_Client.package(scaffold_client)["__wire__"]

    nova_client = object.__new__(FedNova_Client)
    nova_client._nova_grad = [torch.tensor([1.0])]
    nova_client._tau = 2
    with patch.object(
        tFL_Client,
        "package",
        return_value={
            "__wire__": ("regular_model_params", "score"),
            "regular_model_params": {name: torch.tensor([1.0])},
            "score": 3,
        },
    ):
        nova_package = FedNova_Client.package(nova_client)
    assert nova_package["regular_model_params"] == {}
    assert nova_package["__wire__"] == ("nova_grad", "tau", "score")

    class TinyModel(torch.nn.Linear):
        def forward(self, x, x_mark=None, y_mark=None):
            return super().forward(x)

    nova_client = object.__new__(FedNova_Client)
    nova_client.model = TinyModel(1, 1, bias=False)
    nova_client.learning_rate = 0.1
    nova_client.nova_momentum = 0.0
    nova_client.prox_mu = 0.5
    nova_client.device = "cpu"
    nova_client.epochs = 2
    nova_client.loss = torch.nn.MSELoss()
    batch = (
        torch.ones(1, 1),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
    )
    nova_client.load_train_data = lambda: [batch]
    nova_client.fit()
    assert nova_client._tau == 2

    for strategy, expected_v in ((FedAdam, 0.010099), (FedYogi, 0.0101)):
        server = _server(strategy)
        server.beta1_server = 0.9
        server.beta2_server = 0.99
        server.eta_server = 0.1
        server.tau_server = 0.01
        server.m_t = {name: torch.zeros(1, 1)}
        server.v_t = {name: torch.full((1, 1), server.tau_server**2)}
        server.aggregate_client_updates(
            OrderedDict(
                {
                    0: {
                        "score": 1,
                        "regular_model_params": {name: torch.ones(1, 1)},
                    }
                }
            )
        )
        expected = 0.01 / (expected_v**0.5 + 0.01)
        assert torch.allclose(
            server.public_model_params[name],
            torch.tensor([[expected]]),
            atol=1e-6,
        )

    spa = _server(FedSPA, size=2)
    spa.beta1 = 0.0
    spa.beta2 = 0.0
    spa.global_lr = 1.0
    spa.kappa = 1.0
    spa._spa_u = {}
    spa._spa_v = {}
    spa.aggregate_client_updates(
        OrderedDict(
            {
                0: {
                    "spa_delta": {
                        name: (
                            torch.tensor([0], dtype=torch.int32),
                            torch.tensor([2.0]),
                        )
                    }
                },
                1: {
                    "spa_delta": {
                        name: (
                            torch.tensor([0], dtype=torch.int32),
                            torch.tensor([4.0]),
                        )
                    }
                },
            }
        )
    )
    assert torch.allclose(spa.public_model_params[name], torch.tensor([[0.75, 0.0]]))

    spa_client = object.__new__(FedSPA_Client)
    spa_client._spa_mask = {name: torch.tensor([[1.0, 0.0]])}
    spa_client._spa_initial_params = {name: torch.zeros(1, 2)}
    with patch.object(
        tFL_Client,
        "package",
        return_value={
            "__wire__": ("regular_model_params", "score"),
            "regular_model_params": {name: torch.tensor([[2.0, 9.0]])},
        },
    ):
        spa_package = FedSPA_Client.package(spa_client)
    indices, values = spa_package["spa_delta"][name]
    assert indices.dtype == torch.int32
    assert torch.equal(indices, torch.tensor([0], dtype=torch.int32))
    assert torch.equal(values, torch.tensor([2.0]))
    assert spa_package["__wire__"] == ("spa_delta",)

    paq_client = object.__new__(FedPAQ_Client)
    paq_client.s = 8
    paq_client._init_params = {
        "a": torch.zeros(2),
        "b": torch.zeros(2),
    }
    base_package = {
        "__wire__": ("regular_model_params", "score"),
        "regular_model_params": OrderedDict(
            (("a", torch.tensor([1.0, 2.0])), ("b", torch.tensor([3.0, 4.0])))
        ),
    }
    with (
        patch.object(tFL_Client, "package", return_value=base_package),
        patch.object(
            FedPAQShared,
            "quantize_tensor",
            side_effect=lambda tensor, levels: tensor,
        ) as quantize,
    ):
        paq_package = FedPAQ_Client.package(paq_client)
    assert quantize.call_count == 1
    assert quantize.call_args.kwargs["tensor"].shape == (4,)
    assert quantize.call_args.kwargs["levels"] == 8
    assert paq_package["regular_model_params"] == {}
    assert paq_package["__wire__"] == ("quantized_delta",)

    paq = _server(FedPAQ)
    paq.aggregate_client_updates(
        OrderedDict(
            {
                0: {"quantized_delta": {name: torch.tensor([[1.0]])}},
                1: {"quantized_delta": {name: torch.tensor([[3.0]])}},
            }
        )
    )
    assert torch.equal(paq.public_model_params[name], torch.tensor([[2.0]]))

    qat_client = object.__new__(QATFL_Client)
    qat_client.s = 16
    qat_client._init_params = {
        "a": torch.zeros(2),
        "b": torch.zeros(2),
    }
    base_package = {
        "__wire__": ("regular_model_params", "score"),
        "regular_model_params": OrderedDict(
            (("a", torch.tensor([1.0, 2.0])), ("b", torch.tensor([3.0, 4.0])))
        ),
    }
    with (
        patch.object(tFL_Client, "package", return_value=base_package),
        patch.object(
            QATFLShared,
            "quantize_tensor",
            side_effect=lambda tensor, levels: tensor,
        ) as quantize,
    ):
        qat_package = QATFL_Client.package(qat_client)
    assert quantize.call_count == 1
    assert quantize.call_args.kwargs["tensor"].shape == (4,)
    assert quantize.call_args.kwargs["levels"] == 16
    assert qat_package["regular_model_params"] == {}
    assert qat_package["__wire__"] == ("quantized_delta",)

    qat = _server(QATFL)
    qat.aggregate_client_updates(
        OrderedDict(
            {
                0: {"quantized_delta": {name: torch.tensor([[1.0]])}},
                1: {"quantized_delta": {name: torch.tensor([[3.0]])}},
            }
        )
    )
    assert torch.equal(qat.public_model_params[name], torch.tensor([[2.0]]))


from unittest.mock import Mock

from torch.utils.data import TensorDataset

from strategies.Elastic import Elastic, Elastic_Client
from strategies.FedADMM import FedADMM, FedADMM_Client, FedADMMShared
from strategies.FedAWA import FedAWA
from strategies.FedCross import FedCross, FedCross_Client
from strategies.FedLAW import FedLAW, FedLAW_Client
from strategies.FedRCL import FedRCL
from strategies.FedTrend import FedTrend


class TinyForecast(torch.nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[value]]))

    def forward(self, x, x_mark=None, y_mark=None):
        return x @ self.weight


def _linear_server(strategy, size=1):
    server = object.__new__(strategy)
    server.model = torch.nn.Linear(size, 1, bias=False)
    server.model.weight.data.zero_()
    server.public_model_params = OrderedDict(
        (name, value.detach().clone())
        for name, value in server.model.named_parameters()
    )
    return server


def test_fedadmm_uses_augmented_delta_tracking_and_paper_wire():
    name = "weight"
    delta = FedADMMShared.augmented_delta(
        {name: torch.tensor([1.0])},
        {name: torch.tensor([0.0])},
        {name: torch.tensor([2.0])},
        {name: torch.tensor([0.5])},
        rho=0.5,
    )
    assert torch.equal(delta[name], torch.tensor([2.0]))

    server = _linear_server(FedADMM)
    server.server_learning_rate = 0.5
    server.server_learning_rate_2 = 0.25
    server.target_round = 60
    server.current_iter = 0
    server.aggregate_client_updates(
        OrderedDict(
            {
                0: {"delta": {name: torch.tensor([[2.0]])}},
                1: {"delta": {name: torch.tensor([[4.0]])}},
            }
        )
    )
    assert torch.equal(server.public_model_params[name], torch.tensor([[1.5]]))

    server = object.__new__(FedADMM)
    server.public_model_params = OrderedDict({name: torch.tensor([9.0])})
    server.clients_personal_model_params = {
        3: {
            "y_i": OrderedDict({name: torch.tensor([0.0])}),
        }
    }
    with patch.object(
        tFL,
        "package",
        return_value={
            "__wire__": ("regular_model_params",),
            "regular_model_params": OrderedDict({name: torch.tensor([9.0])}),
        },
    ):
        package = FedADMM.package(server, 3)
    assert package["__wire__"] == ("regular_model_params",)
    assert torch.equal(package["regular_model_params"][name], torch.tensor([9.0]))

    client = object.__new__(FedADMM_Client)
    client.rho = 0.5
    client._previous_w = OrderedDict({name: torch.tensor([1.0])})
    client._y = OrderedDict({name: torch.tensor([0.0])})
    client._theta = OrderedDict({name: torch.tensor([1.0])})
    with patch.object(
        tFL_Client,
        "package",
        return_value={
            "__wire__": ("regular_model_params", "score"),
            "regular_model_params": OrderedDict({name: torch.tensor([2.0])}),
        },
    ):
        package = FedADMM_Client.package(client)
    assert package["regular_model_params"] == OrderedDict()
    assert package["__wire__"] == ("delta",)
    assert torch.equal(package["delta"][name], torch.tensor([2.0]))


def test_elastic_is_element_wise_and_uploads_sensitivity():
    name = "weight"
    server = _linear_server(Elastic, size=2)
    server.tau = 0.5
    server.server_learning_rate = 1.0
    packages = OrderedDict(
        {
            0: {
                "score": 1,
                "regular_model_params": {name: torch.tensor([[2.0, 2.0]])},
                "sensitivity": {name: torch.tensor([[1.0, 0.0]])},
            },
            1: {
                "score": 1,
                "regular_model_params": {name: torch.tensor([[2.0, 2.0]])},
                "sensitivity": {name: torch.tensor([[1.0, 0.0]])},
            },
        }
    )
    server.aggregate_client_updates(packages)
    assert torch.equal(server.public_model_params[name], torch.tensor([[1.0, 3.0]]))

    client = object.__new__(Elastic_Client)
    client._sensitivity = {name: torch.ones(1)}
    with patch.object(
        tFL_Client,
        "package",
        return_value={"__wire__": ("regular_model_params", "score")},
    ):
        package = Elastic_Client.package(client)
    assert package["__wire__"] == (
        "regular_model_params",
        "score",
        "sensitivity",
    )


def test_fedcross_dispatches_slots_and_generates_unweighted_global_model():
    name = "weight"
    server = _linear_server(FedCross)
    server.cross_alpha = 0.99
    server.collaborative_model_select_strategy = 1
    server.current_iter = 0
    server.aggregate_client_updates(
        OrderedDict(
            {
                7: {
                    "score": 99,
                    "regular_model_params": {name: torch.tensor([[0.0]])},
                },
                3: {
                    "score": 1,
                    "regular_model_params": {name: torch.tensor([[2.0]])},
                },
            }
        )
    )
    assert torch.allclose(server.middleware_models[0][name], torch.tensor([[0.02]]))
    assert torch.equal(server.public_model_params[name], torch.tensor([[1.0]]))

    server = object.__new__(FedCross)
    server.selected_clients = [7, 3]
    server.middleware_models = [
        {name: torch.tensor([1.0])},
        {name: torch.tensor([2.0])},
    ]
    with patch.object(tFL, "package", return_value={"regular_model_params": {}}):
        package = FedCross.package(server, 3)
    assert torch.equal(package["regular_model_params"][name], torch.tensor([2.0]))

    client = object.__new__(FedCross_Client)
    with patch.object(
        tFL_Client,
        "package",
        return_value={"__wire__": ("regular_model_params", "score")},
    ):
        assert FedCross_Client.package(client)["__wire__"] == ("regular_model_params",)


def test_fedrcl_uses_unweighted_fedavg_and_full_paper_loss_weight():
    name = "weight"
    server = _linear_server(FedRCL)
    server.aggregate_client_updates(
        OrderedDict(
            {
                0: {
                    "score": 99,
                    "regular_model_params": {name: torch.tensor([[0.0]])},
                },
                1: {
                    "score": 1,
                    "regular_model_params": {name: torch.tensor([[2.0]])},
                },
            }
        )
    )
    assert torch.equal(server.public_model_params[name], torch.tensor([[1.0]]))
    assert FedRCL.optional["rcl_weight"] == 1.0


def test_fedlaw_reuses_the_normal_model_upload():
    assert FedLAW_Client.package is tFL_Client.package
    server = object.__new__(FedLAW)
    server.model = TinyForecast()
    server.public_model_params = OrderedDict(
        (name, value.detach().clone())
        for name, value in server.model.named_parameters()
    )
    server.device = "cpu"
    server.server_lr = 0.005
    server.server_epochs = 1
    sample = torch.ones(1, 1)
    marks = torch.zeros(1, 1)
    server.public_loader = [(sample, torch.full((1, 1), 2.0), marks, marks)]
    server.aggregate_client_updates(
        OrderedDict(
            {
                0: {
                    "score": 1,
                    "regular_model_params": {"weight": torch.tensor([[1.0]])},
                },
                1: {
                    "score": 1,
                    "regular_model_params": {"weight": torch.tensor([[3.0]])},
                },
            }
        )
    )
    assert torch.allclose(
        server.public_model_params["weight"], torch.tensor([[2.0]]), atol=1e-5
    )


def test_fedawa_optimizer_updates_the_current_round_logits():
    server = object.__new__(FedAWA)
    server.public_model_params = OrderedDict({"weight": torch.tensor([1.0, 1.0])})
    server.device = "cpu"
    server.server_epochs = 3
    server.server_lr = 0.1
    server.server_optimizer = "Adam"
    server.reg_distance = "cos"
    server.awa_weights = {}
    server._commit_global = lambda new_params: setattr(
        server, "public_model_params", OrderedDict(new_params)
    )

    first_packages = OrderedDict(
        {
            0: {
                "score": 1,
                "regular_model_params": {"weight": torch.tensor([3.0, 0.0])},
            },
            1: {
                "score": 3,
                "regular_model_params": {"weight": torch.tensor([0.0, 2.0])},
            },
        }
    )
    server.aggregate_client_updates(first_packages)
    first = torch.stack([server.awa_weights[0], server.awa_weights[1]])

    second_packages = OrderedDict(
        {
            0: {
                "score": 1,
                "regular_model_params": {"weight": torch.tensor([4.0, 0.0])},
            },
            1: {
                "score": 3,
                "regular_model_params": {"weight": torch.tensor([0.0, 1.0])},
            },
        }
    )
    server.aggregate_client_updates(second_packages)
    second = torch.stack([server.awa_weights[0], server.awa_weights[1]])
    assert not torch.allclose(first, second)


def test_fedtrend_condensation_keeps_the_higher_order_gradient_and_wire():
    server = object.__new__(FedTrend)
    server.model = TinyForecast()
    server.loss = torch.nn.MSELoss()
    server.learning_rate = 0.1
    server.batch_size = 1
    server.logger = Mock()
    trajectory = {
        0: {
            "start": {"weight": torch.tensor([[0.0]])},
            "end": {"weight": torch.tensor([[1.0]])},
            "mask": None,
        }
    }
    torch.manual_seed(0)
    synthetic = server._data_construction(
        trajectories=trajectory,
        data_size=1,
        input_shape=(1,),
        output_shape=(1,),
        synthetic_epochs=1,
        synthetic_inner_epochs=1,
        synthetic_lr=0.1,
        x_mark_shape=(1,),
        y_mark_shape=(1,),
        is_client_trajectory=True,
    )
    assert isinstance(synthetic, TensorDataset)
    assert len(synthetic.tensors) == 4

    server = object.__new__(FedTrend)
    server.D_ct = synthetic
    with patch.object(
        tFL,
        "package",
        return_value={"__wire__": ("regular_model_params",)},
    ):
        package = FedTrend.package(server, 0)
    assert package["__wire__"] == ("regular_model_params", "D_ct")


from argparse import Namespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.InfoTS import InfoTSShared
from models.SimTS import SimTS as SimTSModel
from strategies.Centralized import Centralized
from strategies.InfoTS import InfoTS
from strategies.LocalOnly import LocalOnly
from strategies.nFL import nFL, nFLShared
from strategies.pFL import pFL
from strategies.SimTS import SimTS
from strategies.SL import SL, SL_Client


class _RidgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 1)

    def representation(self, x):
        return x[:, 0]

    def forward(self, x):
        return self.head(self.representation(x))


def test_nfl_hierarchy_and_shared_paper_formulations():
    assert issubclass(nFL, pFL)
    assert all(issubclass(strategy, nFL) for strategy in (LocalOnly, SimTS, InfoTS, SL))
    assert issubclass(Centralized, tFL) and not issubclass(Centralized, nFL)

    centralized = object.__new__(Centralized)
    centralized.num_clients = 3
    centralized.is_new = {0: False, 1: True, 2: False}
    centralized.select_clients()
    assert centralized.selected_clients == [0, 2]

    server = object.__new__(nFL)
    server.clients_personal_model_params = {0: {}}
    server.clients_auxiliary_state = {0: {}}
    server.aggregate_client_updates(
        OrderedDict(
            [
                (
                    0,
                    {
                        "regular_model_params": {"weight": torch.tensor([2.0])},
                        "auxiliary_state": {"pretrained": True},
                    },
                )
            ]
        )
    )
    assert torch.equal(
        server.clients_personal_model_params[0]["weight"], torch.tensor([2.0])
    )
    assert server.clients_auxiliary_state[0] == {"pretrained": True}
    assert server._compute_send_mb({0: {}}) == ({}, 0.0)

    x = torch.tensor([[[0.0]], [[1.0]], [[2.0]]])
    y = 2 * x + 3
    marks = torch.zeros_like(x)
    model = _RidgeModel()
    nFLShared.fit_ridge_head(
        model,
        DataLoader(TensorDataset(x, y, marks, marks), batch_size=2),
        "cpu",
        alpha=0,
    )
    assert torch.allclose(model(x), y[:, 0], atol=1e-6)

    raw = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    expected = torch.log1p(torch.exp(torch.tensor(-1.0)))
    assert torch.allclose(InfoTSShared.global_info_nce(raw, raw), expected)
    assert torch.allclose(InfoTSShared.l1out(raw, raw), torch.tensor(1.0))

    simts = SimTSModel(
        Namespace(
            input_channels=2,
            output_channels=2,
            input_len=8,
            output_len=2,
            simts_hidden_dim=4,
            simts_repr_dim=6,
            simts_K=4,
            simts_kernel_list="auto",
        )
    )
    simts_input = torch.randn(2, 8, 2)
    simts.pretrain_loss(simts_input).backward()
    assert simts(simts_input).shape == (2, 2, 2)
    assert simts.encoder.multi_cnn[0].out_channels == 6

    residual = torch.tensor([[[0.0], [2.0]], [[4.0], [6.0]]])
    assert torch.allclose(
        SL_Client._compute_entropy(residual).squeeze(),
        torch.tensor([0.0, 1.0, 0.0]),
    )


import pytest
import torch

from strategies.FedMedian import FedMedian
from strategies.FedTrimmedAvg import FedTrimmedAvg
from strategies.Krum import Krum
from strategies.sFL import sFL, sFL_Client, sFLShared


def _model(value):
    return OrderedDict(weight=torch.tensor([float(value)]))


def _sfl_server(strategy):
    server = object.__new__(strategy)
    server.public_model_params = _model(0)
    server._commit_global = lambda new_params: setattr(
        server, "public_model_params", OrderedDict(new_params)
    )
    return server


def test_sfl_paper_contracts_and_shared_aggregators():
    assert issubclass(sFL, tFL)

    with pytest.raises(ValueError, match=r"2 \* f \+ 2 < n"):
        sFLShared.krum_scores([_model(i) for i in range(4)], 1)

    krum = _sfl_server(Krum)
    krum.num_malicious_clients = 0
    krum.num_clients_to_keep = 0
    krum.malicious_ids = set()
    krum.aggregate_client_updates(
        OrderedDict(
            {
                9: {"regular_model_params": _model(0)},
                2: {"regular_model_params": _model(2)},
                5: {"regular_model_params": _model(4)},
            }
        )
    )
    assert torch.equal(krum.public_model_params["weight"], torch.tensor([2.0]))

    median = _sfl_server(FedMedian)
    median.aggregate_client_updates(
        OrderedDict(
            {
                0: {"regular_model_params": _model(0)},
                1: {"regular_model_params": _model(2)},
            }
        )
    )
    assert torch.equal(median.public_model_params["weight"], torch.tensor([1.0]))

    trimmed = _sfl_server(FedTrimmedAvg)
    trimmed.beta = 0.25
    trimmed.aggregate_client_updates(
        OrderedDict(
            (i, {"regular_model_params": _model(value)})
            for i, value in enumerate((0, 1, 2, 100))
        )
    )
    assert torch.equal(trimmed.public_model_params["weight"], torch.tensor([1.5]))
    with pytest.raises(ValueError, match=r"\[0, 0.5\)"):
        sFLShared.coordinate_trimmed_mean([_model(0), _model(1)], 0.5)

    client = object.__new__(sFL_Client)
    with patch.object(
        tFL_Client,
        "package",
        return_value={
            "__wire__": ("regular_model_params", "score"),
            "regular_model_params": _model(1),
            "score": 10,
        },
    ):
        package = sFL_Client.package(client)
    assert package["__wire__"] == ("regular_model_params",)
    assert "score" not in package


from strategies import compulsory as strategy_compulsory
from strategies.AirMetapFL import AirMetapFL, AirMetapFL_Client, AirMetapFLShared
from strategies.APFL import APFL, APFLShared
from strategies.base import SharedMethods
from strategies.CFL import CFL, CFL_Client, CFLShared
from strategies.Ditto import Ditto
from strategies.FedALA import FedALA, FedALA_Client
from strategies.FedAMP import FedAMP, FedAMPShared
from strategies.FedBN import FedBN, FedBNShared
from strategies.FedCAC import FedCAC, FedCAC_Client, FedCACShared
from strategies.FedDF import FedDF, FedDF_Client
from strategies.FedDyn import FedDyn, FedDynShared
from strategies.FedFew import FedFew, FedFew_Client, FedFewShared
from strategies.FedIT import FedIT, FedIT_Client
from strategies.FedMD import FedMD, FedMD_Client, FedMDShared
from strategies.FedProx import FedProx
from strategies.FedSA_LoRA import FedSA_LoRA, FedSA_LoRA_Client
from strategies.FedSelect import FedSelect, FedSelectShared
from strategies.FFA_LoRA import FFA_LoRA, FFA_LoRA_Client
from strategies.FlexLoRA import FlexLoRA, FlexLoRA_Client, FlexLoRAShared
from strategies.FML import FML
from strategies.hFL import hFL, hFL_Client
from strategies.LGFedAvg import LGFedAvg, LGFedAvgShared
from strategies.LoRA_FAIR import LoRA_FAIR, LoRA_FAIR_Client
from strategies.mFL import mFL, mFL_Client
from strategies.peftFL import peftFL, peftFL_Client
from strategies.PerAvg import PerAvg, PerAvg_Client
from strategies.pFedHN import pFedHN, pFedHN_Client
from strategies.pFedLA import pFedLA, pFedLA_Client, pFedLAHyperNetwork
from strategies.pFedMe import pFedMe, pFedMe_Client
from strategies.pFL import pFL_Client, pFLShared


def test_personalized_family_paper_contracts():
    assert issubclass(FedProx, tFL) and not issubclass(FedProx, pFL)
    assert all(
        issubclass(strategy, pFL)
        for strategy in (Ditto, pFedMe, APFL, PerAvg, FedAMP, FedBN, FML)
    )
    assert issubclass(FedDyn, tFL) and not issubclass(FedDyn, pFL)

    mean = SharedMethods.mean_models([_model(1), _model(3)])
    assert torch.equal(mean["weight"], torch.tensor([2.0]))
    weighted = SharedMethods.mean_models(models=[_model(0), _model(4)], weights=[1, 3])
    assert torch.equal(weighted["weight"], torch.tensor([3.0]))
    state_mean = SharedMethods.mean_models(
        [
            OrderedDict(weight=torch.tensor([1.0]), counter=torch.tensor(2)),
            OrderedDict(weight=torch.tensor([3.0]), counter=torch.tensor(8)),
        ]
    )
    assert torch.equal(state_mean["counter"], torch.tensor(2))
    nested = pFLShared.personalized_model_state(
        _model(0), {"model_per": _model(4), "alpha": 0.5}
    )
    assert torch.equal(nested["weight"], torch.tensor([4.0]))
    masked = pFLShared.personalized_model_state(
        base_state=_model(0),
        personal_params={
            "mask": {"weight": torch.tensor([True])},
            "local_model_state": _model(5),
        },
    )
    assert torch.equal(masked["weight"], torch.tensor([5.0]))

    personal = OrderedDict(weight=torch.tensor([3.0], requires_grad=True))
    alpha = torch.tensor(0.25, requires_grad=True)
    mixed = APFLShared.mix_parameters(personal, _model(1), alpha)
    mixed["weight"].square().sum().backward()
    assert torch.allclose(mixed["weight"], torch.tensor([1.5]))
    assert torch.allclose(personal["weight"].grad, torch.tensor([0.75]))
    assert torch.allclose(alpha.grad, torch.tensor(6.0))

    normalizer = torch.nn.Sequential(
        OrderedDict([("normalizer", torch.nn.BatchNorm1d(2))])
    )
    assert set(FedBNShared.batch_norm_state_names(normalizer)) == {
        "normalizer.weight",
        "normalizer.bias",
        "normalizer.running_mean",
        "normalizer.running_var",
        "normalizer.num_batches_tracked",
    }
    updated_dual = FedDynShared.update_dual(_model(0), _model(3), _model(1), alpha=0.5)
    assert torch.equal(updated_dual["weight"], torch.tensor([-1.0]))
    with pytest.raises(ValueError, match="sigma > 0"):
        FedAMPShared.validate_hyperparameters(1.0, 0.0, 1.0)

    server = _sfl_server(pFedMe)
    server.beta = 2.0
    server.public_model_params = _model(1)
    server.aggregate_client_updates(
        OrderedDict(
            {
                0: {"regular_model_params": _model(2), "score": 1},
                1: {"regular_model_params": _model(4), "score": 999},
            }
        )
    )
    assert torch.equal(server.public_model_params["weight"], torch.tensor([5.0]))

    client = object.__new__(pFedMe_Client)
    client.upload_model = False
    client._personalized_params = [torch.tensor([2.0])]
    with patch.object(
        pFL_Client,
        "package",
        return_value={
            "__wire__": ("regular_model_params", "personal_model_params"),
            "personal_model_params": {},
        },
    ):
        package = pFedMe_Client.package(client)
    assert package["__wire__"] == ("personal_model_params",)


def test_structural_personalization_paper_contracts():
    assert all(
        issubclass(strategy, pFL) for strategy in (CFL, LGFedAvg, pFedHN, pFedLA)
    )
    assert all(
        issubclass(client, pFL_Client) and client.return_diff
        for client in (CFL_Client, pFedHN_Client, pFedLA_Client)
    )
    assert all(
        strategy_compulsory[name]["return_diff"] for name in ("CFL", "pFedHN", "pFedLA")
    )

    diffs = [[torch.tensor([0.0])], [torch.tensor([4.0])]]
    assert torch.equal(
        CFLShared.weighted_mean(diffs, [3, 1])[0],
        torch.tensor([1.0]),
    )
    assert CFLShared.mean_norm(diffs, [3, 1]) == pytest.approx(1.0)

    names = [
        "encoder.weight",
        "encoder.bias",
        "head.weight",
        "head.bias",
    ]
    assert LGFedAvgShared.global_param_names(names, 1) == {
        "head.weight",
        "head.bias",
    }

    cfl = object.__new__(CFL)
    cfl.num_clients = 3
    cfl.is_new = {0: False, 1: True, 2: False}
    cfl.select_clients()
    assert cfl.selected_clients == [0, 2]

    hn = object.__new__(pFedHN)
    hn.num_clients = 3
    hn.is_new = {0: False, 1: True, 2: False}
    with patch("numpy.random.choice", return_value=2):
        hn.select_clients()
    assert hn.selected_clients == [2]

    layer_aggregate = object.__new__(pFedLA)
    layer_aggregate.num_clients = 3
    layer_aggregate.is_new = {0: False, 1: True, 2: False}
    layer_aggregate.select_clients()
    assert layer_aggregate.selected_clients == [0, 2]
    layer_aggregate.device = "cpu"
    layer_aggregate.public_model_params = _model(0)
    layer_aggregate.clients_personal_model_params = {
        0: _model(9),
        1: _model(100),
        2: _model(11),
    }
    layer_aggregate._round_model_params = {
        0: _model(1),
        1: _model(100),
        2: _model(3),
    }
    aggregated = layer_aggregate._aggregate_model(
        0, OrderedDict(weight=torch.tensor([1.0, 100.0, 1.0]))
    )
    assert torch.equal(aggregated["weight"], torch.tensor([2.0]))

    diff_client = object.__new__(pFedHN_Client)
    diff_client.model = torch.nn.Linear(1, 1, bias=False)
    diff_client.model.weight.data.fill_(3)
    diff_client.optimizer = torch.optim.SGD(diff_client.model.parameters(), lr=0.1)
    diff_client.scheduler = torch.optim.lr_scheduler.LambdaLR(
        diff_client.optimizer, lambda _: 1
    )
    diff_client.id = 0
    diff_client.train_samples = 5
    diff_client.regular_params_name = ["weight"]
    diff_client.personal_params_name = []
    diff_client._initial_regular_params = _model(1)
    diff_client.return_diff = True
    package = diff_client.package()
    assert package["__wire__"] == ("model_params_diff",)
    assert torch.equal(package["model_params_diff"]["weight"], torch.tensor([[-2.0]]))

    hypernetwork = pFedLAHyperNetwork(
        n_clients=3,
        embedding_dim=2,
        hidden_dim=4,
        layer_names=("encoder", "head"),
        retained_layers=0,
    )
    alpha, retained = hypernetwork(1)
    assert tuple(alpha) == ("encoder", "head")
    assert retained == set()
    assert all(weights.shape == (3,) for weights in alpha.values())


def test_parameter_efficient_paper_contracts():
    assert all(
        issubclass(strategy, peftFL)
        for strategy in (FedIT, FFA_LoRA, LoRA_FAIR, FlexLoRA, FedSA_LoRA)
    )
    assert all(
        issubclass(client, peftFL_Client)
        for client in (
            FedIT_Client,
            FFA_LoRA_Client,
            LoRA_FAIR_Client,
            FlexLoRA_Client,
            FedSA_LoRA_Client,
        )
    )
    assert not issubclass(FedIT, pFL)
    assert issubclass(FedSA_LoRA, pFL)
    assert strategy_compulsory["FedSelect"]["optimizer"] == "SGD"

    update = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
    u, singular_values, vh = torch.linalg.svd(update, full_matrices=False)
    a_factor, b_factor = FlexLoRAShared.factors_from_svd(
        u=u,
        singular_values=singular_values,
        vh=vh,
        rank=2,
        alpha=4.0,
    )
    assert torch.allclose((4.0 / 2.0) * a_factor @ b_factor, update)

    mask = OrderedDict(
        weight=torch.zeros(4, dtype=torch.bool),
        bias=torch.zeros(2, dtype=torch.bool),
    )
    updated = FedSelectShared.updated_mask(
        mask=mask,
        trained_state=OrderedDict(
            weight=torch.tensor([1.0, 4.0, 3.0, 2.0]),
            bias=torch.tensor([9.0, 9.0]),
        ),
        initial_state=OrderedDict(
            weight=torch.zeros(4),
            bias=torch.zeros(2),
        ),
        personalization_rate=0.5,
        personalization_limit=0.5,
    )
    assert torch.equal(updated["weight"], torch.tensor([0, 1, 1, 0]).bool())
    assert not updated["bias"].any()

    server = _server(FedSelect, size=2)
    server.aggregate_client_updates(
        packages=OrderedDict(
            {
                0: {
                    "regular_model_params": {"weight": torch.tensor([[2.0, 4.0]])},
                    "personal_model_params": {
                        "mask": {"weight": torch.tensor([[False, True]])}
                    },
                },
                1: {
                    "regular_model_params": {"weight": torch.tensor([[6.0, 8.0]])},
                    "personal_model_params": {
                        "mask": {"weight": torch.tensor([[False, False]])}
                    },
                },
            }
        )
    )
    assert torch.equal(server.public_model_params["weight"], torch.tensor([[4.0, 8.0]]))


def test_heterogeneous_paper_contracts():
    assert all(issubclass(strategy, hFL) for strategy in (FedMD, FedDF))
    assert issubclass(FedMD_Client, hFL_Client)
    assert issubclass(FedDF_Client, hFL_Client)

    consensus = FedMDShared.mean_logits(
        client_logits=[
            [torch.tensor([1.0]), torch.tensor([3.0])],
            [torch.tensor([5.0]), torch.tensor([7.0])],
        ]
    )
    assert torch.equal(consensus[0], torch.tensor([3.0]))
    assert torch.equal(consensus[1], torch.tensor([5.0]))

    stateless = object.__new__(hFL)
    stateless.current_iter = 1
    stateless.public_model_params = OrderedDict()
    stateless.clients_personal_model_params = {0: _model(2)}
    stateless.client_optimizer_states = {0: {}}
    stateless.client_scheduler_states = {0: {}}
    package = stateless.package(client_id=0)
    assert package["__wire__"] == ()
    assert package["personal_model_params"] == {}
    assert torch.equal(package["regular_model_params"]["weight"], torch.tensor([2.0]))

    feddf_client = object.__new__(FedDF_Client)
    with patch.object(
        hFL_Client,
        "package",
        return_value={"regular_model_params": {}, "score": 1},
    ):
        upload = feddf_client.package()
    assert upload["__wire__"] == ("regular_model_params", "score")

    class ForecastScale(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([value]))

        def forward(
            self,
            x: torch.Tensor,
            x_mark: torch.Tensor | None = None,
            y_mark: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return x * self.weight

    worker = SimpleNamespace(model=ForecastScale(value=0.0))
    trainer = SimpleNamespace(
        prototype_workers={0: worker},
        client_prototypes={0: 0, 1: 0},
        prototype_clients={0: [0, 1]},
        worker_for=lambda client_id: worker,
    )
    fusion = object.__new__(FedDF)
    fusion.trainer = trainer
    fusion.device = "cpu"
    fusion.distill_epochs = 1
    fusion.distill_lr = 0.0
    zeros = torch.zeros(2, 1)
    fusion.public_loader = [(zeros, zeros, zeros, zeros)]
    fusion.prototype_model_params = {0: OrderedDict(weight=torch.tensor([0.0]))}
    fusion.clients_personal_model_params = {0: {}, 1: {}}
    fusion.aggregate_client_updates(
        packages=OrderedDict(
            {
                0: {
                    "regular_model_params": OrderedDict(weight=torch.tensor([1.0])),
                    "score": 1,
                },
                1: {
                    "regular_model_params": OrderedDict(weight=torch.tensor([3.0])),
                    "score": 3,
                },
            }
        )
    )
    assert torch.equal(fusion.prototype_model_params[0]["weight"], torch.tensor([2.5]))
    assert torch.equal(
        fusion.clients_personal_model_params[1]["weight"], torch.tensor([2.5])
    )


def test_adaptive_personalization_paper_contracts():
    assert all(issubclass(strategy, pFL) for strategy in (FedALA, FedCAC, FedFew))
    assert all(issubclass(strategy, mFL) for strategy in (PerAvg, AirMetapFL))
    assert all(
        issubclass(client, mFL_Client) for client in (PerAvg_Client, AirMetapFL_Client)
    )
    assert AirMetapFL_Client.return_diff
    assert strategy_compulsory["AirMetapFL"]["return_diff"]

    update = torch.tensor([1.0, 4.0, 3.0, 2.0])
    compressed, memory = AirMetapFLShared.compress_with_memory(
        update, torch.zeros_like(update), ratio=0.5
    )
    assert torch.equal(compressed, torch.tensor([0.0, 4.0, 3.0, 0.0]))
    assert torch.equal(compressed + memory, update)
    air_mean = AirMetapFLShared.aggregate_over_air(
        [torch.tensor([1.0, 0.0]), torch.tensor([3.0, 0.0])],
        compression_ratio=1.0,
        sparsity=1.0,
        learning_rate=0.1,
        power=1.0,
        noise_std=0.0,
        channel_mean=1.0,
        estimator_steps=1,
        seed=0,
        channel_gains=torch.ones(2),
    )
    assert torch.allclose(air_mean, torch.tensor([2.0, 0.0]), atol=1e-6)

    personalized = pFLShared.personalized_model_state(
        _model(0), {"personalized_params": [torch.tensor([4.0])]}, ["weight"]
    )
    assert torch.equal(personalized["weight"], torch.tensor([4.0]))
    evaluator = object.__new__(pFL_Client)
    evaluator.model = torch.nn.Linear(1, 1, bias=False)
    with patch.object(
        tFL_Client, "evaluate_personalized", return_value=0.0
    ) as base_eval:
        evaluator.evaluate_personalized(
            0,
            OrderedDict(weight=torch.tensor([[0.0]])),
            {"personalized_params": [torch.tensor([[4.0]])]},
            "test",
            0,
        )
    assert torch.equal(
        base_eval.call_args.kwargs["personal_params"]["weight"],
        torch.tensor([[4.0]]),
    )
    assert FedALA.optional["threshold"] == pytest.approx(0.01)

    losses = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    alpha, model_weights = FedFewShared.stch_weights(
        losses, torch.tensor([1.0, 1.0]), mu=1.0
    )
    assert torch.allclose(alpha, torch.tensor([0.5, 0.5]))
    assert torch.allclose(model_weights.sum(1), torch.ones(2))
    assert model_weights[0, 0] > model_weights[0, 1]
    assert FedFew.optional == {"num_models": 3, "mu": 0.01}

    few = _sfl_server(FedFew)
    few.num_models = 1
    few.mu = 1.0
    few.learning_rate = 1.0
    few.server_models = [_model(10)]
    few.clients_personal_model_params = {0: {}, 1: {}}
    few.aggregate_client_updates(
        OrderedDict(
            {
                0: {
                    "fedfew_losses": [0.0],
                    "fedfew_gradients": [_model(2)],
                    "score": 1,
                },
                1: {
                    "fedfew_losses": [0.0],
                    "fedfew_gradients": [_model(4)],
                    "score": 1,
                },
            }
        )
    )
    assert torch.equal(few.server_models[0]["weight"], torch.tensor([7.0]))

    for strategy in (FedFew, FedCAC):
        server = object.__new__(strategy)
        server.num_clients = 3
        server.is_new = {0: False, 1: True, 2: False}
        server.select_clients()
        assert server.selected_clients == [0, 2]

    previous = [torch.zeros(4)]
    current = [torch.tensor([1.0, 2.0, 3.0, 4.0])]
    mask = FedCACShared.critical_masks(previous, current, tau=0.5)
    assert torch.equal(mask[0], torch.tensor([0, 0, 1, 1]))
    assert FedCACShared.overlap_rate(
        [torch.tensor([1, 1, 0, 0])],
        [torch.tensor([0, 0, 1, 1])],
    ) == pytest.approx(0.5)

    cac = _sfl_server(FedCAC)
    cac.current_iter = 0
    cac.beta = 170
    cac.clients_personal_model_params = {
        0: {"local_mask": [torch.tensor([1])]},
        1: {"local_mask": [torch.tensor([1])]},
    }
    cac.aggregate_client_updates(
        OrderedDict(
            {
                0: {"regular_model_params": _model(0), "score": 1},
                1: {"regular_model_params": _model(4), "score": 999},
            }
        )
    )
    assert torch.equal(cac.public_model_params["weight"], torch.tensor([2.0]))
    assert torch.equal(
        cac.clients_personal_model_params[0]["model_per"]["weight"],
        torch.tensor([0.0]),
    )

    with patch.object(
        pFL_Client,
        "package",
        return_value={
            "__wire__": ("regular_model_params", "personal_model_params"),
            "regular_model_params": _model(1),
            "personal_model_params": {},
            "score": 1,
        },
    ):
        ala_client = object.__new__(FedALA_Client)
        ala_client.model = torch.nn.Linear(1, 1, bias=False)
        ala_client._ala_weights = None
        ala_client._ala_start_phase = False
        ala_package = FedALA_Client.package(ala_client)
        assert ala_package["__wire__"] == ("regular_model_params", "score")

        cac_client = object.__new__(FedCAC_Client)
        cac_client._local_mask = [torch.tensor([1])]
        cac_package = FedCAC_Client.package(cac_client)
        assert cac_package["__wire__"] == ("regular_model_params", "critical_mask")

        few_client = object.__new__(FedFew_Client)
        few_client._fedfew_losses = [1.0]
        few_client._fedfew_gradients = [_model(1)]
        few_package = FedFew_Client.package(few_client)
        assert few_package["__wire__"] == (
            "fedfew_gradients",
            "fedfew_losses",
            "score",
        )


def test_web_paper_explorer_contract():
    import csv
    from pathlib import Path

    web = Path(__file__).parents[1] / "web"
    page = (web / "index.html").read_text(encoding="utf-8")
    assert not (web / "papers.csv").exists()
    for topic in ("fl", "tsf"):
        path = web / f"papers-{topic}.csv"
        assert path.name in page
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        ids = [row["ID"] for row in rows]
        assert rows and len(ids) == len(set(ids))
        assert all(row["Title"] for row in rows)


from strategies.DFedHPO import DFedHPO, DFedHPO_Client, DFedHPO_Trainer
from strategies.DFedSAM import DFedSAM, DFedSAM_Client
from strategies.dFL import dFL, dFL_Client
from strategies.FedAWA import DFedAWA, FedAWAShared
from strategies.FedProx import DFedProx, DFedProx_Client, FedProx_Client
from strategies.tFL import Trainer


def test_decentralized_paper_contracts():
    assert all(
        issubclass(strategy, dFL) for strategy in (DFedProx, DFedSAM, DFedAWA, DFedHPO)
    )
    assert all(
        issubclass(DFedProx_Client, base) for base in (FedProx_Client, dFL_Client)
    )
    assert issubclass(DFedHPO_Trainer, Trainer)

    gossip = object.__new__(dFL)
    gossip.num_clients = 3
    gossip.topology = {0: [1], 1: [0, 2], 2: [1]}
    gossip.clients_personal_model_params = {
        index: _model(value) for index, value in enumerate((0, 2, 4))
    }
    gossip._gossip_once()
    assert torch.allclose(
        gossip.clients_personal_model_params[0]["weight"], torch.tensor([2 / 3])
    )
    assert torch.equal(
        gossip.clients_personal_model_params[1]["weight"], torch.tensor([2.0])
    )
    assert torch.allclose(
        gossip.clients_personal_model_params[2]["weight"], torch.tensor([10 / 3])
    )

    gossip.get_size = lambda obj: 1.0
    uplink, downlink = gossip._compute_send_mb(
        {index: {"regular_model_params": _model(index)} for index in range(3)}
    )
    assert uplink == {0: 1.0, 1: 2.0, 2: 1.0}
    assert downlink == 4.0

    sam = object.__new__(DFedSAM)
    sam.use_mgs, sam.mgs_steps = True, 4
    assert sam._num_gossip_steps() == 4
    assert DFedSAM.optional == {"use_mgs": False, "mgs_steps": 4, "rho": 0.01}
    model = torch.nn.Linear(2, 1)
    model(torch.ones(1, 2)).sum().backward()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    norm = DFedSAM_Client._grad_norm(model=model)
    parameters, perturbations = DFedSAM_Client._add_perturbation(
        model=model, rho=0.01, grad_norm=norm
    )
    DFedSAM_Client._remove_perturbation(
        parameters=parameters, perturbations=perturbations
    )
    assert all(
        torch.equal(parameter, original)
        for parameter, original in zip(model.parameters(), before)
    )

    weights, _ = FedAWAShared.optimize_weights(
        models=[_model(1), _model(3)],
        reference=_model(0),
        initial_logits=torch.tensor([1.0, 3.0]).log(),
        epochs=0,
        learning_rate=0.001,
        optimizer_name="Adam",
        distance="cos",
        device="cpu",
    )
    assert torch.allclose(weights, torch.tensor([0.25, 0.75]))

    awa = object.__new__(DFedAWA)
    awa.topology = {0: [1], 1: [0]}
    awa.public_model_params = _model(0)
    awa.clients_personal_model_params = {0: _model(0), 1: _model(0)}
    awa._round_reference_models = {0: _model(0), 1: _model(0)}
    awa.awa_weights = {}
    awa.server_epochs = 0
    awa.server_lr = 0.001
    awa.server_optimizer = "Adam"
    awa.reg_distance = "cos"
    awa.device = "cpu"
    awa.aggregate_client_updates(
        packages={
            0: {"regular_model_params": _model(1), "score": 1},
            1: {"regular_model_params": _model(3), "score": 3},
        }
    )
    assert all(
        torch.allclose(state["weight"], torch.tensor([2.5]))
        for state in awa.clients_personal_model_params.values()
    )

    config = DFedHPO_Client._consensus_aggregator(
        candidates=[
            {
                "config": {"lr": 1.0},
                "loss": 2.0,
                "model_vector": torch.tensor([1.0, 0.0]),
            },
            {
                "config": {"lr": 2.0},
                "loss": 1.0,
                "model_vector": torch.tensor([0.9, 0.1]),
            },
            {
                "config": {"lr": 3.0},
                "loss": 0.0,
                "model_vector": torch.tensor([-1.0, 0.0]),
            },
        ]
    )
    assert config == {"lr": 2.0}


from strategies.aFL import aFL
from strategies.FedPSA import FedPSA, FedPSA_Client, FedPSAShared


def test_async_paper_contracts():
    assert issubclass(FedPSA, aFL)
    assert issubclass(FedPSA_Client, FedPSAShared)
    assert FedPSA.compulsory == {"return_diff": True}
    assert FedPSA.optional == {
        "buffer_size": 5,
        "queue_len": 50,
        "sketch_dim": 16,
        "gamma": 5.0,
        "delta": 0.5,
        "calib_size": 32,
    }

    calls = []

    class FakeTrainer:
        def dispatch_one(self, cid, wid):
            calls.append((cid, wid))
            return f"{cid}:{wid}"

    async_server = object.__new__(aFL)
    async_server.trainer = FakeTrainer()
    idle, available, pending = deque([0, 1, 2]), deque([4, 5]), {}
    async_server._dispatch_idle(
        idle=idle,
        available=available,
        pending=pending,
    )
    assert calls == [(4, 0), (5, 1)]
    assert list(idle) == [2] and not available
    assert set(pending.values()) == {(4, 0), (5, 1)}

    projection = FedPSAShared.projection_matrix(
        num_params=2,
        sketch_dim=2,
        seed=7,
    )
    assert torch.equal(
        projection,
        FedPSAShared.projection_matrix(
            num_params=2,
            sketch_dim=2,
            seed=7,
        ),
    )
    weights, similarities = FedPSAShared.similarity_weights(
        sketches=[torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
        global_sketch=torch.tensor([1.0, 0.0]),
        temperature=0.5,
    )
    assert weights[0] > weights[1]
    assert torch.allclose(weights.sum(), torch.tensor(1.0))
    assert torch.equal(similarities, torch.tensor([1.0, 0.0]))

    server = _server(FedPSA)
    server._thermometer = deque(maxlen=3)
    server._initial_magnitude = None
    server.gamma, server.delta = 5.0, 0.5
    server._calibration_x = torch.zeros(1, 1)
    server._calibration_y = torch.zeros(1, 1)
    server._projection = torch.zeros(1, 1)
    server._criterion = torch.nn.MSELoss()
    server.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    packages = OrderedDict(
        {
            0: {
                "model_params_diff": {"weight": torch.tensor([[1.0]])},
                "_psa_s_tilde": torch.tensor([1.0, 0.0]),
            },
            1: {
                "model_params_diff": {"weight": torch.tensor([[3.0]])},
                "_psa_s_tilde": torch.tensor([0.0, 1.0]),
            },
        }
    )
    with patch.object(
        FedPSAShared,
        "sensitivity_sketch",
        return_value=torch.tensor([1.0, 0.0]),
    ):
        server.aggregate_client_updates(packages=packages)
    assert torch.equal(server.public_model_params["weight"], torch.tensor([[-2.0]]))
    assert server._initial_magnitude is None


from strategies.FedDST import FedDST, FedDST_Client
from strategies.FedMef import FedMef, FedMef_Client, FedMefShared
from strategies.FedRTS import FedRTS, FedRTS_Client, FedRTSShared
from strategies.FedSGC import FedSGC, FedSGC_Client, FedSGCShared
from strategies.FedTiny import FedTiny, FedTiny_Client, FedTinyShared
from strategies.PruneFL import PruneFL, PruneFL_Client, PruneFLShared
from strategies.spFL import spFL, spFL_Client, spFLShared


def test_sparse_paper_contracts():
    assert all(
        issubclass(strategy, spFL)
        for strategy in (PruneFL, FedDST, FedTiny, FedMef, FedSGC, FedRTS)
    )
    assert all(
        issubclass(client, spFL_Client)
        for client in (
            PruneFL_Client,
            FedDST_Client,
            FedTiny_Client,
            FedMef_Client,
            FedSGC_Client,
            FedRTS_Client,
        )
    )

    original = {"weight": torch.tensor([True, False])}
    cloned = spFLShared.clone_mask(mask_dict=original)
    cloned["weight"][0] = False
    assert original["weight"][0]

    swapped = spFLShared.swap_mask(
        parameters={"weight": torch.tensor([0.1, 1.0, 0.0, 0.0])},
        gradients={"weight": torch.tensor([100.0, 0.0, 2.0, 3.0])},
        mask_dict={"weight": torch.tensor([True, True, False, False])},
        fraction=0.5,
    )
    assert torch.equal(swapped["weight"], torch.tensor([False, True, False, True]))

    sparse_mean = spFLShared.sparse_weighted_mean(
        models=[
            {"weight": torch.tensor([2.0, 0.0])},
            {"weight": torch.tensor([6.0, 8.0])},
        ],
        masks=[
            {"weight": torch.tensor([True, False])},
            {"weight": torch.tensor([True, True])},
        ],
        weights=[1, 3],
    )
    assert torch.equal(sparse_mean["weight"], torch.tensor([5.0, 8.0]))

    adaptive = PruneFLShared.adaptive_mask(
        parameters={"weight": torch.tensor([10.0, 1.0, 0.0, 0.0])},
        squared_gradients={"weight": torch.tensor([100.0, 1.0, 100.0, 0.0])},
        mask_dict={"weight": torch.tensor([True, True, False, False])},
        max_active=2,
        max_prune_fraction=0.5,
        time_constant=1.0,
    )
    assert torch.equal(adaptive["weight"], torch.tensor([True, False, True, False]))

    five_layers = {f"layer{index}": torch.tensor([True, False]) for index in range(5)}
    assert FedTinyShared.selected_block(
        mask_dict=five_layers,
        current_iter=10,
        delta_T=10,
        num_blocks=5,
    ) == {"layer4"}

    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.copy_(torch.tensor([[3.0, 4.0]]))
    penalty, norm = FedMefShared.extrusion_terms(
        model=model,
        marked={"weight": torch.tensor([0, 1])},
    )
    assert penalty == 25 and norm == 5
    assert (
        FedMefShared.budget_learning_rate(
            initial_lr=0.1,
            scheduled_lr=0.01,
            low_norm=torch.tensor(0.0),
            step=0,
            budget=10,
        )
        == 0.01
    )

    directional = FedSGCShared.directional_mask(
        parameters={"weight": torch.tensor([1.0, 2.0, 0.0, 0.0])},
        gradients={"weight": torch.tensor([0.0, 0.0, 4.0, 3.0])},
        mask_dict={"weight": torch.tensor([True, True, False, False])},
        local_direction={"weight": torch.tensor([-1.0, 1.0, 1.0, -1.0])},
        global_direction={"weight": torch.ones(4)},
        fraction=0.5,
        lambda_param=1.0,
    )
    assert torch.equal(directional["weight"], torch.tensor([False, True, True, False]))
    fallback_directional = FedSGCShared.directional_mask(
        parameters={"weight": torch.tensor([1.0, 2.0, 0.0, 0.0])},
        gradients={"weight": torch.tensor([0.0, 0.0, 4.0, 3.0])},
        mask_dict={"weight": torch.tensor([True, True, False, False])},
        local_direction={"weight": torch.zeros(4)},
        global_direction={"weight": torch.ones(4)},
        fraction=0.5,
        lambda_param=1.0,
    )
    assert fallback_directional["weight"].sum() == 2

    outcomes = FedRTSShared.active_outcome(
        global_parameter=torch.tensor([2.0, 1.0]),
        client_parameters=[
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0, 2.0]),
        ],
        active_indices=torch.tensor([0, 1]),
        core_count=1,
        weights=[1, 1],
        gamma=0.25,
    )
    assert torch.equal(outcomes, torch.tensor([0.75, 0.25], dtype=torch.float64))
    alpha, beta = torch.ones(2), torch.ones(2)
    FedRTSShared.update_posterior(
        alpha=alpha,
        beta=beta,
        indices=torch.tensor([0, 1]),
        outcomes=outcomes,
        evidence_scale=10.0,
    )
    assert torch.equal(alpha, torch.tensor([8.5, 3.5]))
    assert torch.equal(beta, torch.tensor([3.5, 8.5]))


def test_primitive_component_options_are_cli_accessible() -> None:
    import importlib
    import sys

    from utils.options import Options

    missing: dict[str, list[str]] = {}
    for option_name, module_name in Options.COMPONENTS.items():
        module = importlib.import_module(module_name)
        for component in getattr(module, module_name.upper()):
            with patch.object(
                sys,
                "argv",
                ["main.py", f"--{option_name}", component],
            ):
                parsed = vars(Options(root=".").parse_options().args)
            inaccessible = [
                name
                for name, value in module.optional[component].items()
                if type(value) in (bool, int, float, str) and name not in parsed
            ]
            if inaccessible:
                missing[f"{module_name}/{component}"] = inaccessible

    assert not missing, f"Primitive optional values are inaccessible: {missing}"


def test_loss_and_scheduler_integrity() -> None:
    from augs import AUGMENTATIONS
    from losses import CONTEXT_LOSSES, EVAL_LOSSES, LOSSES, evaluation_result
    from losses.EMALE import EMALE
    from losses.ERMSLE import ERMSLE
    from losses.MALE import MALE
    from losses.MQC import MQC
    from losses.msMAPE import msMAPE
    from losses.RMdSPE import RMdSPE
    from losses.RMSE import RMSE
    from losses.RMSLE import RMSLE
    from losses.RMSPE import RMSPE
    from losses.RMSSE import RMSSE
    from losses.RSquared import RSquared
    from losses.sMAPC import sMAPC
    from losses.sMAPE import sMAPE
    from models import CHECKPOINT_DIR
    from scalers import SCALERS
    from scalers.BaseScaler import BaseScaler
    from schedulers import SCHEDULER_MODES, SCHEDULERS, compulsory, optional
    from schedulers.ExpHyperbolicLR import ExpHyperbolicLR
    from schedulers.HyperbolicLR import HyperbolicLR
    from schedulers.StepLR import StepLR
    from strategies.base import SharedMethods
    from topologies import TOPOLOGIES
    from topologies.FullyConnected import FullyConnected
    from topologies.Ring import Ring

    assert AUGMENTATIONS == sorted(AUGMENTATIONS, key=str.casefold)
    assert LOSSES == sorted(LOSSES, key=str.casefold)
    assert EVAL_LOSSES == ["RSquared"]
    assert CONTEXT_LOSSES == ["MQC", "RMSSE", "sMAPC"]
    assert SCHEDULERS == sorted(SCHEDULERS, key=str.casefold)
    assert SCALERS == sorted(SCALERS, key=str.casefold)
    assert TOPOLOGIES == sorted(TOPOLOGIES, key=str.casefold)
    assert CHECKPOINT_DIR == Path(__file__).resolve().parents[1] / "ckpt"
    assert SCHEDULER_MODES == ("batch", "epoch", "iteration")
    assert "ResetStepLR" not in SCHEDULERS
    assert all(params["scheduler_mode"] == "iteration" for params in optional.values())
    assert compulsory["AutoCyclic"]["scheduler_mode"] == "batch"
    assert compulsory["OneCycleLR"]["scheduler_mode"] == "batch"
    assert issubclass(ExpHyperbolicLR, HyperbolicLR)

    means, deviations = BaseScaler.extract_statistics(
        stat={
            "first": {"mean": 1.0, "std": 2.0},
            "second": {"mean": 3.0, "std": 4.0},
        },
        names=("mean", "std"),
    )
    assert means.tolist() == [1.0, 3.0]
    assert deviations.tolist() == [2.0, 4.0]
    assert FullyConnected(num_nodes=3).neighbors == {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1],
    }
    assert Ring(num_nodes=4).neighbors[0] == [3, 1]

    positive = torch.tensor([1.0, 2.0, 4.0])
    for criterion, expected in (
        (MALE(), 0.0),
        (RMSLE(), 0.0),
        (EMALE(), 1.0),
        (ERMSLE(), 1.0),
    ):
        torch.testing.assert_close(
            criterion(input=positive, target=positive),
            torch.tensor(expected),
        )

    for criterion in (RMSE(), RMSLE(), RMSPE(), RMdSPE()):
        prediction = positive.clone().requires_grad_()
        value = criterion(input=prediction, target=positive)
        value.backward()
        torch.testing.assert_close(value, torch.tensor(0.0))
        assert torch.isfinite(prediction.grad).all()

    torch.testing.assert_close(
        sMAPE()(input=torch.zeros(2), target=torch.zeros(2)),
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        msMAPE()(input=torch.tensor([0.0, 1.0]), target=torch.zeros(2)),
        torch.tensor(1000 / 11),
    )
    torch.testing.assert_close(
        RSquared()(input=torch.ones(2), target=torch.ones(2)),
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        RSquared()(input=torch.zeros(2), target=torch.ones(2)),
        torch.tensor(0.0),
    )

    current = torch.tensor([1.0, 2.0])
    previous = torch.tensor([2.0, 1.0])
    torch.testing.assert_close(
        sMAPC()(input=current, target=previous),
        torch.tensor(200 / 3),
    )
    torch.testing.assert_close(
        MQC(quantiles=[0.25, 0.5, 0.75])(
            input=torch.tensor([[[1.0, 2.0, 3.0]]]),
            target=torch.tensor([[[2.0, 1.0, 3.0]]]),
        ),
        torch.tensor(0.25),
    )
    torch.testing.assert_close(
        RMSSE()(
            input=torch.tensor([[[3.0], [5.0]]]),
            target=torch.tensor([[[2.0], [4.0]]]),
            insample=torch.tensor([[[1.0], [2.0], [4.0]]]),
        ),
        torch.tensor(math.sqrt(2 / 5)),
    )
    generic_results = evaluation_result(
        y_pred=torch.tensor([-1.0, 1.0, 3.0]),
        y_true=torch.tensor([-2.0, 2.0, 4.0]),
    )
    assert (
        not {
            "EMALE",
            "ERMSLE",
            "KLDivergence",
            "MALE",
            "RMSLE",
        }
        & generic_results.keys()
    )
    assert all(math.isfinite(value) for value in generic_results.values())

    config = SimpleNamespace(max_epochs=4, upper_bound=2, infimum_lr=0.01)
    for scheduler_type in (HyperbolicLR, ExpHyperbolicLR):
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.SGD(params=[parameter], lr=0.1)
        scheduler = scheduler_type(optimizer=optimizer, configs=config)
        rates = [scheduler.get_last_lr()[0]]
        for _ in range(config.max_epochs - 1):
            optimizer.step()
            scheduler.step()
            rates.append(scheduler.get_last_lr()[0])

        max_iter = config.max_epochs - 1
        upper_bound = config.upper_bound * config.max_epochs
        term0 = math.sqrt(max_iter / upper_bound * (2 - max_iter / upper_bound))
        for iteration, rate in enumerate(rates):
            term = math.sqrt(
                (max_iter - iteration)
                / upper_bound
                * (2 - (max_iter + iteration) / upper_bound)
            )
            expected = (
                0.1 + (0.1 - config.infimum_lr) * (term - term0)
                if scheduler_type is HyperbolicLR
                else 0.1 * math.exp(math.log(0.1 / config.infimum_lr) * (term - term0))
            )
            assert math.isclose(rate, expected)
        assert rates[-1] > config.infimum_lr
        optimizer.step()
        scheduler.step()
        assert math.isclose(scheduler.get_last_lr()[0], rates[-1])

    mode_config = SimpleNamespace(epochs=2, max_epochs=6)
    expected_steps = {"batch": 8, "epoch": 2, "iteration": 6}
    for mode, total_steps in expected_steps.items():
        mode_config.scheduler_mode = mode
        scheduler_config = SharedMethods._scheduler_configs(
            configs=mode_config,
            steps_per_epoch=4,
        )
        assert scheduler_config.max_epochs == total_steps

    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD(params=[parameter], lr=0.1)
    scheduler = StepLR(
        optimizer=optimizer,
        configs=SimpleNamespace(step_size=1, gamma=0.5),
    )
    scheduler.scheduler_mode = "batch"
    SharedMethods.step_scheduler_epoch(scheduler=scheduler)
    assert scheduler.last_epoch == 0
    SharedMethods.step_scheduler_batch(
        scheduler=scheduler,
        batch_data=torch.ones(1),
    )
    assert scheduler.last_epoch == 1
    scheduler.scheduler_mode = "epoch"
    SharedMethods.step_scheduler_batch(
        scheduler=scheduler,
        batch_data=torch.ones(1),
    )
    assert scheduler.last_epoch == 1
    SharedMethods.step_scheduler_epoch(scheduler=scheduler)
    assert scheduler.last_epoch == 2

    root = Path(__file__).parents[1]
    source_paths = [
        *sorted((root / "augs").glob("*.py")),
        *sorted((root / "losses").glob("*.py")),
        *sorted((root / "scalers").glob("*.py")),
        *sorted((root / "topologies").glob("*.py")),
    ]
    source_paths += [
        path
        for path in sorted((root / "schedulers").glob("*.py"))
        if path.name not in {"AutoCyclic.py", "OneCycleLR.py"}
    ]
    for path in source_paths:
        tree = ast.parse(source=path.read_text(encoding="utf-8"))
        for node in tree.body:
            functions = [node] if isinstance(node, ast.FunctionDef) else []
            if isinstance(node, ast.ClassDef):
                functions = [
                    child for child in node.body if isinstance(child, ast.FunctionDef)
                ]
            for function in functions:
                is_method = isinstance(node, ast.ClassDef)
                is_static = any(
                    isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
                    for decorator in function.decorator_list
                )
                arguments = [
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                ]
                if is_method and not is_static:
                    arguments = arguments[1:]
                assert all(argument.annotation is not None for argument in arguments)
                assert function.args.vararg is None or function.args.vararg.annotation
                assert function.args.kwarg is None or function.args.kwarg.annotation
                assert function.returns is not None
