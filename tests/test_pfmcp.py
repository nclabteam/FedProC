import os
import sys
import unittest
from argparse import Namespace

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.PFMCP import PFMCP as PFMCPModel
from strategies.pFL import pFL, pFL_Client
from strategies.PFMCP import (
    PFMCP,
    PFMCP_Client,
    conformal_quantile,
    dynamic_conformal_intervals,
)


class TestPFMCPModel(unittest.TestCase):
    @staticmethod
    def make_config(**overrides) -> Namespace:
        config = Namespace(
            input_len=16,
            output_len=8,
            input_channels=2,
            output_channels=2,
            pfmcp_d_model=8,
            pfmcp_n_heads=2,
            pfmcp_encoder_layers=1,
            pfmcp_ff_dim=16,
            pfmcp_decoder_hidden=12,
            pfmcp_dropout=0.0,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_global_and_personalized_forward_shapes(self) -> None:
        config = self.make_config()
        model = PFMCPModel(config)
        inputs = torch.randn(3, config.input_len, config.input_channels)

        global_output = model(inputs)
        model.initialize_personalization()
        model.set_mode("personalized")
        personal_output = model(inputs)

        expected_shape = (
            3,
            config.output_len,
            config.output_channels,
        )
        self.assertEqual(global_output.shape, expected_shape)
        self.assertEqual(personal_output.shape, expected_shape)
        self.assertTrue(torch.allclose(global_output, personal_output))

    def test_parameter_partition_is_complete_and_disjoint(self) -> None:
        model = PFMCPModel(self.make_config())
        regular = set(model.regular_parameter_names())
        personal = set(model.personal_parameter_names())
        all_parameters = {name for name, _ in model.named_parameters()}

        self.assertFalse(regular & personal)
        self.assertEqual(regular | personal, all_parameters)

    def test_personalization_freezes_global_modules(self) -> None:
        model = PFMCPModel(self.make_config())
        model.set_trainable_phase("personalization")
        regular = set(model.regular_parameter_names())
        personal = set(model.personal_parameter_names())

        for name, parameter in model.named_parameters():
            if name in regular:
                self.assertFalse(parameter.requires_grad)
            if name in personal:
                self.assertTrue(parameter.requires_grad)

    def test_invalid_head_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PFMCPModel(
                self.make_config(
                    pfmcp_d_model=10,
                    pfmcp_n_heads=3,
                )
            )


class TestPFMCPStrategy(unittest.TestCase):
    def test_strategy_uses_personalized_fl_base(self) -> None:
        self.assertTrue(issubclass(PFMCP, pFL))
        self.assertTrue(issubclass(PFMCP_Client, pFL_Client))

    def test_strategy_requires_pfmcp_model(self) -> None:
        self.assertEqual(PFMCP.compulsory["model"], "PFMCP")
        self.assertEqual(PFMCP.compulsory["optimizer"], "SGD")
        self.assertEqual(PFMCP.compulsory["loss"], "MSE")

    def test_server_sends_only_global_parameters(self) -> None:
        server = object.__new__(PFMCP)
        server.public_model_params = {"encoder.weight": torch.ones(2)}
        server.clients_personal_model_params = {0: {"gate.weight": torch.ones(1)}}
        server.client_optimizer_states = {0: {"private": True}}
        server.client_scheduler_states = {0: {"private": True}}
        server.current_iter = 4

        for phase in ("federated", "personalization"):
            server.pfmcp_phase = phase
            package = server.package(0)
            self.assertEqual(package["__wire__"], ("regular_model_params",))
            self.assertEqual(package["personal_model_params"], {})
            self.assertEqual(package["pfmcp_phase"], phase)

    def test_client_uploads_only_federated_model_and_sample_count(self) -> None:
        config = TestPFMCPModel.make_config()
        model = PFMCPModel(config)
        client = object.__new__(PFMCP_Client)
        client.model = model
        client.regular_params_name = model.regular_parameter_names()
        client.personal_params_name = model.personal_parameter_names()
        client.optimizer = torch.optim.SGD(
            [
                parameter
                for name, parameter in model.named_parameters()
                if name in client.regular_params_name
            ],
            lr=0.005,
        )
        client.scheduler = torch.optim.lr_scheduler.LambdaLR(
            client.optimizer,
            lambda _: 1.0,
        )
        client.id = 0
        client.train_samples = 17
        client.return_diff = False

        client.pfmcp_phase = "federated"
        federated = client.package()
        self.assertEqual(
            federated["__wire__"],
            ("regular_model_params", "score"),
        )
        self.assertEqual(federated["personal_model_params"], {})
        self.assertEqual(federated["score"], 17)

        client.pfmcp_phase = "personalization"
        personalized = client.package()
        self.assertEqual(personalized["__wire__"], ())
        self.assertEqual(personalized["regular_model_params"], {})
        self.assertTrue(personalized["personal_model_params"])


class TestPFMCPConformalPrediction(unittest.TestCase):
    def test_finite_sample_corrected_quantile_uses_ceiling_rank(self) -> None:
        scores = torch.tensor(
            [
                [[1.0], [10.0]],
                [[2.0], [20.0]],
                [[3.0], [30.0]],
                [[4.0], [40.0]],
            ]
        )
        # ceil((4 + 1) * (1 - 0.4)) = 3.
        result = conformal_quantile(scores, alpha=0.4)
        self.assertTrue(torch.equal(result, torch.tensor([[3.0], [30.0]])))

    def test_quantile_clips_to_largest_score_for_small_calibration_set(self) -> None:
        scores = torch.tensor([[[1.0]], [[7.0]]])
        result = conformal_quantile(scores, alpha=0.1)
        self.assertTrue(torch.equal(result, torch.tensor([[7.0]])))

    def test_quantile_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            conformal_quantile(torch.empty(0, 2, 1), alpha=0.1)
        with self.assertRaises(ValueError):
            conformal_quantile(torch.ones(2, 2, 1), alpha=0.0)

    def test_dynamic_scores_are_fifo_updated_after_current_interval(self) -> None:
        calibration_prediction = torch.zeros(3, 1, 1)
        calibration_target = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
        test_prediction = torch.zeros(3, 1, 1)
        test_target = torch.tensor([[[10.0]], [[20.0]], [[30.0]]])

        intervals = list(
            dynamic_conformal_intervals(
                calibration_prediction,
                calibration_target,
                test_prediction,
                test_target,
                alpha=0.5,
                delay=1,
            )
        )
        upper_bounds = [float(interval[2].item()) for interval in intervals]
        # t=0 and t=1 use {1,2,3}; after t=1, score 1 is replaced by
        # the now-observable test residual 10, so t=2 uses {2,3,10}.
        self.assertEqual(upper_bounds, [2.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
