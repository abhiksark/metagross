# examples/docker/tests/test_workloads.py
"""GPU correctness tests for the Dockerized PyTorch workloads."""
import math
import unittest

import torch

from workloads import basic_cnn
from workloads import basic_decoder
from workloads import basic_tensor_ops
from workloads import basic_vit
from workloads import complex_pipeline
from workloads import training_step


def _parameters_changed(before: list[torch.Tensor], model: torch.nn.Module) -> bool:
    return any(
        not torch.equal(old, current.detach())
        for old, current in zip(before, model.parameters())
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA-capable PyTorch runtime")
class WorkloadCorrectnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda")

    def setUp(self) -> None:
        torch.manual_seed(101)

    def tearDown(self) -> None:
        torch.cuda.empty_cache()

    def test_basic_tensor_math_and_download(self) -> None:
        left = torch.tensor([[1.0, -2.0], [3.0, 4.0]], device=self.device)
        right = torch.tensor([[2.0, 1.0], [-1.0, 3.0]], device=self.device)
        expected = torch.relu(left @ right)

        result = basic_tensor_ops.compute(left, right)

        torch.testing.assert_close(result, expected)
        self.assertAlmostEqual(
            basic_tensor_ops.download_checksum(result), expected.sum().item()
        )

    def test_mlp_training_updates_parameters(self) -> None:
        model, optimizer = training_step.build_model(self.device)
        features, labels = training_step.make_batch(self.device)
        with torch.no_grad():
            self.assertEqual(tuple(model(features).shape), (64, 10))
        before = [parameter.detach().clone() for parameter in model.parameters()]

        loss = training_step.train_step(
            model, optimizer, features, labels
        )

        self.assertTrue(math.isfinite(loss))
        self.assertTrue(_parameters_changed(before, model))

    def test_cnn_shapes_training_and_evaluation(self) -> None:
        model, optimizer = basic_cnn.build_model(self.device)
        images, labels = basic_cnn.prepare_batch(self.device)
        with torch.no_grad():
            self.assertEqual(tuple(model(images).shape), (32, 10))
        before = [parameter.detach().clone() for parameter in model.parameters()]

        loss = basic_cnn.train_step(model, optimizer, images, labels)
        predictions = basic_cnn.evaluate(model, images)

        self.assertTrue(math.isfinite(loss))
        self.assertTrue(_parameters_changed(before, model))
        self.assertEqual(len(predictions), 8)
        self.assertTrue(all(0 <= prediction < 10 for prediction in predictions))

    def test_vit_shapes_training_and_classification(self) -> None:
        model, optimizer = basic_vit.build_model(self.device)
        images, labels = basic_vit.prepare_image_batch(self.device)
        with torch.no_grad():
            self.assertEqual(tuple(model(images).shape), (16, 10))
        before = [parameter.detach().clone() for parameter in model.parameters()]

        loss = basic_vit.train_vit_step(model, optimizer, images, labels)
        predictions = basic_vit.classify(model, images)

        self.assertTrue(math.isfinite(loss))
        self.assertTrue(_parameters_changed(before, model))
        self.assertEqual(len(predictions), 4)
        self.assertTrue(all(0 <= prediction < 10 for prediction in predictions))

    def test_decoder_is_causal_and_extends_prompt(self) -> None:
        model = basic_decoder.build_decoder(self.device)
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=self.device)
        changed_future = prompt.clone()
        changed_future[0, -1] = 99

        with torch.no_grad():
            original_logits = model(prompt)
            changed_logits = model(changed_future)

        self.assertEqual(tuple(original_logits.shape), (1, 5, 256))
        torch.testing.assert_close(
            original_logits[:, :-1], changed_logits[:, :-1], rtol=1e-5, atol=1e-5
        )

        prefill_logits = basic_decoder.prefill(model, prompt)
        generated = basic_decoder.decode_tokens(model, prompt, token_count=3)

        self.assertEqual(tuple(prefill_logits.shape), (1, 256))
        self.assertEqual(tuple(generated.shape), (1, prompt.shape[1] + 3))
        self.assertTrue(torch.equal(generated[:, : prompt.shape[1]], prompt))
        self.assertTrue(torch.all((generated >= 0) & (generated < 256)).item())

    def test_complex_pipeline_preserves_data_and_updates_model(self) -> None:
        raw_batch = complex_pipeline.load_cpu_batch(seed=303)
        batch = complex_pipeline.preprocess_cpu_batch(raw_batch)
        self.assertTrue(batch.images.is_pinned())
        self.assertTrue(batch.labels.is_pinned())
        self.assertLessEqual(batch.images.abs().max().item(), 1.0)

        model, optimizer, transfer_stream = complex_pipeline.build_pipeline(
            self.device
        )
        images, labels = complex_pipeline.upload_batch(
            batch, self.device, transfer_stream
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(images.cpu(), batch.images)
        self.assertTrue(torch.equal(labels.cpu(), batch.labels))

        with torch.no_grad():
            self.assertEqual(tuple(model(images).shape), (24, 10))
        before = [parameter.detach().clone() for parameter in model.parameters()]

        loss = complex_pipeline.forward_and_backward(model, images, labels)
        complex_pipeline.optimizer_step(optimizer)
        accuracy = complex_pipeline.evaluate_pipeline(model, images, labels)

        self.assertTrue(math.isfinite(loss))
        self.assertTrue(_parameters_changed(before, model))
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
