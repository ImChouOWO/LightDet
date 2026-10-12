import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch import nn

from units.model.tool.config import cfg_to_args, load_model_config, load_train_config
from units.tool.card import BackBone, ResNet50Backbone


class ResNetBackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(49)
        with patch("torch.hub.load_state_dict_from_url") as download:
            cls.backbone = BackBone(
                backbone_name="resnet50",
                weights=None,
                norm_eval=True,
                hidden_dim=256,
                projector_layers=3,
                token_grids=((24, 24), (12, 12), (10, 10)),
            )
            download.assert_not_called()

    def setUp(self):
        self.backbone.eval()
        self.backbone.zero_grad(set_to_none=True)

    def test_resnet_output_preserves_three_scales(self):
        self.assertEqual(self.backbone.output_channels, (512, 1024, 2048))
        with torch.no_grad():
            for size in (64, 128):
                with self.subTest(size=size):
                    maps = self.backbone.resnet(torch.rand(1, 3, size, size))
                    self.assertEqual(tuple(maps), ("c3", "c4", "c5"))
                    for name, channels, stride in (
                        ("c3", 512, 8), ("c4", 1024, 16), ("c5", 2048, 32)
                    ):
                        self.assertEqual(
                            tuple(maps[name].shape), (1, channels, size // stride, size // stride)
                        )

    def test_fpn_and_projector_keep_localization_and_roi_interfaces(self):
        with torch.no_grad():
            tokens, maps = self.backbone(torch.rand(1, 3, 128, 128), return_fpn=True)
        self.assertEqual(tuple(tokens.shape), (1, 820, 256))
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertEqual(tuple(maps), ("c3", "c4", "c5"))
        for name, stride in (("c3", 8), ("c4", 16), ("c5", 32)):
            self.assertEqual(tuple(maps[name].shape), (1, 256, 128 // stride, 128 // stride))
            self.assertTrue(torch.isfinite(maps[name]).all())

    def test_rgb_is_normalized_exactly_once_before_first_convolution(self):
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        observed = []
        hook = self.backbone.resnet.stem[0].register_forward_pre_hook(
            lambda module, inputs: observed.append(inputs[0].detach().clone())
        )
        try:
            with torch.no_grad():
                self.backbone(mean.expand(1, 3, 64, 64))
                self.backbone((mean + 0.5 * std).expand(1, 3, 64, 64))
        finally:
            hook.remove()
        self.assertEqual(len(observed), 2)
        torch.testing.assert_close(observed[0], torch.zeros_like(observed[0]), atol=1e-6, rtol=0)
        torch.testing.assert_close(observed[1], torch.full_like(observed[1], 0.5), atol=1e-6, rtol=0)

    def test_train_freezes_bn_statistics_but_keeps_affine_and_convolution_gradients(self):
        self.backbone.train()
        batch_norms = [module for module in self.backbone.resnet.modules()
                       if isinstance(module, nn.BatchNorm2d)]
        previous = [(module.running_mean.clone(), module.running_var.clone(),
                     module.num_batches_tracked.clone()) for module in batch_norms]
        self.assertTrue(self.backbone.training)
        self.assertTrue(self.backbone.resnet.training)
        self.assertTrue(all(not module.training for module in batch_norms))
        self.assertTrue(all(module.weight.requires_grad and module.bias.requires_grad
                            for module in batch_norms))
        tokens, maps = self.backbone(torch.rand(2, 3, 64, 64), return_fpn=True)
        loss = tokens.square().mean() + sum(feature.square().mean() for feature in maps.values())
        loss.backward()
        for module, (mean, variance, tracked) in zip(batch_norms, previous):
            torch.testing.assert_close(module.running_mean, mean, atol=0, rtol=0)
            torch.testing.assert_close(module.running_var, variance, atol=0, rtol=0)
            torch.testing.assert_close(module.num_batches_tracked, tracked, atol=0, rtol=0)
        for parameter in (self.backbone.resnet.stem[0].weight,
                          self.backbone.resnet.stem[1].weight,
                          self.backbone.fpn.inner_blocks[0][0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_norm_eval_false_allows_batch_statistics_to_update(self):
        model = ResNet50Backbone(weights=None, norm_eval=False).train()
        bn = model.stem[1]
        previous_mean = bn.running_mean.clone()
        with torch.no_grad():
            model(torch.rand(2, 3, 64, 64))
        self.assertTrue(bn.training)
        self.assertEqual(bn.num_batches_tracked.item(), 1)
        self.assertFalse(torch.equal(bn.running_mean, previous_mean))

    def test_default_bottlenet_preserves_state_keys_and_outputs(self):
        torch.manual_seed(17)
        default = BackBone(hidden_dim=256).eval()
        torch.manual_seed(17)
        explicit = BackBone(backbone_name="bottlenet", hidden_dim=256).eval()
        self.assertEqual(default.output_channels, (192, 384, 768))
        self.assertFalse(hasattr(default, "resnet"))
        default_state, explicit_state = default.state_dict(), explicit.state_dict()
        self.assertEqual(tuple(default_state), tuple(explicit_state))
        self.assertEqual({key.split(".")[0] for key in default_state},
                         {"bottle_net", "fpn", "img_projector"})
        for key in default_state:
            torch.testing.assert_close(default_state[key], explicit_state[key], atol=0, rtol=0)
        image = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            default_tokens, default_maps = default(image, return_fpn=True)
            explicit_tokens, explicit_maps = explicit(image, return_fpn=True)
        torch.testing.assert_close(default_tokens, explicit_tokens, atol=0, rtol=0)
        for name in default_maps:
            torch.testing.assert_close(default_maps[name], explicit_maps[name], atol=0, rtol=0)


class BackboneConfigTests(unittest.TestCase):
    def setUp(self):
        self.original = load_model_config()

    def load_config(self, backbone):
        config = copy.deepcopy(self.original)
        config["model"]["backbone"] = backbone
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.yaml"
            path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            return load_model_config(str(path))

    def test_new_resnet_and_existing_bottlenet_configs_build_args(self):
        train = load_train_config()
        old_args = cfg_to_args(self.original, train)
        explicit = dict(self.original["model"]["backbone"], name="bottlenet")
        explicit_args = cfg_to_args(self.load_config(explicit), train)
        self.assertEqual(old_args.backbone_config, explicit_args.backbone_config)
        new = {"name": "resnet50", "in_channels": 3, "weights": None, "norm_eval": True}
        new_args = cfg_to_args(self.load_config(new), train)
        self.assertEqual(new_args.backbone_config, new)
        self.assertEqual(new_args.hidden_dim, old_args.hidden_dim)

    def test_unknown_backbone_and_invalid_resnet_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported backbone"):
            BackBone(backbone_name="resnet18", weights=None)
        valid = {"name": "resnet50", "in_channels": 3, "weights": None, "norm_eval": True}
        for update, exception in (
            ({"name": "resnet18"}, ValueError),
            ({"in_channels": 1}, ValueError),
            ({"weights": "UNKNOWN"}, ValueError),
            ({"norm_eval": "true"}, TypeError),
        ):
            with self.subTest(update=update), self.assertRaises(exception):
                self.load_config(dict(valid, **update))


if __name__ == "__main__":
    unittest.main()
