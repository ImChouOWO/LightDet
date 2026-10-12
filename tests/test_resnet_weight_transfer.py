import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from units.model._train import load_weights_only


class TransferModel(nn.Module):
    def __init__(self, image_channels=2, head_channels=2):
        super().__init__()
        self.img_model = nn.Linear(image_channels, 2)
        self.head = nn.Linear(head_channels, 1)


class ResNetWeightTransferTests(unittest.TestCase):
    def checkpoint(self, root):
        old = TransferModel()
        with torch.no_grad():
            for parameter in old.parameters():
                parameter.fill_(3.)
        path = Path(root)/'weights.pt'
        torch.save(dict(ema=old.state_dict(), epoch=7, best_metric=.45), path)
        return path, old

    def test_transfer_restores_heads_but_preserves_new_vision_and_ema(self):
        with tempfile.TemporaryDirectory() as root:
            path, old = self.checkpoint(root)
            model = TransferModel(image_channels=3)
            vision = {key: value.clone() for key, value in model.img_model.state_dict().items()}
            ema = SimpleNamespace(ema=TransferModel(image_channels=3), num_updates=10)
            info = load_weights_only(str(path), model, ema, torch.device('cpu'),
                                     exclude_prefixes=('img_model.',))
            for key, value in vision.items():
                torch.testing.assert_close(model.img_model.state_dict()[key], value)
            for key, value in old.head.state_dict().items():
                torch.testing.assert_close(model.head.state_dict()[key], value)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(ema.ema.state_dict()[key], value)
            self.assertEqual(ema.num_updates, 0)
            self.assertEqual(info['excluded_prefixes'], ['img_model.'])

    def test_mismatch_outside_explicit_image_exclusion_fails(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.checkpoint(root)
            with self.assertRaises(RuntimeError):
                load_weights_only(str(path), TransferModel(image_channels=3, head_channels=3),
                                  None, torch.device('cpu'), exclude_prefixes=('img_model.',))

    def test_default_loading_remains_strict_and_restores_original_model(self):
        with tempfile.TemporaryDirectory() as root:
            path, old = self.checkpoint(root)
            model = TransferModel()
            load_weights_only(str(path), model, None, torch.device('cpu'))
            for key, value in old.state_dict().items():
                torch.testing.assert_close(model.state_dict()[key], value)
            with self.assertRaises(RuntimeError):
                load_weights_only(str(path), TransferModel(image_channels=3), None, torch.device('cpu'))

    def test_invalid_or_unmatched_exclusions_fail(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.checkpoint(root)
            for prefix in ('', 'img_model', 'unknown.'):
                with self.assertRaises(ValueError):
                    load_weights_only(str(path), TransferModel(), None, torch.device('cpu'),
                                      exclude_prefixes=(prefix,))


if __name__ == '__main__':
    unittest.main()
