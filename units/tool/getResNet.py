import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import os
from pathlib import Path
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
torch.hub.set_dir(f"{PROJECT_ROOT}/LightDet/units/model/resnet")

resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
backbone = nn.Sequential(*list(resnet.children())[:-2])
