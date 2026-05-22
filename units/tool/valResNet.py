import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

torch.hub.set_dir("LightDet/units/model/resnet")

resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
backbone = nn.Sequential(*list(resnet.children())[:-2])
backbone.eval()

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    y = backbone(x)

print(y.shape)
print(torch.hub.get_dir())