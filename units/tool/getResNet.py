"""Download ImageNet ResNet-50 into the existing model cache."""
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
    directory=ROOT/'units/model/resnet/checkpoints'
    if not directory.is_dir():raise FileNotFoundError('Create the documented existing cache before downloading')
    torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        model_dir=str(directory),check_hash=True,map_location='cpu')
