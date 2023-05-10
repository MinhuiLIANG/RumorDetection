from transformers import ViTModel, ViTConfig
import torch

import os
import numpy as np
import random

seed_value = 3407

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device=', device)

pretrained = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
pretrained.to(device)

for param in pretrained.parameters():
    param.requires_grad_(False)


class ViTImgFeat(torch.nn.Module):

    def __init__(self):
        super(ViTImgFeat, self).__init__()

    def forward(self, Img):
        with torch.no_grad():
            out = pretrained(Img)
        return out.last_hidden_state[:, 0]
