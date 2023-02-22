import requests
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device=', device)

pretrained = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
pretrained.to(device)

for param in pretrained.parameters():
    param.requires_grad_(False)


class ClipSim(torch.nn.Module):

    def __init__(self):
        super(ClipSim, self).__init__()

    def forward(self, inputs):
        with torch.no_grad():
            out = pretrained(**inputs)
            logits_per_image = out.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        return probs
