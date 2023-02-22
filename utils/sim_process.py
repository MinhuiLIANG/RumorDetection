import requests
import torch
from PIL import Image
from transformers import ChineseCLIPProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def img_text_process(txts, imgs):
    #img = Image.open(img_path)
    processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    inputs = processor(text=txts, images=imgs, return_tensors="pt", padding=True)
    return inputs
