import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import ChineseCLIPImageProcessor, AutoImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vis = True
vis_row = 4

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])


def img_transform(img_rgb, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_rgb)
    return img_t


def img_2_tensor(img_path):
    img_rgb = Image.open(img_path).convert('RGB')
    img_tensor = img_transform(img_rgb, inference_transform)
    img_tensor.unsqueeze_(0)
    return img_tensor


def clip_process(imgs):
    image_processor = ChineseCLIPImageProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    inputs = image_processor(imgs, return_tensors="pt")
    return inputs


def vit_process(imgs):
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    inputs = image_processor(imgs, return_tensors="pt")
    return inputs

