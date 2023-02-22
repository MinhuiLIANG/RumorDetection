import torch
import torchvision.transforms as transforms
import torchvision.models as models

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


def get_model():
    resnet50_model = models.resnet50(pretrained=True)
    resnet50_model.to(device)
    resnet50_model.eval()
    return resnet50_model

model = get_model()
#同样要移动到cuda
model.to(device)

#虚拟一批数据,需要把所有的数据都移动到cuda上
imgs = torch.ones(16,1,3,224,224).float().to(device)
imgs = imgs.squeeze()
labels = torch.ones(16).long().to(device)

#试算
print(model(imgs).shape)