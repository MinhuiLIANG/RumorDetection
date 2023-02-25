from transformers import ViTModel, ViTConfig
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device=', device)

configuration = ViTConfig(num_hidden_layers=6)

pretrained = ViTModel(configuration)
pretrained.to(device)

for param in pretrained.parameters():
    param.requires_grad_(False)


class ViTImgFeat(torch.nn.Module):

    def __init__(self):
        super(ViTImgFeat, self).__init__()

    def forward(self, Img):
        with torch.no_grad():
            out = pretrained(Img)
        return out.last_hidden_state

img = torch.ones(4,1,3,224,224).long().to(device)
img = img.squeeze()

#试算
model = ViTImgFeat()
print(model(img))
