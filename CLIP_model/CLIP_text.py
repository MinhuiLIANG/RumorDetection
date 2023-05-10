from transformers import CLIPTokenizer, ChineseCLIPTextModel, ChineseCLIPTextConfig
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


pretrained = ChineseCLIPTextModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
pretrained.to(device)

for param in pretrained.parameters():
    param.requires_grad_(False)


class ClipTextFeat(torch.nn.Module):

    def __init__(self):
        super(ClipTextFeat, self).__init__()

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        return out.last_hidden_state[:, 0]
