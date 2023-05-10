import torch
from transformers import BertModel, BertConfig

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

pretrained = BertModel.from_pretrained('bert-base-chinese')
pretrained.to(device)

for param in pretrained.parameters():
    param.requires_grad_(False)


class Bert_pretrain(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return out.last_hidden_state[:, 0]

