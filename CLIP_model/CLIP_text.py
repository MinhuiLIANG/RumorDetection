from transformers import CLIPTokenizer, ChineseCLIPTextModel, ChineseCLIPTextConfig
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device=', device)

configuration = ChineseCLIPTextConfig(num_hidden_layers=6)

pretrained = ChineseCLIPTextModel(configuration)
pretrained.to(device)

for param in pretrained.parameters():
    param.requires_grad_(False)


class ClipTextFeat(torch.nn.Module):

    def __init__(self):
        super(ClipTextFeat, self).__init__()

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        return out.last_hidden_state
