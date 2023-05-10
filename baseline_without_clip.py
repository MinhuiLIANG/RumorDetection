import torch
import torch.nn as nn
import torch.nn.functional as F

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


class BaselineModel_WC(torch.nn.Module):

    def __init__(self):
        super(BaselineModel_WC, self).__init__()

        self.text_projection_1 = nn.Sequential(nn.Linear(768 , 512), nn.BatchNorm1d(512))
        self.text_projection_2 = nn.Sequential(nn.Linear(512, 128), nn.BatchNorm1d(128))

        self.img_projection_1 = nn.Sequential(nn.Linear(768, 512), nn.BatchNorm1d(512))
        self.img_projection_2 = nn.Sequential(nn.Linear(512, 128), nn.BatchNorm1d(128))

        self.fused_projection_1 = nn.Sequential(nn.Linear(768, 512), nn.BatchNorm1d(512))
        self.fused_projection_2 = nn.Sequential(nn.Linear(512, 128), nn.BatchNorm1d(128))

        self.cross_att = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        self.ahead_att = nn.MultiheadAttention(embed_dim=128 + 4, num_heads=4)
        self.self_att = nn.MultiheadAttention(embed_dim=128 + 4 + 100, num_heads=8)

        self.mlp = nn.Sequential(nn.Linear(100 + 4, 128), nn.BatchNorm1d(128))

        self.fc1 = nn.Sequential(nn.Linear(128 , 16), nn.BatchNorm1d(16))
        self.fc2 = nn.Sequential(nn.Linear(64, 1))
        self.fc3 = nn.Sequential(nn.Linear(16, 2))

        self.drop_out = nn.Dropout(0.2)

    def forward(self, bert_text, vit_img, clip_text, clip_img, clip_sim, ntm, bert_input_ids, bert_attention_mask,
                bert_token_type_ids, clip_input_ids, clip_attention_mask, clip_token_type_ids, vit_imgs_tensor,
                clip_imgs_tensor, clip_sim_feat, bow_tensor, senti_vec):
        # text models:
        bert_tensor = bert_text(input_ids=bert_input_ids, attention_mask=bert_attention_mask,
                                token_type_ids=bert_token_type_ids)  # [16,768]
        clip_text_tensor = clip_text(input_ids=clip_input_ids, attention_mask=clip_attention_mask,
                                     token_type_ids=clip_token_type_ids)  # [16,768]

        # image models:
        vit_tensor = vit_img(vit_imgs_tensor)  # [16,768]
        clip_image_tensor = clip_img(clip_imgs_tensor)  # [16,768]

        sim_tensor = clip_sim(clip_sim_feat) #[16,16]
        sim_weight, _ = sim_tensor.max(1) #[16,1]
        sim_weight = sim_weight.reshape((64, 1))
        sim_weight = sim_weight.expand(64, 128)

        # concat
        '''
        text_feat = torch.cat((bert_tensor, clip_text_tensor), dim=1)  # [16,768*2]
        img_feat = torch.cat((vit_tensor, clip_image_tensor), dim=1)  # [16,768*2]
        fused_feat = torch.cat((clip_text_tensor, clip_image_tensor), dim=1)  # [16,768*2]
        '''

        # projection
        text_feat = F.relu(self.text_projection_1(bert_tensor))  # [16,128]
        text_feat = self.drop_out(text_feat)
        text_feat = F.relu(self.text_projection_2(text_feat))

        img_feat = F.relu(self.img_projection_1(vit_tensor))  # [16,128]
        img_feat = self.drop_out(img_feat)
        img_feat = F.relu(self.img_projection_2(img_feat))

        fused_feat = F.relu(self.fused_projection_1(clip_text_tensor))  # [16,128]
        fused_feat = self.drop_out(fused_feat)
        fused_feat = F.relu(self.fused_projection_2(fused_feat))

        # weighted_sum
        feat = text_feat + img_feat  # [16,128]

        # MLP
        out = F.relu(self.fc1(feat))
        out = self.drop_out(out)
        # out = F.sigmoid(self.fc2(out))
        out = self.fc3(out)

        return out, feat