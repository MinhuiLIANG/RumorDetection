import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device=', device)


class RumorDetectionModel(torch.nn.Module):

    def __init__(self):
        super(RumorDetectionModel, self).__init__()

        self.text_projection = nn.Sequential(nn.Linear(768 * 2, 128), nn.BatchNorm1d(128))

        self.img_projection = nn.Sequential(nn.Linear(768 * 2, 128), nn.BatchNorm1d(128))

        self.fused_projection = nn.Sequential(nn.Linear(768 * 2, 128), nn.BatchNorm1d(128))

        self.cross_att = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        self.self_att = nn.MultiheadAttention(embed_dim=128 + 4 + 100, num_heads=8)

        self.fc1 = nn.Sequential(nn.Linear(128 + 4 + 100, 1))

        self.drop_out = nn.Dropout(0.2)

    def forward(self, bert_text, vit_img, clip_text, clip_img, clip_sim, ntm, bert_input_ids, bert_attention_mask, bert_token_type_ids, clip_input_ids, clip_attention_mask, clip_token_type_ids, vit_imgs_tensor, clip_imgs_tensor, clip_sim_feat, bow_tensor, senti_vec):
        #text models:
        bert_tensor = bert_text(input_ids=bert_input_ids, attention_mask=bert_attention_mask, token_type_ids=bert_token_type_ids) #[16,768]
        clip_text_tensor = clip_text(input_ids=clip_input_ids, attention_mask=clip_attention_mask, token_type_ids=clip_token_type_ids) #[16,768]

        #image models:
        vit_tensor = vit_img(vit_imgs_tensor) #[16,768]
        clip_image_tensor = clip_img(clip_imgs_tensor) #[16,768]

        #fused models:
        sim_tensor = clip_sim(clip_sim_feat) #[16,16]
        sim_weight, _ = sim_tensor.max(1) #[16,1]
        sim_weight = sim_weight.reshape((16, 1))
        sim_weight = sim_weight.expand(16, 128)

        #weights:
        unimodal_weight = 1

        #ntm
        inputs_hat, mean, log_var, z = ntm(bow_tensor) #[16,40535]

        #senti
        sentiment = senti_vec #[16,4]

        #concat
        text_feat = torch.cat((bert_tensor, clip_text_tensor), dim=1) #[16,768*2]
        img_feat = torch.cat((vit_tensor, clip_image_tensor), dim=1) #[16,768*2]
        fused_feat = torch.cat((clip_text_tensor, clip_image_tensor), dim=1) #[16,768*2]

        #projection
        text_feat = F.relu(self.text_projection(text_feat)) #[16,128]
        img_feat = F.relu(self.img_projection(img_feat)) #[16,128]
        fused_feat = F.relu(self.fused_projection(fused_feat)) #[16,128]

        #get_weighted_fused
        fused_weighted_feat = fused_feat * sim_weight #[16,128]

        #Transpose_n_reshape
        text_feat = text_feat.reshape((16, 1, 128))
        img_feat = img_feat.reshape((16, 1, 128))
        text_feat_trans = torch.transpose(text_feat, 0, 1)
        img_feat_trans = torch.transpose(img_feat, 0, 1)

        #text_image_cross_attention
        txt_attn_feat, txt_attn_weights = self.cross_att(text_feat_trans, img_feat_trans, img_feat_trans)
        img_attn_feat, img_attn_weights = self.cross_att(img_feat_trans, text_feat_trans, text_feat_trans)

        #Transpose
        txt_attn_feat = torch.transpose(txt_attn_feat, 0, 1)
        img_attn_feat = torch.transpose(img_attn_feat, 0, 1)
        txt_attn_feat = txt_attn_feat.squeeze() #[16, 128]
        img_attn_feat = img_attn_feat.squeeze() #[16, 128]

        #weighted_sum
        feat = fused_weighted_feat * sim_weight + txt_attn_feat * unimodal_weight + img_attn_feat * unimodal_weight #[16,128]

        #integrate_with_senti_n_topic
        feat_inte = torch.cat((feat, sentiment, z), dim=1) #[16, 128 + 4 + 100]

        #transform_n_reshape
        feat_inte = feat_inte.reshape((16, 1, 128 + 4 + 100))
        feat_inte = torch.transpose(feat_inte, 0, 1)

        #self_attentoin
        feat_attn_inte, feat_attn_inte_weights = self.self_att(feat_inte, feat_inte, feat_inte)

        #Transpose_n_squeeze
        feat_attn_inte = torch.transpose(feat_attn_inte, 0, 1) #[16, 128 + 4 + 100]
        feat_attn_inte = feat_attn_inte.squeeze()

        #MLP
        out = F.sigmoid(self.fc1(feat_attn_inte))

        return out, mean, log_var, inputs_hat

