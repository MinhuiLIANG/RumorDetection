import torch
import torch.nn as nn
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device=', device)


class RumorDetectionModel(torch.nn.Module):

    def __init__(self):
        super(RumorDetectionModel, self).__init__()

        self.text_projection_1 = nn.Sequential(nn.Linear(768 * 2, 512), nn.BatchNorm1d(512))
        self.text_projection_2 = nn.Sequential(nn.Linear(512, 128), nn.BatchNorm1d(128))

        self.img_projection_1 = nn.Sequential(nn.Linear(768 * 2, 512), nn.BatchNorm1d(512))
        self.img_projection_2 = nn.Sequential(nn.Linear(512, 128), nn.BatchNorm1d(128))

        self.fused_projection_1 = nn.Sequential(nn.Linear(768 * 2, 512), nn.BatchNorm1d(512))
        self.fused_projection_2 = nn.Sequential(nn.Linear(512, 128), nn.BatchNorm1d(128))

        self.cross_att = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        self.ahead_att = nn.MultiheadAttention(embed_dim=128 + 4, num_heads=4)
        self.self_att = nn.MultiheadAttention(embed_dim=128 + 4 + 100, num_heads=8)
        self.pre_att = nn.MultiheadAttention(embed_dim=768*2, num_heads=8)

        self.mlp = nn.Sequential(nn.Linear(100 + 4, 128), nn.BatchNorm1d(128))

        self.fc1 = nn.Sequential(nn.Linear(128 + 4 + 100, 64), nn.BatchNorm1d(64))
        self.fc2 = nn.Sequential(nn.Linear(64, 1))
        self.fc3 = nn.Sequential(nn.Linear(64, 2))

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
        sim_weight = sim_weight.reshape((64, 1))
        sim_weight = sim_weight.expand(64, 128)

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
        text_feat = F.relu(self.text_projection_1(text_feat)) #[16,128]
        text_feat = self.drop_out(text_feat)
        text_feat = F.relu(self.text_projection_2(text_feat))

        img_feat = F.relu(self.img_projection_1(img_feat)) #[16,128]
        img_feat = self.drop_out(img_feat)
        img_feat = F.relu(self.img_projection_2(img_feat))

        fused_feat = F.relu(self.fused_projection_1(fused_feat)) #[16,128]
        fused_feat = self.drop_out(fused_feat)
        fused_feat = F.relu(self.fused_projection_2(fused_feat))

        #get_weighted_fused
        fused_weighted_feat = fused_feat * sim_weight #[16,128]

        #Transpose_n_reshape
        text_feat = text_feat.reshape((64, 1, 128))
        img_feat = img_feat.reshape((64, 1, 128))
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
        feat = fused_weighted_feat + txt_attn_feat * unimodal_weight + img_attn_feat * unimodal_weight #[16,128]


        # two-step-attention
        feat_n_senti = torch.cat((feat, sentiment), dim=1) #[64, 128 + 4]
        feat_n_senti = feat_n_senti.reshape((64, 1, 128 + 4))
        feat_n_senti = torch.transpose(feat_n_senti, 0, 1)
        feat_n_senti, _ = self.ahead_att(feat_n_senti, feat_n_senti, feat_n_senti)
        feat_n_senti = torch.transpose(feat_n_senti, 0, 1) #[16, 128 + 4]
        feat_n_senti = feat_n_senti.squeeze()
        
        feat_inte = torch.cat((feat_n_senti, z), dim=1) #[16, 128 + 4 + 100]
        feat_inte = feat_inte.reshape((64, 1, 128 + 4 + 100))
        feat_inte = torch.transpose(feat_inte, 0, 1)
        feat_attn_inte, feat_attn_inte_weights = self.self_att(feat_inte, feat_inte, feat_inte)
        feat_attn_inte = torch.transpose(feat_attn_inte, 0, 1) #[16, 128 + 4 + 100]
        feat_attn_inte = feat_attn_inte.squeeze()
        '''
        # let senti_n_topic be query
        senti_n_topic = torch.cat((senti, z), dim=1) #[64, 100 + 4]
        senti_n_topic = senti_n_topic.reshape((64, 1, 100 + 4))
        senti_n_topic = torch.transpose(senti_n_topic, 0, 1)
        senti_n_topic, _ = self.ahead_att(senti_n_topic, senti_n_topic, senti_n_topic)
        senti_n_topic = torch.transpose(senti_n_topic, 0, 1) #[64, 100 + 4]
        senti_n_topic = senti_n_topic.squeeze()
        
        senti_n_topic = F.relu(self.mlp(senti_n_topic)) #[64, 128]
        
        senti_n_topic = senti_n_topic.reshape((64, 1, 128))
        senti_n_topic = torch.transpose(senti_n_topic, 0, 1)
        feat_attn_inte, feat_attn_inte_weights = self.self_att(senti_n_topic, feat, feat)
        feat_attn_inte = torch.transpose(feat_attn_inte, 0, 1) #[16, 128 + 4 + 100]
        feat_attn_inte = feat_attn_inte.squeeze()
        '''

        '''
        #integrate_with_senti_n_topic
        feat_inte = torch.cat((feat, sentiment, z), dim=1) #[16, 128 + 4 + 100]

        #transform_n_reshape
        feat_inte = feat_inte.reshape((64, 1, 128 + 4 + 100))
        feat_inte = torch.transpose(feat_inte, 0, 1)

        #self_attentoin
        feat_attn_inte, feat_attn_inte_weights = self.self_att(feat_inte, feat_inte, feat_inte)

        #Transpose_n_squeeze
        feat_attn_inte = torch.transpose(feat_attn_inte, 0, 1) #[16, 128 + 4 + 100]
        feat_attn_inte = feat_attn_inte.squeeze()
        '''

        #MLP
        out = F.relu(self.fc1(feat_attn_inte))
        out = self.drop_out(out)
        #out = F.sigmoid(self.fc2(out))
        out = self.fc3(out)

        return out, mean, log_var, inputs_hat, feat_attn_inte

