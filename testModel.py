import torch, gc
from PIL import Image

from Bert_model.Bert_pretrain import Bert_pretrain
from CLIP_model.CLIP_img import ClipImgFeat
from CLIP_model.CLIP_similar import ClipSim
from CLIP_model.CLIP_text import ClipTextFeat
from NTM_model.NTM import VAE
from RDModel import RumorDetectionModel
from ViT_model.ViT import ViTImgFeat
from test_data_obtain import feed_dataset_test
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import MyDataset
from utils.img_process import img_2_tensor, clip_process
from utils.sim_process import img_text_process
from utils.txt_process import bert_token, clip_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
Threshold = 0.5
lambda_weight = 0.0001
path = './models/model_best.pth'

def collate_fn(data):
    txts = [i[0] for i in data]
    imgs = [i[1] for i in data]
    bows = [i[2] for i in data]
    sentis = [i[3] for i in data]
    labels = [i[4] for i in data]

    txts_list = []
    imgs_list = []
    labels_list = []
    vit_img_list = []
    clip_img_list = []
    senti_list = []

    for j in range(batch_size):
        txts_list.append(txts[j][0])

        imgs_list.append(Image.open(imgs[j][0]))

        labels_list.append(int(labels[j]))

        vit_img_tensor = img_2_tensor(imgs[j][0])
        clip_img_tensor = img_2_tensor(imgs[j][0])
        vit_img_list.append(vit_img_tensor)
        clip_img_list.append(clip_img_tensor)

        senti = []
        for k in range(len(sentis[j])):
            senti.append(float(sentis[j][k]))
        senti_list.append(senti)

    bert_input_ids, bert_attention_mask, bert_token_type_ids, bert_labels = bert_token(txts_list, labels_list)
    clip_input_ids, clip_attention_mask, clip_token_type_ids, clip_labels = clip_token(txts_list, labels_list)

    vit_imgs_tensor = torch.stack(vit_img_list).to(device)
    clip_imgs_tensor = torch.stack(clip_img_list).to(device)

    clip_sim_feat = img_text_process(txts_list, imgs_list).to(device)

    bow_tensor = torch.tensor(bows).to(device)

    senti_tensor = torch.tensor(senti_list).to(device)

    labels_tensor = torch.tensor(labels_list).to(device)

    return bert_input_ids, bert_attention_mask, bert_token_type_ids, clip_input_ids, clip_attention_mask, clip_token_type_ids, vit_imgs_tensor, clip_imgs_tensor, clip_sim_feat, bow_tensor, senti_tensor, labels_tensor

txt_test, img_test, bow_test, sentiment_test, label_test = feed_dataset_test()

dataset_test = MyDataset(txt_data=txt_test, img_data=img_test, bow_data=bow_test, senti_data=sentiment_test, label=label_test)
loader_test = DataLoader(dataset=dataset_test, batch_size=64, collate_fn=collate_fn, shuffle=True, drop_last=True)

model = RumorDetectionModel().to(device)
bert_text = Bert_pretrain().to(device)
vit_img = ViTImgFeat().to(device)
clip_text = ClipTextFeat().to(device)
clip_img = ClipImgFeat().to(device)
clip_sim = ClipSim().to(device)
ntm = VAE().to(device)

checkpoint = torch.load(path)
model.load_state_dict(checkpoint)

acc_total = 0
model.eval()
loss_test_total = 0
with torch.no_grad():
    for i, (bert_input_ids, bert_attention_mask, bert_token_type_ids, clip_input_ids, clip_attention_mask, clip_token_type_ids, vit_imgs_tensor, clip_imgs_tensor, clip_sim_feat, bow_tensor, senti_tensor, labels_tensor) in enumerate(loader_test):
        vit_imgs_tensor = vit_imgs_tensor.squeeze()
        clip_imgs_tensor = clip_imgs_tensor.squeeze()
        out_test, mu_test, log_var_test, inputs_hat_test = model(bert_text, vit_img, clip_text, clip_img, clip_sim, ntm, bert_input_ids, bert_attention_mask, bert_token_type_ids, clip_input_ids, clip_attention_mask, clip_token_type_ids, vit_imgs_tensor, clip_imgs_tensor, clip_sim_feat, bow_tensor, senti_tensor)
        reconst_loss_test = F.binary_cross_entropy(bow_tensor, inputs_hat_test, size_average=False)
        kl_div_test = - 0.5 * torch.sum(1 + log_var_test - mu_test.pow(2) - log_var_test.exp())

        # dtype_transfer
        out_test = out_test.float()
        labels_tensor = labels_tensor.long()

        #labels_tensor = labels_tensor.float()
        #labels_tensor = labels_tensor.reshape((16, 1))

        ntm_loss_test = reconst_loss_test + kl_div_test
        cls_loss_test = F.cross_entropy(out_test, labels_tensor)
        #cls_loss_test = F.binary_cross_entropy(out_test, labels_tensor, size_average=False)
        loss_test = cls_loss_test + lambda_weight * ntm_loss_test

        loss_test_total = loss_test_total + loss_test

        _, pred_test = out_test.max(1)
        acc_num_test = float(torch.eq(pred_test, labels_tensor).sum())

        #acc_calcu_test = out_test
        #acc_calcu_test[acc_calcu_test > Threshold] = 1
        #acc_calcu_test[acc_calcu_test < Threshold] = 0
        #acc_num_test = float(torch.eq(acc_calcu_test, labels_tensor).sum())
        acc_total = acc_total + acc_num_test / 64

    print(loss_test_total / len(loader_test))
    print(acc_total / len(loader_test))




