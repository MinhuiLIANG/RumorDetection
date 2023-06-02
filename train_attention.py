import torch, gc
from PIL import Image

from Bert_model.Bert_pretrain import Bert_pretrain
from CLIP_model.CLIP_img import ClipImgFeat
from CLIP_model.CLIP_similar import ClipSim
from CLIP_model.CLIP_text import ClipTextFeat
from NTM_model.NTM import VAE
from RDModel import RumorDetectionModel
from ViT_model.ViT import ViTImgFeat
from attention import AttentionModel
from baseline import BaselineModel
from data_obtain import feed_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import MyDataset
from utils.img_process import img_2_tensor
from utils.sim_process import img_text_process
from utils.txt_process import bert_token, clip_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_num = 500
batch_size = 64
epoch_num = 35
Threshold = 0.5
learning_rate = 0.001
lambda_weight = 0.01

root = './attention_models/model_'

def split_train():
    txts, imgs, bow, sentiment, label = feed_dataset()
    txts_train = txts[val_num:]
    txts_val = txts[:val_num]
    imgs_train = imgs[val_num:]
    imgs_val = imgs[:val_num]
    bow_train = bow[val_num:]
    bow_val = bow[:val_num]
    sentiment_train = sentiment[val_num:]
    sentiment_val = sentiment[:val_num]
    label_train = label[val_num:]
    label_val = label[:val_num]

    return txts_train, txts_val, imgs_train, imgs_val, bow_train, bow_val, sentiment_train, sentiment_val, label_train, label_val


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


txts_train, txts_val, imgs_train, imgs_val, bow_train, bow_val, sentiment_train, sentiment_val, label_train, label_val = split_train()

dataset = MyDataset(txt_data=txts_train, img_data=imgs_train, bow_data=bow_train, senti_data=sentiment_train, label=label_train)
loader = DataLoader(dataset=dataset, batch_size=64, collate_fn=collate_fn, shuffle=True, drop_last=True)

dataset_val = MyDataset(txt_data=txts_val, img_data=imgs_val, bow_data=bow_train, senti_data=sentiment_val, label=label_val)
loader_val = DataLoader(dataset=dataset_val, batch_size=64, collate_fn=collate_fn, shuffle=True, drop_last=True)

model = AttentionModel().to(device)
bert_text = Bert_pretrain().to(device)
vit_img = ViTImgFeat().to(device)
clip_text = ClipTextFeat().to(device)
clip_img = ClipImgFeat().to(device)
clip_sim = ClipSim().to(device)
ntm = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []
acc = []
losses_val = []
acc_val = []
best_performance = 0

for epoch in range(epoch_num):

    train_loss_per_epoch = 0
    train_acc_per_epoch = 0

    if epoch % 5 == 0 and epoch != 0:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5

    #train
    model.train()
    for i, (bert_input_ids, bert_attention_mask, bert_token_type_ids, clip_input_ids, clip_attention_mask, clip_token_type_ids, vit_imgs_tensor, clip_imgs_tensor, clip_sim_feat, bow_tensor, senti_tensor, labels_tensor) in enumerate(loader):
        vit_imgs_tensor = vit_imgs_tensor.squeeze()
        clip_imgs_tensor = clip_imgs_tensor.squeeze()
        out, _ = model(bert_text, vit_img, clip_text, clip_img, clip_sim, ntm, bert_input_ids, bert_attention_mask, bert_token_type_ids, clip_input_ids, clip_attention_mask, clip_token_type_ids, vit_imgs_tensor, clip_imgs_tensor, clip_sim_feat, bow_tensor, senti_tensor)

        #dtype_transfer
        out = out.float()
        labels_tensor = labels_tensor.long()

        #labels_tensor = labels_tensor.float()
        #labels_tensor = labels_tensor.reshape((16, 1))

        # Backprop and optimize
        cls_loss = F.cross_entropy(out, labels_tensor)
        #cls_loss = F.binary_cross_entropy(out, labels_tensor, size_average=False)
        loss = cls_loss

        train_loss_per_epoch = train_loss_per_epoch + loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = out.max(1)
        acc_num = float(torch.eq(pred, labels_tensor).sum())

        #acc_calcu = out.squeeze()
        #lbs = labels_tensor.squeeze()
        #acc_calcu[acc_calcu > Threshold] = 1
        #acc_calcu[acc_calcu < Threshold] = 0
        #acc_num = float(torch.eq(acc_calcu, lbs).sum())
        train_acc_per_epoch = train_acc_per_epoch + acc_num / 64

    losses.append(train_loss_per_epoch/len(loader))
    acc.append(train_acc_per_epoch/len(loader))

    model_path = root + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_path)
    print('epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(epoch, train_loss_per_epoch/len(loader), train_acc_per_epoch/len(loader)))

print('---training_process_end---')