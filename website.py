from gevent import monkey
monkey.patch_all()

import torch
import random
import urllib.request
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from gevent.pywsgi import WSGIServer
from multiprocessing import cpu_count, Process
from torch.utils.data import DataLoader

from dataset import MyDataset
from utils.img_process import img_2_tensor
from utils.sim_process import img_text_process
from utils.txt_process import bert_token, clip_token
from PIL import Image

from Bert_model.Bert_pretrain import Bert_pretrain
from CLIP_model.CLIP_img import ClipImgFeat
from CLIP_model.CLIP_similar import ClipSim
from CLIP_model.CLIP_text import ClipTextFeat
from NTM_model.NTM import VAE
from RDModel import RumorDetectionModel
from ViT_model.ViT import ViTImgFeat

root = './images/img_'
path = './attention_models/attention_model.pth'
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
cors = CORS(app)

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

#https://img1.imgtp.com/2023/03/26/z7OpOqok.jpg
@app.route('/rumorDetection', methods=['POST', 'GET'])
def rumorDetection():
    bow = [0] * 40535
    senti = [0] * 4

    user_request = request.get_json()

    get_txt = user_request.get("text")
    get_img = user_request.get("image")

    index = random.randint(1,10000)

    r = urllib.request.urlretrieve(get_img, root + str(index) + '.jpg')

    text = list(get_txt)
    image = list(root + str(index) + '.jpg')
    label = [1]

    dataset = MyDataset(txt_data=text, img_data=image, bow_data=bow, senti_data=senti, label=label)
    loader = DataLoader(dataset=dataset, batch_size=1, collate_fn=collate_fn, shuffle=True, drop_last=True)

    model = RumorDetectionModel().to(device)
    bert_text = Bert_pretrain().to(device)
    vit_img = ViTImgFeat().to(device)
    clip_text = ClipTextFeat().to(device)
    clip_img = ClipImgFeat().to(device)
    clip_sim = ClipSim().to(device)
    ntm = VAE().to(device)

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint, strict=False)

    model.eval()
    with torch.no_grad():
        for i, (bert_input_ids, bert_attention_mask, bert_token_type_ids, clip_input_ids, clip_attention_mask,
                clip_token_type_ids, vit_imgs_tensor, clip_imgs_tensor, clip_sim_feat, bow_tensor, senti_tensor,
                labels_tensor) in enumerate(loader):
            vit_imgs_tensor = vit_imgs_tensor.squeeze()
            clip_imgs_tensor = clip_imgs_tensor.squeeze()
            out, mu, log_var, inputs_hat, feat = model(bert_text, vit_img, clip_text, clip_img,
                                                                                clip_sim, ntm, bert_input_ids,
                                                                                bert_attention_mask,
                                                                                bert_token_type_ids, clip_input_ids,
                                                                                clip_attention_mask,
                                                                                clip_token_type_ids, vit_imgs_tensor,
                                                                                clip_imgs_tensor, clip_sim_feat,
                                                                                bow_tensor, senti_tensor)

            _, pred = out.max(0)
            return jsonify(result=pred)

def run(MULTI_PROCESS):
    if MULTI_PROCESS == False:
        WSGIServer(('0.0.0.0', 8086), app).serve_forever()
    else:
        mulserver = WSGIServer(('0.0.0.0', 8086), app)
        mulserver.start()

        def server_forever():
            mulserver.start_accepting()
            mulserver._stop_event.wait()

        for i in range(cpu_count()):
            p = Process(target=server_forever)
            p.start()


if __name__ == "__main__":
    # 单进程 + 协程
    run(False)
    # 多进程 + 协程
    # run(True)








