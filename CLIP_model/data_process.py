from PIL import Image
from transformers import CLIPProcessor
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig
import numpy as np

train_positive = '../data/train/train_positive.txt'
train_negative = '../data/train/train_negative.txt'
root_positive = '../weibo/rumor_images/'
root_negative = '../weibo/nonrumor_images/'


config_text = CLIPTextConfig(max_position_embeddings=1024)
config_vision = CLIPVisionConfig()

config = CLIPConfig.from_text_vision_configs(config_text, config_vision)


def data_reader(corpus, label):
    with open(corpus, "rb") as f:
        data = f.read().decode("utf-8")
        data = data.split("\n")

    txt_list = []
    img_list = []
    for i in range(len(data)):
        imgs = []
        temp = data[i].split('[SEP]')
        if len(temp) > 1:
            txt_list.append(temp[1])
            for j in range(2, len(temp)):
                if temp[j] != '\r':

                    #img = img_process(temp[j], label)
                    dir = ''
                    if label == '0':
                        dir = root_negative + temp[j]
                    if label == '1':
                        dir = root_positive + temp[j]

                    img = Image.open(dir, 'r')
                    imgs.append(img)
                    img.close()

            img_list.append(imgs)


if __name__ == '__main__':
    data_reader(train_positive, '1')



