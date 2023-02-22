con_corpus_pos = './data/train/train_positive.txt'
senti_corpus_pos = './data/train/positive_senti.txt'
bow_corpus_pos = './data/train/positive_tf.txt'

con_corpus_neg = './data/train/train_negative.txt'
senti_corpus_neg = './data/train/negative_senti.txt'
bow_corpus_neg = './data/train/negative_tf.txt'

pos_img_dir = '../weibo/rumor_images/'
neg_img_dir = '../weibo/nonrumor_images/'


def senti_dict(senti):
    if senti == 'positive':
        return 1
    if senti == 'neutral':
        return 0.5
    if senti == 'negative':
        return 0


def data_obtain(con_corpus_pos, senti_corpus_pos, bow_corpus_pos, con_corpus_neg, senti_corpus_neg, bow_corpus_neg):
    con_data = []
    label = []
    with open(con_corpus_pos, "rb") as f:
        posi = f.read().decode("utf-8")
        posi = posi.split("\n")

    for i in range(len(posi)):
        temp = posi[i].split('[SEP]')
        content = []
        txt = []
        img = []
        if len(temp) > 1:
            label.append(temp[0])
            txt.append(temp[1])
            for j in range(2, len(temp)):
                if temp[j] != '\r':
                    img.append(pos_img_dir + temp[j])
                    break

            content.append(txt)
            content.append(img)
            con_data.append(content)


    with open(con_corpus_neg, "rb") as f:
        nega = f.read().decode("utf-8")
        nega = nega.split("\n")

    for i in range(len(nega)):
        temp = nega[i].split('[SEP]')
        content = []
        txt = []
        img = []
        if len(temp) > 1:
            label.append(temp[0])
            txt.append(temp[1])
            for j in range(2, len(temp)):
                if temp[j] != '\r':
                    img.append(neg_img_dir + temp[j])
                    break

            content.append(txt)
            content.append(img)
            con_data.append(content)

    senti_data = []
    with open(senti_corpus_pos, 'rb') as f:
        pos = f.read().decode("utf-8")
        pos = pos.split("\n")

    for i in range(len(pos)):
        temp = pos[i].split('[AND]')
        if len(temp) > 1:
            tmp = temp[1].split(',')
            if len(tmp) > 1:
                sent = []
                for j in range(len(tmp) - 2):
                    sent.append(float(tmp[j]))

                sent.append(senti_dict(tmp[3]))
                senti_data.append(sent)


    with open(senti_corpus_neg, 'rb') as f:
        neg = f.read().decode("utf-8")
        neg = neg.split("\n")

    for i in range(len(neg)):
        temp = neg[i].split('[AND]')
        if len(temp) > 1:
            tmp = temp[1].split(',')
            if len(tmp) > 1:
                sent = []
                for j in range(len(tmp) - 2):
                    sent.append(tmp[j])
                sent.append(senti_dict(tmp[3]))
                senti_data.append(sent)
            else:
                print(i)


    bow_data = []
    with open(bow_corpus_pos, 'rb') as f:
        bow_pos = f.read().decode("utf-8")
        bow_pos = bow_pos.split("\n")

    for i in range(len(bow_pos)):
        temp = bow_pos[i].split('[AND]')
        if len(temp) > 1:
            tmp = temp[1].split(',')
            bow = []
            for j in range(len(tmp) - 1):
                bow.append(float(tmp[j]))

            bow_data.append(bow)


    with open(bow_corpus_neg, 'rb') as f:
        bow_neg = f.read().decode("utf-8")
        bow_neg = bow_neg.split("\n")

    for i in range(len(bow_neg)):
        temp = bow_neg[i].split('[AND]')
        if len(temp) > 1:
            tmp = temp[1].split(',')
            bow = []
            for j in range(len(tmp) - 1):
                bow.append(float(tmp[j]))

            bow_data.append(bow)

    return con_data, senti_data, bow_data, label


def feed_dataset():
    content, sentiment, bow, label = data_obtain(con_corpus_pos, senti_corpus_pos, bow_corpus_pos, con_corpus_neg, senti_corpus_neg, bow_corpus_neg)
    txts = []
    imgs = []
    for i in range(len(content)):
        txts.append(content[i][0])
        imgs.append(content[i][1])

    return txts, imgs, bow, sentiment, label


