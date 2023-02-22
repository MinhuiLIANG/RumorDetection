import os


corpus_nonrumor = '../weibo/tweets/train_nonrumor.txt'
img_nonrumor = '../weibo/nonrumor_images'
corpus_rumor = '../weibo/tweets/train_rumor.txt'
img_rumor = '../weibo/rumor_images'
test_nonrumor = '../weibo/tweets/test_nonrumor.txt'
test_rumor = '../weibo/tweets/test_rumor.txt'

stop_words = '../weibo/stop_words.txt'

train_positive = '../data/train/train_positive.txt'
train_negative = '../data/train/train_negative.txt'
test_positive = '../data/test/test_positive.txt'
test_negative = '../data/test/test_negative.txt'

positive = '1'
negative = '0'


def word_filter(corpus, txt):
    with open(corpus, "rb") as f:
        data = f.read().decode("utf-8")
        data = data.split("\n")

    for i in range(len(data)):
        txt = txt.replace(data[i], '')

    return txt


def data_process(corpus, image):
    with open(corpus, "rb") as f:
        data = f.read().decode("utf-8")
        data = data.split("\r\n")

    raw_data = []
    temp = []
    for i in range(len(data)):
        if i % 3 == 0 and i != 0:
            raw_data.append(temp)
            temp = []
        temp.append(data[i])

    url = []
    txt = []
    for i in range(len(raw_data)):
        url.append(raw_data[i][1])
        words = word_filter(stop_words, raw_data[i][2])
        txt.append(words)

    img = os.listdir(image)

    good_pair = []
    for i in range(len(url)):
        has_txt = 0
        temp = []
        for j in range(len(img)):
            if img[j] in url[i]:
                if has_txt == 0:
                    temp.append(txt[i])
                temp.append(img[j])
                has_txt = 1
        if has_txt != 0 and txt[i] != '':
            good_pair.append(temp)

    return good_pair


def data_output(corpus, data, label):
    with open(corpus, "w", encoding="utf-8") as f:
        for i in range(len(data)):
            item = label + '[SEP]'
            for j in range(len(data[i])):
                data[i][j] = data[i][j].replace(' ', '')
                item = item + data[i][j] + '[SEP]'
            f.write(item + '\n')


if __name__ == '__main__':
    nonrumor_train_set = data_process(corpus_nonrumor, img_nonrumor)
    rumor_train_set = data_process(corpus_rumor, img_rumor)
    nonrumor_test_set = data_process(test_nonrumor, img_nonrumor)
    rumor_test_set = data_process(test_rumor, img_rumor)

    data_output(train_negative, nonrumor_train_set, negative)
    data_output(train_positive, rumor_train_set, positive)
    data_output(test_negative, nonrumor_test_set, negative)
    data_output(test_positive, rumor_test_set, positive)
