from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import jieba

train_positive = '../data/train/train_positive.txt'
train_negative = '../data/train/train_negative.txt'
test_positive = '../data/test/test_positive.txt'
test_negative = '../data/test/test_negative.txt'

train_positive_bow = '../data/train/positive_bow.txt'
train_negative_bow = '../data/train/negative_bow.txt'
test_positive_bow = '../data/test/positive_bow.txt'
test_negative_bow = '../data/test/negative_bow.txt'

train_positive_tf = '../data/train/positive_tf.txt'
train_negative_tf = '../data/train/negative_tf.txt'
test_positive_tf = '../data/test/positive_tf.txt'
test_negative_tf = '../data/test/negative_tf.txt'


def cut_func(corpus):
    with open(corpus, "rb") as f:
        data = f.read().decode("utf-8")
        data = data.split("\n")

    txt_list = []
    for i in range(len(data)):
        temp = data[i].split('[SEP]')
        if len(temp) > 1:
            seg_res = jieba.cut(temp[1], cut_all=False)
            txt_list.append(' '.join(seg_res))

    return txt_list


def bow_func(corpus, label, output):
    vectorize = CountVectorizer()
    bow = vectorize.fit_transform(corpus)
    list = bow.toarray()

    with open(output, "w", encoding="utf-8") as f:
        for i in range(len(list)):
            item = label + '[AND]'
            print(len(list[i]))
            for j in range(len(list[i])):
                item = item + str(list[i][j]) + ','
            f.write(item + '\n')


def tf_idf_func(corpus, label, output):
    vectorize = CountVectorizer()
    transformer = TfidfTransformer()
    res = vectorize.fit_transform(corpus)
    tfidf = transformer.fit_transform(res)
    list = tfidf.toarray()

    with open(output, "w", encoding="utf-8") as f:
        for i in range(len(list)):
            item = label + '[AND]'
            print(len(list[i]))
            for j in range(len(list[i])):
                item = item + str(list[i][j]) + ','
            f.write(item + '\n')


if __name__ == '__main__':
    train_pos = cut_func(train_positive)
    train_neg = cut_func(train_negative)
    test_pos = cut_func(test_positive)
    test_neg = cut_func(test_negative)

    words_dict = train_pos + train_neg + test_pos + test_neg

    print(len(words_dict))

    bow_func(words_dict, '1', train_positive_bow)
    bow_func(words_dict, '0', train_negative_bow)
    bow_func(words_dict, '1', test_positive_bow)
    bow_func(words_dict, '0', test_negative_bow)

    tf_idf_func(words_dict, '1', train_positive_tf)
    tf_idf_func(words_dict, '0', train_negative_tf)
    tf_idf_func(words_dict, '1', test_positive_tf)
    tf_idf_func(words_dict, '0', test_negative_tf)








