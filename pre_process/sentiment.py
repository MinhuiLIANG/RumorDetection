import json
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.nlp.v20190408 import nlp_client, models

import time

SecretId = 'AKIDqX2KVUGo8D6kEZlpjZijjX7XjkYAaooa'
SecretKey = 'hCAtR0ZKvISO6ZptlV4RbgdFCyZBrOis'

train_positive = '../data/train/train_positive.txt'
train_negative = '../data/train/train_negative.txt'
test_positive = '../data/test/test_positive.txt'
test_negative = '../data/test/test_negative.txt'

train_positive_senti = '../data/train/positive_senti.txt'
train_negative_senti = '../data/train/negative_senti.txt'
test_positive_senti = '../data/test/positive_senti.txt'
test_negative_senti = '../data/test/negative_senti.txt'


def senti_analysis(corpus):
    senti_list = []
    try:
        cred = credential.Credential(SecretId, SecretKey)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "nlp.tencentcloudapi.com"

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = nlp_client.NlpClient(cred, "ap-guangzhou", clientProfile)

        req = models.SentimentAnalysisRequest()
        params = {
          "Text": corpus,
          "Flag": 2,
          "Mode": "3class"
        }
        req.from_json_string(json.dumps(params))

        resp = client.SentimentAnalysis(req)

        senti_list.append(resp.Positive)
        senti_list.append(resp.Neutral)
        senti_list.append(resp.Negative)
        senti_list.append(resp.Sentiment)
        print(senti_list)

    except TencentCloudSDKException as err:
        print(err)

    return senti_list


def make_senti(corpus):
    with open(corpus, "rb") as f:
        data = f.read().decode("utf-8")
        data = data.split("\n")

    sen_file = []
    for i in range(len(data)):
        temp = data[i].split('[SEP]')
        if len(temp) > 1:
            sent = temp[1]
            sen_list = senti_analysis(sent)
            sen_file.append(sen_list)

    return sen_file


def make_file(sen_file, output, label):
    with open(output, "w", encoding="utf-8") as f:
        for i in range(len(sen_file)):
            item = label + '[AND]'
            for j in range(len(sen_file[i])):
                item = item + str(sen_file[i][j]) + ','
            f.write(item + '\n')


if __name__ == '__main__':
    train_pos = make_senti(train_positive)
    print('flag')
    train_neg = make_senti(train_negative)
    print('flag')
    test_pos = make_senti(test_positive)
    print('flag')
    test_neg = make_senti(test_negative)
    print('flag')

    make_file(train_pos, train_positive_senti, '1')
    make_file(train_neg, train_negative_senti, '0')
    make_file(test_pos, test_positive_senti, '1')
    make_file(test_neg, test_negative_senti, '0')


