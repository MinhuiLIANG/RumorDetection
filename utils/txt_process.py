import torch
from transformers import BertTokenizer
from transformers import AutoTokenizer

token_bert = BertTokenizer.from_pretrained('bert-base-chinese')
token_clip = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bert_token(txts, labels):
    data = token_bert.batch_encode_plus(batch_text_or_text_pairs=txts,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=197,
                                   return_tensors='pt',
                                   return_length=True)

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)

    return input_ids, attention_mask, token_type_ids, labels


def clip_token(txts, labels):
    data = token_clip.batch_encode_plus(batch_text_or_text_pairs=txts,
                                        truncation=True,
                                        padding='max_length',
                                        max_length=197,
                                        return_tensors='pt',
                                        return_length=True)

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)

    return input_ids, attention_mask, token_type_ids, labels
