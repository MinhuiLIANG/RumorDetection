from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, txt_data, img_data, bow_data, senti_data, label):
        self.txt_data = txt_data
        self.img_data = img_data
        self.bow_data = bow_data
        self.senti_data = senti_data
        self.label = label

    def __getitem__(self, index):
        piece_txt_data = self.txt_data[index]
        piece_img_data = self.img_data[index]
        piece_bow_data = self.bow_data[index]
        piece_senti_data = self.senti_data[index]
        piece_label = self.label[index]

        return piece_txt_data, piece_img_data, piece_bow_data, piece_senti_data, piece_label

    def __len__(self):
        return len(self.txt_data)