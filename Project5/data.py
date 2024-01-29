import pandas as pd
import transformers
import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader


# 读入训练集和验证集
class FitDataset(Dataset):
    def __init__(self, meta, mode, device):
        super().__init__()
        self.meta = meta
        self.mode = mode
        self.meta['tag'] = self.meta['tag'].map({'positive': 0, 'neutral': 1, 'negative': 2})
        self.img_trans = tv.transforms.Compose([tv.transforms.Resize((224, 224))])
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.device = device

    def __getitem__(self, index):
        uid = self.meta['guid'][index]
        img = self.img_trans(tv.io.read_image(f'dataset/{self.mode}/img/{uid}.jpg'))
        with open(f'dataset/{self.mode}/text/{uid}.txt', encoding='utf-8', errors='surrogateescape') as f:
            txt = f.read().strip()
        tag = self.meta['tag'][index]
        return img, txt, tag, uid

    def __len__(self):
        return self.meta.__len__()

    def to_dataloader(self, batch_size, shuffle=False):
        def collate_fn(input):
            img = torch.stack([i[0] for i in input]).to(self.device)
            text = self.tokenizer([i[1] for i in input], padding=True, return_tensors='pt').to(self.device)
            tag = torch.tensor([i[2] for i in input]).to(self.device)
            uid = torch.tensor([i[3] for i in input]).to(self.device)
            return (img / 255), text, tag, uid
        return DataLoader(self, batch_size, shuffle, collate_fn=collate_fn)

    @staticmethod
    def load(mode, device):
        if mode == 'train':
            meta = pd.read_csv('dataset/train/train.csv')
            return FitDataset(meta, mode, device)
        else:
            meta = pd.read_csv('dataset/val/val.csv')
            return FitDataset(meta, mode, device)


# 读入测试集
class TestSet(Dataset):
    def __init__(self, meta, device):
        super().__init__()
        self.meta = meta
        self.img_trans = tv.transforms.Compose([tv.transforms.Resize((224, 224))])
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.device = device

    def __getitem__(self, index):
        uid = self.meta['guid'][index]
        img = self.img_trans(tv.io.read_image(f'dataset/test/img/{uid}.jpg'))
        with open(f'dataset/test/text/{uid}.txt', encoding='utf-8', errors='surrogateescape') as f:
            txt = f.read().strip()
        return img, txt, uid

    def __len__(self):
        return self.meta.__len__()

    def to_dataloader(self, batch_size, shuffle=False):
        def collate_fn(input):
            img = torch.stack([i[0] for i in input]).to(self.device)
            text = self.tokenizer([i[1] for i in input], padding=True, return_tensors='pt').to(self.device)
            uid = torch.tensor([i[2] for i in input]).to(self.device)
            return (img / 255), text, uid
        return DataLoader(self, batch_size, shuffle, collate_fn=collate_fn)

    @staticmethod
    def load(device):
        meta = pd.read_csv('dataset/test_without_label.txt')
        return TestSet(meta, device)
