import pandas as pd
import numpy as np
import argparse
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


class DataProcess:
    def __init__(self):
        super().__init__()
        self.train_source = []
        self.train_target = []
        self.val_source = []
        self.val_target = []
        self.test_source = []
        self.test_target = []

    def data_split(self):
        data = pd.read_csv("./data/train.csv", encoding='utf-8')
        src_text = []
        tgt_text = []

        for i in range(18000):
            src_words = data['description'][i].split(" ")
            tgt_words = data['diagnosis'][i].split(" ")
            src_words = [int(x) for x in src_words]
            tgt_words = [int(x) for x in tgt_words]
            src_text.append(src_words)
            tgt_text.append(tgt_words)

        # 划分训练集、验证集
        val_index = np.random.choice(18000, size=1800, replace=False)
        train_index = np.setdiff1d(np.arange(18000), val_index)

        self.train_source = [src_text[i] for i in train_index]
        self.train_target = [tgt_text[i] for i in train_index]
        self.val_source = [src_text[i] for i in val_index]
        self.val_target = [tgt_text[i] for i in val_index]

        return self.train_source, self.train_target, self.val_source, self.val_target

    def test_data(self):
        data = pd.read_csv("./data/test.csv", encoding='utf-8')
        src_text = []
        tgt_text = []

        for i in range(2000):
            src_words = data['description'][i].split(" ")
            tgt_words = data['diagnosis'][i].split(" ")
            src_words = [int(x) for x in src_words]
            src_text.append(src_words)
            tgt_text.append(tgt_words)

        self.test_source = src_text
        self.test_target = tgt_text
        return self.test_source, self.test_target


def train_data_process(train_src, train_tgt, batch_size, tokenizer):
    for i in range(0, 16200, batch_size):
        src_list = []
        tgt_list = []
        for j in range(batch_size):
            if i + j < len(train_src):
                src = tokenizer.encode_plus(train_src[i + j], padding='max_length', truncation=True, max_length=100,
                                            return_tensors='pt')['input_ids'][0]
                tgt = tokenizer.encode_plus(train_tgt[i + j], padding='max_length', truncation=True, max_length=100,
                                            return_tensors='pt')['input_ids'][0]
                src_list.append(src)
                tgt_list.append(tgt)

        train_input.append(torch.stack(src_list, dim=0))
        train_label.append(torch.stack(tgt_list, dim=0))

    return train_input, train_label


def train(model, train_input, train_label, val_src, val_tgt, num_epochs, optimizer, scorer, device):
    best_model = None
    best_acc = 0.0
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for source, labels in zip(train_input, train_label):
            optimizer.zero_grad()
            input_ids = torch.tensor(source, dtype=torch.long).to(device)
            input_labels = torch.tensor(labels, dtype=torch.long).to(device)
            outputs = model(input_ids=input_ids, labels=input_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, total loss: {total_loss}')

        # 验证集
        acc_now = eval(model, val_src, val_tgt, epoch, num_epochs, scorer, device)
        if acc_now > best_acc:
            best_acc = acc_now
            best_model = model

    torch.save(best_model.state_dict(), 'best_bart.pth')
    return best_model


def eval(model, val_src, val_tgt, epoch, num_epochs, scorer, device):
    model.eval()
    total_bleu = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_fmeasure = 0.0
    for source, target in zip(val_src, val_tgt):
        source = [0] + source + [2]
        encode_input = torch.tensor([source], dtype=torch.long).to(device)
        output = model.generate(encode_input)
        res = output[0, 2:-1].tolist()

        res = [str(i) for i in res]
        res = ' '.join(res)
        target = [str(i) for i in target]
        target = ' '.join(target)

        total_bleu += sentence_bleu([target], res, weights=(0.5, 0.5, 0, 0),
                                    smoothing_function=SmoothingFunction().method0)

        rouge_score = scorer.score(target, res)
        total_precision += rouge_score['rouge1'].precision
        total_recall += rouge_score['rouge1'].recall
        total_fmeasure += rouge_score['rouge1'].fmeasure

    acc_now = total_precision / 1800
    print(f'Epoch: {epoch}/{num_epochs}, val bleu: {total_bleu / 1800}, precision: {acc_now}, '
          f'recall: {total_recall / 1800}, fmeasure: {total_fmeasure / 1800}')
    return acc_now


def predict(best_model, test_src, test_tgt, scorer, device):
    best_model.to(device)
    best_model.eval()

    test_bleu = 0.0
    test_precision = 0.0
    test_recall = 0.0
    test_fmeasure = 0.0
    test_precision2 = 0.0
    test_recall2 = 0.0
    test_fmeasure2 = 0.0

    for source, target in zip(test_src, test_tgt):
        source = [0] + source + [2]
        encode_input = torch.tensor([source], dtype=torch.long).to(device)
        output = best_model.generate(encode_input)
        res = output[0, 2:-1].tolist()
        res = [str(i) for i in res]
        target = [str(i) for i in target]
        test_bleu += sentence_bleu([' '.join(target)], ' '.join(res),
                                   weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method0)

        rouge_score = scorer.score(' '.join(target), ' '.join(res))
        test_precision += rouge_score['rouge1'].precision
        test_recall += rouge_score['rouge1'].recall
        test_fmeasure += rouge_score['rouge1'].fmeasure

        test_precision2 += rouge_score['rouge2'].precision
        test_recall2 += rouge_score['rouge2'].recall
        test_fmeasure2 += rouge_score['rouge2'].fmeasure

    print(f'test bleu: {test_bleu / 2000}, test precision1: {test_precision / 2000}, '
          f'test recall1: {test_recall / 2000}, test fmeasure1: {test_fmeasure / 2000}, '
          f'test precision2: {test_precision2 / 2000}, test recall2: {test_recall2 / 2000}, '
          f'test fmeasure2: {test_fmeasure2 / 2000}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a text abstraction model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epoch', type=int, default=20, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=65, help='traning batch size')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_process = DataProcess()
    train_src, train_tgt, val_src, val_tgt = data_process.data_split()

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    model.to(device)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    num_epochs = args.epoch
    batch_size = args.batch_size
    train_input, train_label = train_data_process(train_src, train_tgt, batch_size, tokenizer)

    best_model = train(model, train_input, train_label, val_src, val_tgt, num_epochs, optimizer, scorer, device)

    test_src, test_tgt = data_process.test_data()
    scorer1 = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    predict(best_model, test_src, test_tgt, scorer1, device)
