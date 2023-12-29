import pandas as pd
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
from rouge_score import rouge_scorer
from torchtext.data import Field, BucketIterator, TabularDataset
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# 全局初始化配置参数，固定随机种子，使得每次运行的结果相同
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True


def tokenize(text):
    tokens = text.split(" ")
    return tokens


def train_val_split():
    path = "./data/train.csv"
    data = pd.read_csv(path)
    data_train, data_val = train_test_split(data, test_size=0.1, random_state=42)
    data_train.to_csv("./data/train_data.csv")
    data_val.to_csv("./data/val_data.csv")


class DataProcess:
    def __init__(self):
        super().__init__()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.DESCRIPTION = Field(sequential=True, tokenize=tokenize, include_lengths=True,
                                 init_token='<sos>', eos_token='<eos>')
        self.DIAGNOSIS = Field(sequential=True, tokenize=tokenize, init_token='<sos>',
                               eos_token='<eos>')

    def load_data(self):
        fields = [("description", self.DESCRIPTION), ("diagnosis", self.DIAGNOSIS)]
        self.train_data = TabularDataset(path="./data/train_data.csv", format="csv", fields=fields, skip_header=True)
        self.val_data = TabularDataset(path="./data/val_data.csv", format="csv", fields=fields, skip_header=True)
        self.test_data = TabularDataset(path="./data/test.csv", format="csv", fields=fields, skip_header=True)

        self.DESCRIPTION.build_vocab(self.train_data, min_freq=0)
        self.DIAGNOSIS.build_vocab(self.train_data, min_freq=0)

    def create_data_loader(self, batch_size, device):
        train_iter = BucketIterator(self.train_data, batch_size=batch_size, device=device,
                                    sort_key=lambda x: len(x.description), sort_within_batch=True)
        val_iter = BucketIterator(self.val_data, batch_size=batch_size, device=device,
                                  sort_key=lambda x: len(x.description), sort_within_batch=True)
        test_iter = BucketIterator(self.test_data, batch_size=batch_size, device=device,
                                   sort_key=lambda x: len(x.description), sort_within_batch=True)

        return train_iter, val_iter, test_iter


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.gru = nn.GRU(embed_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_len):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_len)
        packed_outputs, hidden = self.gru(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        text_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, text_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e-10)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.gru = nn.GRU((enc_hid_dim * 2) + embed_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs, mask)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        gru_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.gru(gru_input, hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, text_pad_index, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.text_pad_index = text_pad_index
        self.device = device

    def create_mask(self, text):
        mask = (text != self.text_pad_index).permute(1, 0)
        return mask

    def forward(self, text, text_len, sum, teacher_forcing_rate=0.5):
        batch_size = text.shape[1]
        sum_len = sum.shape[0]
        sum_vocab_size = self.decoder.ouput_dim

        outputs = torch.zeros(sum_len, batch_size, sum_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(text, text_len)
        input = sum[0, :]
        mask = self.create_mask(text)
        for t in range(1, sum_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_rate
            top1 = output.argmax(1)
            input = sum[t] if teacher_force else top1

        return outputs


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0.0
    for batch in iterator:
        text, text_len = batch.description
        text_len = text_len.cpu()
        summary = batch.diagnosis
        output = model(text, text_len, summary)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        summary = summary[1:].view(-1)
        loss = criterion(output, summary)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def eval(model, iterator, diagnosis_field):
    model.eval()
    predictions = []
    references = []
    smoothing = SmoothingFunction().method0

    with torch.no_grad():
        for batch in iterator:
            text, text_len = batch.description
            text_len = text_len.cpu()
            summary = batch.diagnosis
            output = model(text, text_len, summary, teacher_forcing_rate=0)
            output_index = torch.argmax(output, dim=2).tolist()
            target_index = summary.tolist()
            output_index = [
                [index for index in indices if index != diagnosis_field.vocab.stoi[diagnosis_field.eos_token]]
                for indices in output_index
            ]
            target_index = [
                [index for index in indices if index != diagnosis_field.vocab.stoi[diagnosis_field.eos_token]]
                for indices in target_index
            ]

            output_string = [' '.join([diagnosis_field.vocab.itos[index] for index in indices])
                             for indices in output_index]
            target_string = [' '.join([diagnosis_field.vocab.itos[index] for index in indices])
                             for indices in target_index]

            predictions.extend(output_string)
            references.extend(target_string)

    bleu_scores = [sentence_bleu([reference], prediction, smoothing_function=smoothing)
                   for reference, prediction in zip(references, predictions)]
    bleu_score = sum(bleu_scores) / len(bleu_scores)

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    precision = []
    recall = []
    fmeasure = []
    for target, output in zip(target_string, output_string):
        rouge_score = scorer.score(target, output)
        precision.append(rouge_score['rouge1'].precision)
        recall.append(rouge_score['rouge1'].recall)
        fmeasure.append(rouge_score['rouge1'].fmeasure)

    return bleu_score, sum(precision) / len(precision), sum(recall) / len(recall), sum(fmeasure) / len(fmeasure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a text abstraction model')
    parser.add_argument('--embed_size', type=int, default=8, help='word embedding size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden size')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_val_split()
    data_process = DataProcess()
    data_process.load_data()

    input_dim = len(data_process.DESCRIPTION.vocab)
    output_dim = len(data_process.DIAGNOSIS.vocab)

    enc_emb_dim = args.embed_size
    dec_emb_dim = args.embed_size
    enc_hid_dim = args.hid_size
    dec_hid_dim = args.hid_size

    enc_dropout = args.dropout
    dec_dropout = args.dropout
    text_pad_index = data_process.DESCRIPTION.vocab.stoi[data_process.DESCRIPTION.pad_token]

    attention = Attention(enc_hid_dim, dec_hid_dim)
    encoder = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
    decoder = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attention)
    model = Seq2Seq(encoder, decoder, text_pad_index, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=data_process.DIAGNOSIS.vocab.stoi[data_process.DIAGNOSIS.pad_token])
    num_epochs = 20
    batch_size = 1000
    best_acc = 0.0
    clip_norm = 1
    best_model = None

    train_iter, val_iter, test_iter = data_process.create_data_loader(batch_size, device)

    for epoch in range(num_epochs):
        train_loss = train(model, train_iter, optimizer, criterion, clip_norm)
        val_acc, val_precision, val_recall, val_fmeasure = eval(model, val_iter, data_process.DIAGNOSIS)
        print(f'Epoch: {epoch + 1}, train Loss: {train_loss}, bleu: {val_acc}, precision: {val_precision}, '
              f'recall: {val_recall}, fmeasure: {val_fmeasure}')
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model

    torch.save(model.state_dict(), 'best_gru_model.pth')

    test_bleu, test_precision, test_recall, test_fmeasure = eval(best_model, test_iter, criterion,
                                                                 data_process.DIAGNOSIS)
    print(f'test bleu: {test_bleu}, test precision: {test_precision}, test recall: {test_recall}, '
          f'test fmeasure: {test_fmeasure}')
