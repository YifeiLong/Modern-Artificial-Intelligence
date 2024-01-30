import torch
import torch.optim as optim
import pandas as pd
import argparse

import aggmodel
import data


# 输出测试集结果
def output_test(model, batch_size, num_epochs, lr):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data = data.FitDataset.load("train", device)
    val_data = data.FitDataset.load('val', device)
    test_data = data.TestSet.load(device)

    train_dataloader = train_data.to_dataloader(batch_size)
    val_dataloader = val_data.to_dataloader(batch_size)
    test_dataloader = test_data.to_dataloader(batch_size)

    # 训练
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_model = None
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0.0
        cnt = 0.0

        for img, text, label, uid in train_dataloader:
            loss = -((model(img, text) + 1e-80) / (1 + 1e-80)).log()
            loss = loss[torch.arange(0, batch_size), label].mean()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            avg_loss = (avg_loss * cnt + loss.item()) / (cnt + 1)
            cnt += 1

        model.eval()
        classification = []
        for img, text, label, uid in val_dataloader:
            out = model(img, text).argmax(dim=-1).cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            classification += [{"out": out[i], "tag": label[i]} for i in range(out.shape[0])]

        df = pd.DataFrame(classification)
        acc_now = ((df['out'] == df['tag']) * 1.0).mean()
        if acc_now > best_acc:
            best_model = model
            best_acc = acc_now
        print(f'Epoch {epoch}, avg_loss: {avg_loss}, acc_now: {acc_now}, best_acc: {best_acc}')

    # 输出测试集
    classification = []
    label_map = ["positive", "neutral", "negative"]
    best_model.eval()
    for img, text, uid in test_dataloader:
        out = best_model(img, text).argmax(dim=-1).cpu().detach().numpy()
        uid = uid.cpu().detach().numpy()
        classification += [{"guid": uid[i], "tag": label_map[out[i]]} for i in range(out.shape[0])]

    res = pd.DataFrame(classification)
    res.set_index("guid")
    res.to_csv(f'answer_lr{lr}.txt', index=False)


# 消融实验
def ablation_test(model, batch_size, num_epochs, lr):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data = data.FitDataset.load("train", device)
    val_data = data.FitDataset.load('val', device)

    train_dataloader = train_data.to_dataloader(batch_size)
    val_dataloader = val_data.to_dataloader(batch_size)

    # 训练
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0.0
        cnt = 0.0

        for img, text, label, uid in train_dataloader:
            loss = -((model(img, text) + 1e-80) / (1 + 1e-80)).log()
            loss = loss[torch.arange(0, batch_size), label].mean()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            avg_loss = (avg_loss * cnt + loss.item()) / (cnt + 1)
            cnt += 1

        model.eval()
        classification = []
        for img, text, label, uid in val_dataloader:
            out = model(img, text).argmax(dim=-1).cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            classification += [{"out": out[i], "tag": label[i]} for i in range(out.shape[0])]

        df = pd.DataFrame(classification)
        acc_now = ((df['out'] == df['tag']) * 1.0).mean()
        if acc_now > best_acc:
            best_acc = acc_now
        print(f'Epoch {epoch}, avg_loss: {avg_loss}, acc_now: {acc_now}, best_acc: {best_acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a multi modal sentiment analysis model')
    parser.add_argument('--model', type=str, default='agg', help='choose a model')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    args = parser.parse_args()

    if args.model == 'agg_out':
        output_test(aggmodel.AttentionModel(), 36, 10, 1e-5)
    elif args.model == 'text':
        ablation_test(aggmodel.OnlyTextModel(), 36, 10, args.lr)
    elif args.model == 'image':
        ablation_test(aggmodel.OnlyImageModel(), 36, 10, args.lr)
    else:
        ablation_test(aggmodel.AttentionModel(), 36, 10, args.lr)
