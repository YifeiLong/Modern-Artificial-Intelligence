import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split

from LeNet import LeNet
from AlexNet import AlexNet
from ResNet import resnet18, resnet34, resnet50
from DenseNet import densenet121, densenet169
from VGG import vgg11


def get_model(model_name, args):
    if model_name == 'lenet':
        return LeNet(args)
    elif model_name == 'resnet':
        return resnet34()
    elif model_name == 'alexnet':
        return AlexNet(args)
    elif model_name == 'densenet':
        return densenet121(args)
    elif model_name == 'vggnet':
        return vgg11()
    else:
        return ValueError("Invalid model name")


def resize(data, size):
    resized_img = []
    for img, label in data:
        resized = torch.nn.functional.interpolate(img.unsqueeze(0), size=(size, size),
                                                  mode='bilinear', align_corners=False)
        resized = resized.squeeze(0)
        resized_img.append((resized, label))
    return resized_img


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train a CNN model on MNIST dataset')
    parser.add_argument('--model', type=str, default='lenet', help='choose a model')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    args = parser.parse_args()

    # 如果有就使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据集路径
    data_path = "./dataset"

    # 定义数据转换
    if args.model == 'alexnet' or args.model == 'vggnet':
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    full_train_data = datasets.MNIST(root=data_path, train=True, transform=transform, download=False)
    test_data = datasets.MNIST(root=data_path, train=False, transform=transform, download=False)

    # 训练集50000张图片，验证集10000张
    train_data, val_data = random_split(full_train_data, [50000, 10000])
    if args.model == 'vggnet':
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=16, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=16, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=True)

    model = get_model(args.model, args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    epochs = 10

    best_model, best_val_acc = train(model, train_loader, val_loader, criterion, optimizer, epochs, device, epochs)
    # 加载最佳模型参数
    model.load_state_dict(best_model)
    # 测试集准确率
    test_acc = test(model, test_loader, device)

    print(f'best validation accuracy: {np.round(best_val_acc, 6)}')
    print(f'test accuracy: {np.round(test_acc, 6)}')


def train(model, train_loader, val_loader, criterion, optimizer, epochs, device, num_epochs):
    best_val_acc = 0.0
    best_model = None
    train_acc_history = []
    val_acc_history = []
    loss_history = []

    for epoch in range(epochs):
        # 训练
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train
        train_acc_history.append(train_acc)

        # 在验证集上进行验证
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val
        val_acc_history.append(val_acc)

        loss_history.append(running_loss / len(train_loader))

        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader):.4f} "
              f"Train Accuracy: {train_acc:.2f}% Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()

    return best_model, best_val_acc


def test(model, test_loader, device):
    model.eval()
    total_test = 0
    correct_test = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, dim=1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

    test_acc = correct_test / total_test
    print(f"Accuracy on test set: {100 * test_acc:.2f}%")
    return test_acc


if __name__ == "__main__":
    main()
