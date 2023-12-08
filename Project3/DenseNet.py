import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)


# Transition Layer
class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


# Dense Block中一组卷积组合
class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck_size, drop_rate):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, growth_rate * bottleneck_size, kernel_size=1, bias=False)

        self.norm2 = nn.BatchNorm2d(growth_rate * bottleneck_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(growth_rate * bottleneck_size, growth_rate, kernel_size=3, padding=1, bias=False)

        self.drop_rate = float(drop_rate)

    def forward(self, x):
        x = [x] if torch.is_tensor(x) else x
        x = self.conv1(self.relu1(self.norm1(torch.cat(x, 1))))
        output = self.conv2(self.relu2(self.norm2(x)))

        if self.drop_rate > 0:
            output = F.dropout(output, p=self.drop_rate, training=self.training)
        return output


# 建立一个DenseBlock，根据网络结构输入
class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, in_channels, growth_rate, bottleneck_size, drop_rate):
        super().__init__()

        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, bottleneck_size, drop_rate)
            self.add_module(f'denselayer{i + 1}', layer)

    def forward(self, x):
        xs = [x]

        for name, layer in self.items():
            x_new = layer(xs)
            xs.append(x_new)
        return torch.cat(xs, 1)


# 完整DenseNet网络结构
class DenseNet(nn.Module):
    def __init__(self, args, block_config, in_channels=32, growth_rate=16, bottleneck_size=4, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, in_channels, kernel_size=7, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(in_channels)),
            ('relu0', nn.ReLU()),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        num_features = in_channels

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, growth_rate, bottleneck_size, args.dropout)
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features += (num_layers * growth_rate)

            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        self.features.add_module(f'norm{i + 2}', nn.BatchNorm2d(num_features))
        self.features.add_module(f'relu{i + 2}', nn.ReLU(inplace=True))
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(self.features(x), (1, 1))
        out = self.classifier(flatten(out))
        return out


# DenseNet-121
def densenet121(args):
    return DenseNet(args, block_config=(6, 12, 24, 16))


# DenseNet-169
def densenet169(args):
    return DenseNet(args, block_config=(6, 12, 32, 32))
