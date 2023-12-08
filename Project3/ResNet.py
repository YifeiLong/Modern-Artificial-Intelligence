import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        identity = X
        y = self.relu(self.bn1(self.conv1(X)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            identity = self.downsample(X)

        return self.relu(y + identity)


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        identity = X

        y = self.relu(self.bn1(self.conv1(X)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))

        if self.downsample is not None:
            identity = self.downsample(X)

        return self.relu(y + identity)


# 根据residual和num_residuals设定ResNet网络深度与结构: ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
class ResNet(nn.Module):
    def __init__(self, residual, num_residuals, num_classes=10, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.out_channel = 64

        # 最开始共用的卷积、池化层
        self.conv1 = nn.Conv2d(1, self.out_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 根据输入决定网络残差块结构与数量
        self.conv2 = self.residual_block(residual, 64, num_residuals[0])
        self.conv3 = self.residual_block(residual, 128, num_residuals[1], stride=2)
        self.conv4 = self.residual_block(residual, 256, num_residuals[2], stride=2)
        self.conv5 = self.residual_block(residual, 512, num_residuals[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * residual.expansion, num_classes)

        # 对conv层进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def residual_block(self, residual, channel, num_residuals, stride=1):
        downsample = None
        if stride != 1 or self.out_channel != channel * residual.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.out_channel, channel * residual.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * residual.expansion)
            )

        block = []
        block.append(residual(self.out_channel, channel, downsample=downsample, stride=stride))
        self.out_channel = channel * residual.expansion

        for _ in range(1, num_residuals):
            block.append(residual(self.out_channel, channel))

        return nn.Sequential(*block)

    def forward(self, X):
        y = self.relu(self.bn1(self.conv1(X)))
        y = self.maxpool(y)
        y = self.conv5(self.conv4(self.conv3(self.conv2(y))))

        if self.include_top:
            y = self.avgpool(y)
            y = torch.flatten(y, 1)
            y = self.fc(y)

        return y


def resnet18(num_classes=10, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=10, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=10, include_top=True):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
