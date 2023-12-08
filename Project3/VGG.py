import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, conv_arch):
        super(VGG, self).__init__()
        self.conv_layers = self._make_conv_layers(conv_arch)
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def _make_conv_layers(self, conv_arch):
        layers = []
        in_channels = 1
        for (num_convs, out_channels) in conv_arch:
            layers.extend(self._make_vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return layers

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def vgg11():
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    return VGG(conv_arch)
