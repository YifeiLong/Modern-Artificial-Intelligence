import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, args):
        super(AlexNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, padding=1, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Flatten(),
            nn.Linear(6400, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            # nn.Dropout(args.dropout),

            nn.Linear(4096, 10)
        )

    def forward(self, img):
        output = self.net(img)
        return output
