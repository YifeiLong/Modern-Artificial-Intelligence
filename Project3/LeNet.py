import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),

            nn.Dropout(args.dropout),
            nn.Linear(120, 84),
            nn.ReLU(),

            nn.Linear(84, 10)
        )

    def forward(self, img):
        output = self.net(img)
        return output
