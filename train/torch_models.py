import torch
import torch.nn as nn

from torchvision.models import resnet18 as resnet


class TransferModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        backbone = resnet(pretrained=pretrained)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        num_target_classes = 2
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, num_target_classes),
        )

    def forward(self, x):
        self.feature_extractor.eval()
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.Hardswish(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Hardswish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(3),
        )

    def forward(self, input):
        x = self.conv1(input)
        x = self.fc(x)  # + self.conv2(x))
        return x


class SeqModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.Hardswish(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Hardswish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.Hardswish(),
        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            proj_size=3,
        )

    def forward(self, input, hiddens=None):
        if input.shape[1] > 1:
            timestep = input.split(split_size=1, dim=1)
        else:
            timestep = [input.squeeze(dim=1)]
        features = []
        for x in timestep:
            features.append(self.conv(x.squeeze(dim=1)))
        x = torch.stack(features, dim=1)

        x, hiddens = self.lstm(x, hiddens)

        return x, hiddens
