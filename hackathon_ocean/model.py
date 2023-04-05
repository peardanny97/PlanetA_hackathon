import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        in_features = 22
        num_outputs = 14

        self._make_layer(in_features, 256, num_outputs, num_layer=2)


    def forward(self, feature):
        h = self.features(feature)
        out = self.head(h)
        out = out.squeeze()

        return out


    def _make_layer(self, in_dim, h_dim, num_classes, num_layer):
        
        if num_layer == 1:
            self.features = nn.Identity()
            h_dim = in_dim
        else:
            features = []
            for i in range(num_layer-1):
                features.append(nn.Linear(in_dim, h_dim) if i == 0 else nn.Linear(h_dim, h_dim))
                features.append(nn.ReLU(inplace=False))
                features.append(nn.Dropout(p=0.5))
            self.features = nn.Sequential(*features)

        self.head = nn.Linear(h_dim, num_classes)

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(5, 64, (2, 2), 2, padding = (1,1)),
            nn.Conv2d(64, 128, (2, 2), 2, padding = (1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, (2, 2), 2, padding = (1,1)),
            nn.Conv2d(256, 64, (2, 2), 2, padding = (1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, (2, 2), 2, padding = (1,1)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 14), nn.ReLU(inplace = True)
        )
    def forward(self, feature_layer):
        out = self.layer(feature_layer)
        out = self.fc(out)
        out = out.squeeze()
        
        return out
