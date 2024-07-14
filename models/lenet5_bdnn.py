from torch import nn
from .bdnn import BDNN
from .configs import ForwardConfig, FeatureModel


class LeNet5_BDNN(FeatureModel):
    def __init__(self, out_features: int, forward_only: bool):
        super().__init__(forward_only, out_features)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten()
        )
        self.bdnn = BDNN(16*5*5,
                         self.out_features,
                         self.forward_only,
                         [120, 84])

    def forward(self, x, config: ForwardConfig=ForwardConfig.FORWARD):
        if config == ForwardConfig.BACKWARD:
            x = self.bdnn(x, config=config)
            return x

        # Feed forward.
        x = self.features(x)
        if config == ForwardConfig.FEATURES_ONLY:
            return x

        x = self.bdnn(x)
        return x

    def freeze_features_seq(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_features_seq(self):
        for param in self.features.parameters():
            param.requires_grad = True
