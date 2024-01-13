import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class Encoder(nn.Module):
    def __init__(self, channels):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3))

        # Using a pre-trained model
        # self.pretrained_model = models.resnet18(pretrained=True)
        # for param in self.pretrained_model.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        # # Accessing intermediate layers for skip connections
        # x = self.pretrained_model.conv1(x)
        # x = self.pretrained_model.bn1(x)
        # x = self.pretrained_model.relu(x)
        # x = self.pretrained_model.maxpool(x)
        #
        # layer_outputs = [self.pretrained_model.layer1(x),
        #                  self.pretrained_model.layer2(x),
        #                  self.pretrained_model.layer3(x),
        #                  self.pretrained_model.layer4(x)]
        # skip_connections.extend(layer_outputs)

        return skip_connections


# Example usage
channels = [3, 64, 128, 256, 512]
encoder = Encoder(channels)
print(encoder.forward(torch.randn(1, 3, 224, 224)))
