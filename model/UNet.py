import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, nrChannelsPerBlock=[3, 64, 128, 256, 512, 1024]):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for block in range(len(nrChannelsPerBlock) - 1):
            self.layers.append(Block(nrChannelsPerBlock[block], nrChannelsPerBlock[block + 1]))
            # self.layers.append(nn.MaxPool2d(2))

    def forward(self, x):
        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        return skip_connections


class UNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

