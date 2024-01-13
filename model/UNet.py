import torch
import torch.nn as nn
import torchvision.transforms as transforms


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
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


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, middle_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_block = EncoderBlock(middle_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.bn(x)
        x = self.relu(x)

        # Crop the skip connection and concatenate
        crop_size = x.size()[2:]
        crop = transforms.CenterCrop(crop_size)(skip)
        x = torch.cat([x, crop], dim=1)

        x = self.conv_block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, nrChannelsPerBlock=[3, 64, 128, 256, 512, 1024]):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for block in range(len(nrChannelsPerBlock) - 1):
            self.layers.append(EncoderBlock(nrChannelsPerBlock[block], nrChannelsPerBlock[block + 1]))
            # self.layers.append(nn.MaxPool2d(2))

    def forward(self, x):
        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        return skip_connections


class Decoder(nn.Module):
    def __init__(self, depths):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(depths) - 2, -1, -1):
            self.layers.append(DecoderBlock(depths[i + 1], depths[i], depths[i]))

    def forward(self, x, enc_activations):
        for i, layer in enumerate(self.layers):
            x = layer(x, enc_activations[i])
        return x


class UNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
