import torch
import torch.nn as nn
import torchvision.transforms as transforms


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_block = EncoderBlock(in_channels, out_channels)

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
    def __init__(self, channels):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for block in range(len(channels) - 1):
            self.layers.append(EncoderBlock(channels[block], channels[block + 1]))

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
        for i in range(len(depths)):
            in_channels = depths[i] * 2
            out_channels = depths[i]
            self.layers.append(DecoderBlock(in_channels, out_channels))

    def forward(self, x, enc_activations):
        for i, layer in enumerate(self.layers):
            x = layer(x, enc_activations[len(enc_activations) - i - 1])
        return x


class UNet(nn.Module):
    def __init__(self, encoder_channels, decoder_depths, num_classes):
        super(UNet, self).__init__()
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_depths)

        # 1x1 convolution to get the segmentation map
        self.final_conv = nn.Conv2d(encoder_channels[1], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc_activations = self.encoder(x)

        # Decoder with skip connections
        dec_output = self.decoder(enc_activations[-1], enc_activations[:-1])

        # Segmentation map
        seg_map = self.final_conv(dec_output)

        # Resize to match the input dimensions
        seg_map = nn.functional.interpolate(seg_map, size=x.size()[2:], mode='bilinear', align_corners=False)

        return seg_map
