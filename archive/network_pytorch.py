import torch
import torch.nn as nn
import torch.nn.functional as F

class EncodeBlock(nn.Module):
    def __init__(self, n_ch, n_kernel=3, n_padding=1, drop_rate=0.1):
        super(EncodeBlock, self).__init__()
        # self.n_ch = n_ch
        # self.n_kernel = n_kernel
        # self.n_padding = n_padding
        # self.drop_rate = drop_rate
        n_in_ch = int(n_ch / 2)

        self.encode_block = nn.Sequential(
            nn.BatchNorm2d(n_in_ch),
            nn.Tanh(),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(n_in_ch, n_ch, n_kernel, padding=n_padding),
            nn.BatchNorm2d(n_ch),
            nn.Tanh(),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(n_ch, n_ch, n_kernel, padding=n_padding)
        )

        # self.bn1 = nn.BatchNorm2d(n_ch)
        # self.drop1 = nn.Dropout2d(drop_rate)
        # self.conv1 = nn.Conv2d(n_ch, n_ch, kernel)
        # self.bn2 = nn.BatchNorm2d(ch)
        # self.drop2 = nn.Dropout2d(drop_rate)
        # self.conv2 = nn.Conv2d(ch, ch, kernel, padding=padding)

    def forward(self, x):
        # x = self.bn1(x)
        # x = F.tanh(x)
        # x = self.drop1(x)
        # x = self.conv1(x)
        # x = self.bn2(x)
        # x = F.tanh(x)
        # x = self.drop2(x)
        # x = self.conv2(x)
        # return x
        return self.encode_block(x)

class DecodeBlock(nn.Module):
    def __init__(self, n_ch, n_kernel=3, n_padding=1, drop_rate=0.1):
        super(DecodeBlock, self).__init__()
        # self.conv0 = nn.ConvTranspose2d(ch, ch, kernel)
        # self.bn1 = nn.BatchNorm2d(ch)
        # self.drop1 = nn.Dropout2d(drop_rate)
        # self.conv1 = nn.ConvTranspose2d(ch, ch, kernel)
        # self.bn2 = nn.BatchNorm2d(ch)
        # self.drop2 = nn.Dropout2d(drop_rate)
        # self.conv2 = nn.ConvTranspose2d(ch, ch, kernel, padding=padding)

        n_in_ch = n_ch * 2

        self.conv = nn.ConvTranspose2d(n_ch * 2, n_ch, n_kernel, padding=n_padding)
        self.up = nn.Upsample(scale_factor=2)
        self.decode_block = nn.Sequential(
            nn.BatchNorm2d(n_ch),
            nn.Tanh(),
            nn.Dropout2d(drop_rate),
            nn.ConvTranspose2d(n_ch, n_ch, n_kernel, padding=n_padding),
            nn.BatchNorm2d(n_ch),
            nn.Tanh(),
            nn.Dropout2d(drop_rate),
            nn.ConvTranspose2d(n_ch, n_ch, n_kernel, padding=n_padding)
        )

    def forward(self, x, c):
        # x = self.bn1(x)
        # x = F.tanh(x)
        # x = self.drop1(x)
        # x = self.conv1(x)
        # x = self.bn2(x)
        # x = F.tanh(x)
        # x = self.drop2(x)
        # x = self.conv2(x)
        # return x
        x = self.conv(x)
        x = self.up(x)
        x = torch.cat([x, c], dim=0)
        return self.decode_block(x)

class UNet(nn.Module):
    def __init__(self, n_in_ch, n_out_ch, n_kernel=3, n_padding=1):
        super(UNet, self).__init__()

        self.conv0 = nn.Conv2d(n_in_ch, 8, n_kernel, padding=n_padding)
        self.tanh0 = nn.Tanh()
        self.enc0 = EncodeBlock(16)

        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.enc1 = EncodeBlock(32)

        self.pool2 = nn.AvgPool2d(2, stride=2)
        self.enc2 = EncodeBlock(64)

        self.pool3 = nn.AvgPool2d(2, stride=2)
        self.enc3 = EncodeBlock(128)

        self.dec2 = DecodeBlock(64)
        self.dec1 = DecodeBlock(32)
        self.dec0 = DecodeBlock(16)

        self.deconv0 = nn.ConvTranspose2d(16, n_out_ch, n_kernel, padding=n_padding)
        self.output = nn.Tanh()

    def forward(self, x):
        e0 = self.conv0(x)
        e0 = self.tanh0(e0)
        e0 = self.enc0(e0)

        e1 = self.pool1(e0)
        e1 = self.enc1(e1)

        e2 = self.pool2(e1)
        e2 = self.enc2(e2)

        e3 = self.pool3(e2)
        e3 = self.enc3(e3)

        d2 = self.dec2(e3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)

        out = self.deconv0(d0)
        out = self.output(out)
        return out
