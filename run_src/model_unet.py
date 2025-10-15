import torch
import torch.nn as tnn

class TNN(tnn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = tnn.Sequential(
            tnn.Conv2d(in_channels, out_channels, 3, padding=1),
            tnn.BatchNorm2d(out_channels),
            tnn.ReLU(inplace=True),
            tnn.Conv2d(out_channels, out_channels, 3, padding=1),
            tnn.BatchNorm2d(out_channels),
            tnn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(tnn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.down1 = TNN(n_channels, 64)
        self.pool1 = tnn.MaxPool2d(2)
        self.down2 = TNN(64, 128)
        self.pool2 = tnn.MaxPool2d(2)
        self.down3 = TNN(128, 256)
        self.pool3 = tnn.MaxPool2d(2)
        self.down4 = TNN(256, 512)
        self.pool4 = tnn.MaxPool2d(2)

        self.middle = TNN(512, 1024)

        self.up1 = tnn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = TNN(1024, 512)
        self.up2 = tnn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = TNN(512, 256)
        self.up3 = tnn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = TNN(256, 128)
        self.up4 = tnn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = TNN(128, 64)

        self.out = tnn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        mid = self.middle(self.pool4(d4))

        u1 = self.up1(mid)
        u1 = torch.cat([u1, d4], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)
        u4 = self.conv4(u4)

        return torch.sigmoid(self.out(u4))

