import torch
import torch.nn as nn

class SegmentationModule(nn.Module):
    def __init__(self, num_classes=21):
        super(SegmentationModule, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 40, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(40)
        self.hs1 = nn.Hardswish()

        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(40, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.Hardswish(),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, groups=24),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 40, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(40),
            nn.Hardswish()
        )

        self.mib_conv2 = self._make_mib_conv(40, 40, 2, 1, 1)
        self.mib_conv3 = self._make_mib_conv(40, 40, 1, 2, 2)
        self.mib_conv4 = self._make_mib_conv(40, 40, 2, 2, 2)
        self.mib_conv5 = self._make_mib_conv(40, 40, 1, 4, 4)
        self.mib_conv6 = self._make_mib_conv(40, 40, 1, 4, 4)
        self.mib_conv7 = self._make_mib_conv(40, 112, 1, 4, 4)
        self.mib_conv8 = self._make_mib_conv(112, 112, 1, 8, 8)
        self.mib_conv9 = self._make_mib_conv(112, 160, 1, 8, 8)

        self.aspp = nn.Sequential(
            nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(960),
            nn.Hardswish(),
            nn.Conv2d(960, 960, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(960),
            nn.Hardswish(),
            nn.Conv2d(960, 960, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(960),
            nn.Hardswish(),
            nn.Conv2d(960, 960, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(960),
            nn.Hardswish()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(960, 40, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(40),
            nn.Hardswish(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(40),
            nn.Hardswish(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(40),
            nn.Hardswish()
        )

        # Logits
        self.logits = nn.Conv2d(40, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        x = self.hs1(self.bn1(self.conv1(x)))
        x = self.bottleneck1(x)
        x = self.mib_conv2(x)
        x = self.mib_conv3(x)
        x = self.mib_conv4(x)
        x = self.mib_conv5(x)
        x = self.mib_conv6(x)
        x = self.mib_conv7(x)
        x = self.mib_conv8(x)
        x = self.mib_conv9(x)
        x = self.aspp(x)

        # Decoder
        x = self.decoder(x)

        # Logits
        x = self.logits(x)

        return x

def _make_mib_conv(self, in_channels, out_channels, stride, dilation, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.Hardswish()
    )

