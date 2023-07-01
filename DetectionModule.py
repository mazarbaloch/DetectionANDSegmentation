import torch.nn as nn
import torch.nn.functional as F

class DetectionModule(nn.Module):
    def __init__(self, num_classes=2):
        super(DetectionModule, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 40, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(40)
        self.bneck1 = nn.Sequential(
            nn.Conv2d(40, 24, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(24),
            nn.Hardswish(inplace=True)
        )
        self.bneck2 = nn.Sequential(
            nn.Conv2d(24, 40, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(40),
            nn.Hardswish(inplace=True)
        )
        self.bneck3 = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(40),
            nn.Hardswish(inplace=True)
        )
        self.bneck4 = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(40),
            nn.Hardswish(inplace=True)
        )
        self.bneck5 = nn.Sequential(
            nn.Conv2d(40, 112, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(112),
            nn.Hardswish(inplace=True)
        )
        self.bneck6 = nn.Sequential(
            nn.Conv2d(112, 112, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(112),
            nn.Hardswish(inplace=True)
        )
        self.bneck7 = nn.Sequential(
            nn.Conv2d(112, 160, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(160),
            nn.Hardswish(inplace=True)
        )
        
        # Feature Pyramid Network
        self.fpn = nn.Conv2d(160, 256, kernel_size=1)
        
        # Regional Proposal Network
        self.rpn = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Region of Interest Align
        self.roi_align = nn.AdaptiveAvgPool2d(output_size=7)
        
        # Classification layer
        self.cls_layer = nn.Conv2d(256, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        
        # Bounding box regressor
        self.bbox_layer = nn.Conv2d(256, 4, kernel_size=1)

    def forward(self, x):
        # Convolutional layers
        x = self.bn1(self.conv1(x))
        x = self.bneck1(x) + x
        x = self.bneck2(x)
        self.bneck3(x) + x
        x = self.bneck4(x) + x
        x = self.bneck5(x)
        x = self.bneck6(x) + x
        x = self.bneck7(x)
        
        # Feature Pyramid Network
        fpn_out = self.fpn(x)
        
        # Regional Proposal Network
        rpn_out = self.rpn(fpn_out)
        
        # Region of Interest Align
        roi_align_out = self.roi_align(fpn_out)
        
        # Classification layer
        cls_out = self.cls_layer(roi_align_out)
        cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
        cls_out = self.softmax(cls_out.view(-1, cls_out.shape[-1]))
        cls_out = cls_out.view(cls_out.shape[0], cls_out.shape[1], cls_out.shape[2], -1)
        
        # Bounding box regressor
        bbox_out = self.bbox_layer(roi_align_out)
        
        return cls_out, bbox_out