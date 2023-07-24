import math
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn as nn
import torch
from models.axialnet import AxialAttention

class SRGN(nn.Module):
    def __init__(self, model, feature_size, classes_num):
        super(SRGN, self).__init__()

        self.features = model
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=True)

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.cam = nn.Sequential(
            BasicConv(self.num_ftrs//2, 1, kernel_size=1, stride=1, padding=0, relu=True),
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.axis = AxialAttention(in_planes=1024, out_planes=1024, kernel_size=14, groups=8)
        self.axisw = AxialAttention(in_planes=1024, out_planes=1024, kernel_size=14, groups=8, width=True)     
        self.conv1 = nn.Conv2d(1024, 1, 1)
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )
        """ 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
         """

    def forward(self, x):
        xf5 = self.features(x)

        xl3 = self.conv_block3(xf5)
        xlv = self.max3(xl3)
        xlv = xhv.view(xl3.size(0), -1)
        xv3 = self.classifier1(xlv)

        xlh = self.axis(xl3)
        xhv = self.max3(xlh)
        xhv = xhv.view(xl3.size(0), -1)
        xh3 = self.classifier2(xhv)


        xlw = self.axisw(xlh)
        xwv = self.max3(xlw)
        xwv = xl3.view(xwv.size(0), -1)
        xw3 = self.classifier3(xwv)

        nwc = self.cam(xlw)
        xlf = torch.mul(xlw, nwc).view(xlw.size(0), xlw.size(1), xlw.size(2), xlw.size(3))
        xlf = self.max3(xlf)
        xlf = xhv.view(xl3.size(0), -1)
        xf3 = self.classifier2(xlf)


          
        return xv3, xh3, xw3, xf3
    
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
