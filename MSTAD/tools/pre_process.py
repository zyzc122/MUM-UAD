import torch
import torch.nn as nn
import torch.nn.functional as F

class PreNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(PreNet, self).__init__()

        self.conv_relu = nn.Sequential(nn.Conv2d(in_c, in_c*4, kernel_size=(3, 3), padding=1),
                                       nn.BatchNorm2d(in_c*4), nn.ReLU(inplace=True),
                                       nn.Conv2d(in_c*4, out_c, kernel_size=(3, 3), padding=1),)
        # self.up = nn.Sequential(nn.BatchNorm2d(2*in_c),  nn.ReLU(inplace=True),
        #                         nn.Upsample(scale_factor=2),
        #                         nn.Conv2d(2*in_c, out_c, kernel_size=(3, 3), padding=1),
        #                         nn.BatchNorm2d(out_c),  nn.ReLU(inplace=True))

        # self.reduce_conv = nn.Sequential(nn.Conv2d(2 * out_c, out_c, kernel_size=(1, 1), padding=0),
        #                                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    def forward(self, x):
        y = self.conv_relu(x)
        return y

class SEblock(nn.Module):  # Squeeze and Excitation block
    def __init__(self, channels, ratio=16):
        super(SEblock, self).__init__()
        channels = channels  # 输入的feature map通道数
        hidden_channels = channels * ratio  # 中间过程的通道数，原文reduction ratio设为16
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # avgpool
            nn.Conv2d(channels, hidden_channels, 1, 1, 0),  # 1x1conv，替代linear
            nn.ReLU(),  # relu
            nn.Conv2d(hidden_channels, channels, 1, 1, 0),  # 1x1conv，替代linear
            nn.Sigmoid()  # sigmoid，将输出压缩到(0,1)
        )

    def forward(self, x):
        weights = self.attn(x)  # feature map每个通道的重要性权重(0,1)，对应原文的sc
        return weights * x  # 将计算得到的weights与输入的feature map相乘


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes*16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes*16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = out.view(out.size(0), -1)
        out = self.conv2(out)
        return out


# def SENet18():
#     return SENet(PreActBlock, [2,2,2])
#
#
# def test():
#     net = SENet18()
#     y = net(torch.randn(1,3,32,32))
#     print(y.size())

# test()
