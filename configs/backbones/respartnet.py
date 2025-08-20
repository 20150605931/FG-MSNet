import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler


class ResPartNet(nn.Module):

    def __init__(self, part_num):
        super(ResPartNet, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, )
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, )
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, )

    def forward(self, x):

        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

