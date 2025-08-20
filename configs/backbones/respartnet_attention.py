import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from fightingcv_attention.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
from fightingcv_attention.attention.ECAAttention import ECAAttention
from fightingcv_attention.attention.SEAttention import SEAttention
from .RecurrentNet import *

class Attention_ResPartNet(nn.Module):

    def __init__(self, part_num):
        super(Attention_ResPartNet, self).__init__()

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
        
        # ##### self attention mechinism ######
        # self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        self.conv0 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(1)

        self.TokenAttention = TokenAttention(in_channels=16, ratio=16)


    # def forward(self, x):

    #     #### convert RGB image into gray image
    #     # 定义灰度转换的权重系数
    #     weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)

    #     # 将RGB图像转换为灰度图像
    #     x_gray = torch.sum(x * weights, dim=1, keepdim=True)
    #     #print("Gray image size:", x_gray.size())  # 应该输出 [32, 1, 512, 512]
    #     x = x_gray.squeeze(1)     # x 为（32,512,512）对应（batch_size, seq_length, embedding_dim）
    #     #print("attention input size:", x.size())


    #     ##### self attention mechinism ######
    #     # 自注意力计算，注意这里需要将输入数据转置，因为 MultiheadAttention 要求输入为 (seq_length, batch_size, embedding_dim)
    #     x_transposed = x.transpose(0, 1)  # 转置成 (seq_length, batch_size, embedding_dim)
    #     x_attention, _ = self.attention(x_transposed, x_transposed, x_transposed)

    #     # 输出结果的形状为 (seq_length, batch_size, embedding_dim)，需要转置回来
    #     x_attention = x_attention.transpose(0, 1)  # 转置成 (batch_size, seq_length, embedding_dim)
    #     #print("attention output size:", x.size())

    #     x_attention = x_attention.unsqueeze(1) 

    #     # 使用 torch.repeat 沿着通道维度复制数据
    #     # x = x_attention.repeat(1, 3, 1, 1)

    #     x = self.relu(self.bn0(self.conv0(x_attention)))
    #     #print("resnet input size:", x.size())

    #     features = self.resnet_conv(x)
    #     # features_c = torch.squeeze(self.pool_c(features))
    #     # features_e = torch.squeeze(self.pool_e(features))
    #     return features


    def forward(self, x):

        #### convert RGB image into gray image
        # 定义灰度转换的权重系数
        weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)

        # 将RGB图像转换为灰度图像
        x_gray = torch.sum(x * weights, dim=1, keepdim=True)
        #print("Gray image size:", x_gray.size())  # 应该输出 [32, 1, 512, 512]
        #x = x_gray.squeeze(1)     # x 为（32,512,512）对应（batch_size, seq_length, embedding_dim）
        
        x_gray = self.bn1(x_gray)
        print("attention input size:", x_gray)

        ##### self attention mechinism ######
        x = self.TokenAttention(x_gray)
        x = self.relu(self.bn0(self.conv0(x)))
        #print("resnet input size:", x.size())

        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features


class ResPartNet_Tokken_Space_Attention(nn.Module):

    def __init__(self, part_num):
        super(ResPartNet_Tokken_Space_Attention, self).__init__()

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
        
        # ##### self attention mechinism ######
        # self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        self.conv0 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

        self.SpatialAttention = SpatialAttention(kernel_size=7)
        self.TokenAttention = TokenAttention(in_channels=16, ratio=16)
        


    def forward(self, x):

        #### convert RGB image into gray image
        # 定义灰度转换的权重系数
        #weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)

        # 将RGB图像转换为灰度图像
        #x_gray = torch.sum(x * weights, dim=1, keepdim=True)  # 应该输出 [32, 1, 512, 512]
        #print("attention input size:", x.size())

        #####attention mechinism ######
        x = self.SpatialAttention(x)
        x = self.TokenAttention(x)
        x = self.relu(self.bn0(self.conv0(x)))
        #print("resnet input size:", x.size())

        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

class Self_Attention_ResPartNet(nn.Module):

    def __init__(self, part_num):
        super(Self_Attention_ResPartNet, self).__init__()

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
        
        # ##### self attention mechinism ######
        self.self_attention = SimplifiedScaledDotProductAttention(d_model=224, h=8)

        self.conv0 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

        # self.SpatialAttention = SpatialAttention(kernel_size=7)
        # self.TokenAttention = TokenAttention(in_channels=16, ratio=16)       

    def forward(self, x):

        ### convert RGB image into gray image
        #定义灰度转换的权重系数
        weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to('cuda:0')

        #将RGB图像转换为灰度图像
        x_gray = torch.sum(x * weights, dim=1, keepdim=True)  # 应该输出 [32, 1, 512, 512]
        #print("attention input size:", x.size())
        x = x_gray.squeeze(1)

        #####attention mechinism ######
        x = self.self_attention(x,x,x)
        x = x.unsqueeze(1)
        #print("AAAA", x.size())
        x = self.relu(self.bn0(self.conv0(x)))
        
        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

class TSA_ResPartNet(nn.Module):      

    '''''''''''
    Tokken_Space_Attention(TSA) - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(TSA_ResPartNet, self).__init__()

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
        

        self.SpatialAttention = SpatialAttention(kernel_size=7)
        self.TokenAttention = TokenAttention(in_channels=16, ratio=16)
        

    def forward(self, x):

        ##### TSA_attention mechinism ######
        x = self.SpatialAttention(x)
        x = self.TokenAttention(x)
        #print("resnet input size:", x.size())

        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

class Tokken_ResPartNet(nn.Module):      

    '''''''''''
    Tokken_Attention - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(Tokken_ResPartNet, self).__init__()

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
        

        self.TokenAttention = TokenAttention(in_channels=3, ratio=3)
        

    def forward(self, x):

        ##### Tokken_attention mechinism ######
        x = self.TokenAttention(x)
        #print("resnet input size:", x.size())

        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features


class Recurrent_Tokken_ResPartNet(nn.Module):      

    '''''''''''
    Recurrent_Tokken_Attention - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(Recurrent_Tokken_ResPartNet, self).__init__()

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
        

        self.TokenAttention = TokenAttention(in_channels=16, ratio=16)

        self.recurrentNet = RecurrentNet(num_classes = 5)
        
    def forward(self, images):
    
        ##### Tokken_attention mechinism ######
        x = self.TokenAttention(images)

        features = self.resnet_conv(x)

        attention_input = self.recurrentNet(features,images)
        recurrent_features = self.resnet_conv(attention_input)
        
        return [features, recurrent_features]

class ResPartNet_CBAM_Recurrent(nn.Module):      

    def __init__(self, part_num):
        super(ResPartNet_CBAM_Recurrent, self).__init__()

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
        
        self.ChannelAttention = ChannelAttention(in_channels=2048, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)

        self.recurrentNet = RecurrentNet(num_classes = 5)
        
    def forward(self, images):

        x = self.resnet_conv(images)
        #####attention mechinism #####
        x = self.ChannelAttention(x)
        features = self.SpatialAttention(x)

        attention_input = self.recurrentNet(features,images)
        recurrent_features = self.resnet_conv(attention_input)
        
        return [features, recurrent_features]


class Tokken_ResPartNet_CBAM_Recurrent_v2(nn.Module):      

    def __init__(self, part_num):
        super(Tokken_ResPartNet_CBAM_Recurrent_v2, self).__init__()

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
        
        self.ChannelAttention = ChannelAttention(in_channels=2048, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)

        self.TokenAttention = TokenAttention(in_channels=3, ratio=3)

        self.recurrentNet = RecurrentNet(num_classes = 5)
        
    def forward(self, images):
        
        X = self.TokenAttention(images)
        features = self.resnet_conv(X)
        features = self.ChannelAttention(features)
        features = self.SpatialAttention(features)

        attention_input = self.recurrentNet(features,X)
        recurrent_features = self.resnet_conv(attention_input)
        # recurrent_features = self.ChannelAttention(recurrent_features)
        # recurrent_features = self.SpatialAttention(recurrent_features)
        
        return [features, recurrent_features]


class Spectrum_ResPartNet(nn.Module):      

    '''''''''''
    Spectrum_Attention(ChannelAttention) - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(Spectrum_ResPartNet, self).__init__()

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
        

        self.ChannelAttention = ChannelAttention(in_channels=224, ratio=16)
        

    def forward(self, x):

        ### 将x的RT维度转换为通道维度
        x_transpose = x.permute(0, 2, 1, 3)
        
        ##### Channel_attention mechinism ######
        x_transpose = self.ChannelAttention(x_transpose)
        ### 将x_transpose维度还原
        x = x_transpose.permute(0, 2, 1, 3)
        
        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

class CBAM_ResPartNet(nn.Module):      

    '''''''''''
    CBAM_Attention(ChannelAttention-spatialAttention) - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(CBAM_ResPartNet, self).__init__()

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
        

        self.ChannelAttention = ChannelAttention(in_channels=224, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)

    def forward(self, x):

        ### 将x的RT维度转换为通道维度
        x_transpose = x.permute(0, 2, 1, 3)
        
        ##### Channel_attention mechinism ######
        x_transpose = self.ChannelAttention(x_transpose)
        ### 将x_transpose维度还原
        x = x_transpose.permute(0, 2, 1, 3)

        x = self.SpatialAttention(x)
        
        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

class spatial_Spectrum_ResPartNet(nn.Module):      

    '''''''''''
    spatial_Spectrum_Attention(ChannelAttention) - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(spatial_Spectrum_ResPartNet, self).__init__()

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
        
        self.SpatialAttention = SpatialAttention(kernel_size=7)
        self.ChannelAttention = ChannelAttention(in_channels=224, ratio=16)
        

    def forward(self, x):

        x = self.SpatialAttention(x)
        ### 将x的RT维度转换为通道维度
        x_transpose = x.permute(0, 2, 1, 3)
        
        ##### Channel_attention mechinism ######
        x_transpose = self.ChannelAttention(x_transpose)
        ### 将x_transpose维度还原
        x = x_transpose.permute(0, 2, 1, 3)
        
        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

class Spectrum_ECA_ResPartNet(nn.Module):      

    '''''''''''
    Spectrum_ECA_Attention - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(Spectrum_ECA_ResPartNet, self).__init__()

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
        

        self.ECAAttention = ECAAttention(kernel_size=3)
        

    def forward(self, x):

        ### 将x的RT维度转换为通道维度
        x_transpose = x.permute(0, 2, 1, 3)
        
        ##### Channel_attention mechinism ######
        x_transpose = self.ECAAttention(x_transpose)
        ### 将x_transpose维度还原
        x = x_transpose.permute(0, 2, 1, 3)
       
        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

class Spectrum_SE_ResPartNet(nn.Module):      

    '''''''''''
    Spectrum_SE_Attention - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(Spectrum_SE_ResPartNet, self).__init__()

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
        

        self.SEAttention = SEAttention(channel=224,reduction=8)
        

    def forward(self, x):

        ### 将x的RT维度转换为通道维度
        x_transpose = x.permute(0, 2, 1, 3)
        
        ##### Channel_attention mechinism ######
        x_transpose = self.SEAttention(x_transpose)
        ### 将x_transpose维度还原
        x = x_transpose.permute(0, 2, 1, 3)
       
        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features


class TSA_1_ResPartNet(nn.Module):      

    '''''''''''
    Tokken_Space_1_Attention(TSA_1) - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(TSA_1_ResPartNet, self).__init__()

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
        

        self.SpatialAttention = SpatialAttention(kernel_size=7)
        self.TokenAttention = TokenAttention(in_channels=16, ratio=16)
        

    def forward(self, x):

        ##### TSA_attention mechinism ######
        x = self.TokenAttention(x)
        x = self.SpatialAttention(x)
        #print("resnet input size:", x.size())

        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features


class TSA_2_ResPartNet(nn.Module):      

    '''''''''''
    Tokken_Space_2_Attention(TSA_1) - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(TSA_2_ResPartNet, self).__init__()

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

        self.conv0 = nn.Conv2d(3, 48, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        
        self.SpatialAttention = SpatialAttention(kernel_size=7)
        self.TokenAttention_1 = TokenAttention_1(in_channels=48, ratio=16)

        self.conv1 = nn.Conv2d(48, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(3)
       

    def forward(self, x):

        
        ##### TSA_attention mechinism ######
        x = self.SpatialAttention(x)
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.TokenAttention_1(x)
        x = self.relu(self.bn1(self.conv1(x)))
        #print("resnet input size:", x.size())

        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features


class ResPartNet_CBAM(nn.Module):

    def __init__(self, part_num):
        super(ResPartNet_CBAM, self).__init__()

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
        
        self.ChannelAttention = ChannelAttention(in_channels=2048, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)
        
    def forward(self, x):

        x = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))

        #####attention mechinism #####
        x = self.ChannelAttention(x)
        features = self.SpatialAttention(x)
      
        return features

class ResPartNet_CA(nn.Module):

    def __init__(self, part_num):
        super(ResPartNet_CA, self).__init__()

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
        
        # ##### self attention mechinism ######
        # self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        
        self.CA = CA(inp=2048, reduction=16)
       


    def forward(self, x):

        x = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))

        #####attention mechinism #####
        features = self.CA(x)
      
        return features

class TSA_ResPartNet_CBAM(nn.Module):      

    '''''''''''
    Tokken_Space_Attention(TSA) - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(TSA_ResPartNet_CBAM, self).__init__()

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
        
        #### TSA attention ######
        self.SpatialAttention = SpatialAttention(kernel_size=7)
        self.TokenAttention = TokenAttention(in_channels=16, ratio=16)

        ### CBAM attention #####
        self.ChannelAttention = ChannelAttention(in_channels=2048, ratio=16)
        self.SpatialAttention = SpatialAttention(kernel_size=7)

    def forward(self, x):

        ##### TSA_attention mechinism ######
        x = self.SpatialAttention(x)
        x = self.TokenAttention(x)
        #print("resnet input size:", x.size())

        x = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))

        #### CBAM attention #####
        x = self.ChannelAttention(x)
        features = self.SpatialAttention(x)

        return features

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_channels // 16, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out * x

class TokenAttention(nn.Module):
    """
    token注意力机制
    """

    def __init__(self, in_channels, ratio=16):
        super(TokenAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            # 全连接层
            # nn.Linear(in_planes, in_planes // ratio, bias=False),
            # nn.ReLU(),
            # nn.Linear(in_planes // ratio, in_planes, bias=False)

            # 利用1x1卷积代替全连接，避免输入必须尺度固定的问题，并减小计算量
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=3, keepdim=True) # （b,3,h,1）
        max_out, _ = torch.max(x, dim=3, keepdim=True)  
        out = avg_out + max_out
        out = self.fc(out)
        out = self.sigmoid(out)
        return out * x

class TokenAttention_1(nn.Module):
    """
    token注意力机制
    """

    def __init__(self, in_channels, ratio=16):
        super(TokenAttention_1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            # 全连接层
            # nn.Linear(in_planes, in_planes // ratio, bias=False),
            # nn.ReLU(),
            # nn.Linear(in_planes // ratio, in_planes, bias=False)

            # 利用1x1卷积代替全连接，避免输入必须尺度固定的问题，并减小计算量
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc(torch.mean(x, dim=3, keepdim=True))   #(32,3,224,1)
        max_out, _ = self.fc(torch.max(x, dim=3, keepdim=True))  
        out = avg_out + max_out   #(32,3,224,2) 
        out = self.sigmoid(out) 
        return out * x


class CA(nn.Module):
    def __init__(self, inp, reduction):
        super(CA, self).__init__()
        # h:height(行)   w:width(列)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (b,c,h,w)-->(b,c,h,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (b,c,h,w)-->(b,c,1,w)
 
 
        mip = max(8, inp // reduction)  # 论文作者所用
        # mip =  inp // reduction  # 博主所用   reduction = int(math.sqrt(inp))
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
 
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x):
        identity = x
 
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # (b,c,h,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b,c,w,1)
 
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
 
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
 
        out = identity * a_w * a_h
 
        return out


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6












