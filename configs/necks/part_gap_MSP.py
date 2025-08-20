import torch.nn as nn
import torch

# ########### original ###############
# class PartGlobalAveragePooling(nn.Module):

#     def __init__(self, part_num):
#         super(PartGlobalAveragePooling, self).__init__()
#         self.avgpool_c = nn.AdaptiveAvgPool2d((part_num, 1))
#         dropout = nn.Dropout(p=0.5)
#         self.maxpool_g = nn.AdaptiveAvgPool2d((1, 1))

#         self.pool_c = nn.Sequential(self.avgpool_c, dropout)
#         self.pool_g = nn.Sequential(self.maxpool_g)

#     def forward(self, features):
#         features_part = torch.squeeze(self.pool_c(features))
#         features_global = torch.squeeze(self.pool_g(features))
#         return features_part, features_global

class PartGlobalAveragePooling_MSP(nn.Module):

    def __init__(self, part_num):
        super(PartGlobalAveragePooling_MSP, self).__init__()
        self.avgpool_c = nn.AdaptiveAvgPool2d((part_num, 1))
        self.avgpool_r = nn.AdaptiveAvgPool2d((1, part_num))
        dropout = nn.Dropout(p=0.5)
        self.maxpool_g = nn.AdaptiveAvgPool2d((1, 1))

        self.pool_c = nn.Sequential(self.avgpool_c, dropout)
        self.pool_r = nn.Sequential(self.avgpool_r, dropout)
        self.pool_g = nn.Sequential(self.maxpool_g)

    def forward(self, features):
        features_part_c = self.pool_c(features)
        features_part_r = self.pool_r(features)
        features_global = self.pool_g(features)
        return features_part_c, features_part_r, features_global

# class PartGlobalAveragePooling(nn.Module):

#     def __init__(self, part_num):
#         super(PartGlobalAveragePooling, self).__init__()
#         self.avgpool_c = nn.AdaptiveAvgPool2d((part_num, 1))
#         self.avgpool_r = nn.AdaptiveAvgPool2d((1, part_num))
#         dropout = nn.Dropout(p=0.5)
#         self.maxpool_g = nn.AdaptiveAvgPool2d((1, 1))

#         self.pool_c = nn.Sequential(self.avgpool_c, dropout)
#         self.pool_r = nn.Sequential(self.avgpool_r, dropout)
#         #self.pool_g = nn.Sequential(self.maxpool_g)

#         #self.Bilinear_pool = Bilinear_pool(self,x)

#     def Bilinear_pool(self,x):
#         #x = self.features(x) 
#         #print(x.shape)
#         batch_size = x.size(0) 
#         x = x.view(batch_size, 2048, 14 ** 2) #将（batch_size,channel ,h,w）变为（batch_size,channel,h*w）维度的 
#         x = (torch.bmm(x, torch.transpose(x, 1, 2)) / 14 ** 2)  #torch.transpose(x, 1, 2))转置矩阵，将维度1和2的转置 
#         # torch.bmm做外积 torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,h),注意两个tensor的维度必须为3. 得到(batch_szie,512,512) # / 28 ** 2平均池化 
#         #print(x.shape) 
#         x=x.view(batch_size, -1)    #view(batch_size, -1)变成一维的张量 
#         #normalize标准化,开方和归一化操作 
#         x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10)) # feature = feature.view(feature.size(0), -1) 
#         #x = self.classifiers(x) 
#         return x

#     def forward(self, features):
#         features_part_c = self.pool_c(features)
#         features_part_r = self.pool_r(features)
#         fusion_features = self.Bilinear_pool(features)
#         return features_part_c, features_part_r, fusion_features


