import torch.nn as nn
import torch


class PartGlobalAveragePooling_recurrent_no_BP(nn.Module):

    def __init__(self, part_num):
        super(PartGlobalAveragePooling_recurrent_no_BP, self).__init__()
        self.avgpool_c = nn.AdaptiveAvgPool2d((part_num, 1))
        self.avgpool_r = nn.AdaptiveAvgPool2d((1, part_num))
        dropout = nn.Dropout(p=0.5)
        self.maxpool = nn.AdaptiveAvgPool2d((1, 1))

        self.pool_c = nn.Sequential(self.avgpool_c, dropout)
        self.pool_r = nn.Sequential(self.avgpool_r, dropout)
        #self.pool_g = nn.Sequential(self.maxpool_g)

        #self.Bilinear_pool = Bilinear_pool(self,x)

    def Bilinear_pool(self,x):
        #x = self.features(x) 
        #print(x.shape)
        batch_size = x.size(0) 
        x = x.view(batch_size, 2048, x.size(2) ** 2) #将（batch_size,channel ,h,w）变为（batch_size,channel,h*w）维度的 
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / x.size(2) ** 2)  #torch.transpose(x, 1, 2))转置矩阵，将维度1和2的转置 
        # torch.bmm做外积 torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,h),注意两个tensor的维度必须为3. 得到(batch_szie,512,512) # / 28 ** 2平均池化 
        #print(x.shape) 
        x=x.view(batch_size, -1)    #view(batch_size, -1)变成一维的张量 
        #normalize标准化,开方和归一化操作 
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10)) # feature = feature.view(feature.size(0), -1) 
        #x = self.classifiers(x) 
        return x

    def forward(self, features):
        ini_features, recurrent_features = features
        features_part_c = self.pool_c(ini_features)
        features_part_r = self.pool_r(ini_features)
        # ini_features = self.Bilinear_pool(ini_features)
        # recurrent_features = self.Bilinear_pool(recurrent_features)
        ini_features = self.maxpool(ini_features)
        recurrent_features = self.maxpool(recurrent_features)
        return features_part_c, features_part_r, ini_features, recurrent_features