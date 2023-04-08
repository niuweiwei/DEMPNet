import math
import os.path

import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, isReLU=True,*args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        # self.bn = BatchNorm2d(out_chan)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.isRelu = isReLU
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if(self.isRelu):
            x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu
        self.init_weight()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu
        self.init_weight()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class SpatialAttention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(SpatialAttention, self).__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        # 空间权重归一化
        x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = inputs * x
        return outputs

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        #(inplanes=512,branch_planes=128,outplanes=128)
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm2d(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )
        # self.refine_context = SE(in_chnls=branch_planes * 5 ,ratio=4)

        self.init_weight()

    def forward(self, x):

        #x = self.downsample(x)
        width = x.shape[-1] # W/64
        height = x.shape[-2]  # H/64
        x_list = []

        x_list.append(self.scale0(x)) #(B,128,H/64,W/64) x_list[0]
        x_list.append(self.process1((F.interpolate(self.scale1(x),  #scale1->(B,128,H/128,W/128)->F.interpolate+x_list[0]->(B,128,H/64,W/64)->process1->(B,128,H/64,W/64) x_list[1]
                        size=[height, width],
                        mode='bilinear')+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x), #scale2->(B,128,H/256,W/256)->F.interpolate+x_list[1]->(B,128,H/64,W/64)->process2->(B,128,H/64,W/64) x_list[2]
                        size=[height, width],
                        mode='bilinear')+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x), #scale3->(B,128,H/512,W/512)->F.interpolate+x_list[2]->(B,128,H/64,W/64)->process3->(B,128,H/64,W/64) x_list[3]
                        size=[height, width],
                        mode='bilinear')+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x), #scale4->(B,128,1,1)->F.interpolate+x_list[3]->(B,128,H/64,W/64)->process4->(B,128,H/64,W/64) x_list[4]
                        size=[height, width],
                        mode='bilinear')+x_list[3])))
        # x_multi_context = torch.cat(x_list, 1)
        # x_refine = torch.mul(x_multi_context, self.refine_context(x_multi_context))
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x) #cat->(B,128*5,H/64,W/64)->[compression->(B,128,H/64,W/64)]+[shortcut->(B,128,H/64,W/64)]->(B,128,H/64,W/64)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        # inplanes=128,interplanes=64,outplanes=num_classes
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor
        self.init_weight()

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))  #(B,interplanes,H/8,W/8) = (B,64,H/8,W/8)
        out = self.conv2(self.relu(self.bn2(x))) #(B,outplanes,H/8,W/8) = (B,num_classes,H/8,W/8)

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear')
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)



class DualResNet(nn.Module):

    def __init__(self, block, layers, num_classes=19, planes=64, spp_planes=128, head_planes=128, augment=True, detail = False):
        super(DualResNet, self).__init__()
        #planes=32 spp_planes=128 head_planes=64
        highres_planes = planes * 2
        self.augment = augment
        self.detail = detail

        self.conv1 =  nn.Sequential(
                          nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, planes, planes, layers[0])

        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)

        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)

        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)

        self.compression3 = nn.Sequential(
                                          nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        self.down3 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   )

        self.down4 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 8, momentum=bn_mom),
                                   )

        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)


        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)


        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        self.spatial_attention = SpatialAttention(kernel_size=3)

        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)
        if self.detail:
            self.seghead_boundary = segmenthead(highres_planes, head_planes, 1)


        self.ffm = FeatureFusionModule(256,128)
        
        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        self.init_weight()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []

        x = self.conv1(x) # (B,32,H/4,W/4)

        x = self.layer1(x) # conv2 (B,32,H/4,W/4) layers[0]
        layers.append(x)

        x = self.layer2(self.relu(x)) # conv3 (B,64,H/8,W/8) layers[1]
        layers.append(x)

        if self.detail:
            boundary = x


        x = self.layer3(self.relu(x)) # conv4-low-resolution (B,128,H/16,W/16) layers[2]

        layers.append(x)
        x_ = self.layer3_(self.relu(layers[1])) # conv4-high-resolution (B,64,H/8,W/8)

        x_ = x_ + self.spatial_attention(x_)
       

        x = x + self.down3(self.relu(x_)) # conv4 hight-to-low (B,64,H/8,W/8)->(B,128,H/16,W/16)
        x_ = x_ + F.interpolate(
                        self.compression3(self.relu(layers[2])),
                        size=[height_output, width_output],
                        mode='bilinear')   # conv4 low-to-high (B,128,H/16,W/16)[compression3]->(B,64,H/16,W/16)[F.interpolate]->(B,64,H/8,W/8)

        if self.augment:
            temp = x_

        x = self.layer4(self.relu(x)) # conv5-low-resolution (B,256,H/32,W/32) (BasicBlock)

        layers.append(x)
        x_ = self.layer4_(self.relu(x_)) # conv5-high-resolution (B,64,H/8,W/8) (BasicBlock)
        x_ = x_ + self.spatial_attention(x_)
     
        x = x + self.down4(self.relu(x_)) # conv5-high-to-low (B,64,H/8,W/8)[3×3,s=2,128 + 3×3,s=2,256]->(B,256,H/32,W/32)
        x_ = x_ + F.interpolate(
                        self.compression4(self.relu(layers[3])),
                        size=[height_output, width_output],
                        mode='bilinear') # conv5-low-to-high (B,256,H/32,W/32)[compression4]->(B,64,H/32,W/32)[F.interpolate]->(B,64,H/8,W/8)


        x_ = self.layer5_(self.relu(x_))# conv5-high-resolution (B,64,H/8,W/8) [Bottleneck stride=1 block_num=1]->(B,128,H/8,W/8)
        x_ = x_ + self.spatial_attention(x_)

        x = F.interpolate(
                        self.spp(self.layer5(self.relu(x))), # layer5:low-resolution (B,256,H/32,W/32)[Bottleneck stride=2 block_num=1] -> spp(B,512,H/64,W/64):将low-resolution的输出送入DAPPM->(B,128,H/64,W/64)
                        size=[height_output, width_output],  # F.interpolate->(B,128,H/8,W/8)
                        mode='bilinear')
       

        # low-resolution与 high-resolution最终的输出通过 FFM模块进行融合
        # low-resolution x : (B,128,H/8,W/8)    high-resolution x_ : _(B,128,H/8,W/8)
        x_fusion = self.ffm(x_,x)

        # x_ = self.final_layer(x + x_) # x(B,128,H/8,W/8)+x_(B,128,H/8,W/8)->final_layer(B,128,H/8,W/8)->(B,num_classes,H/8,W/8)
        x_ = self.final_layer(x_fusion)

        if self.detail:
            x_boundary = self.seghead_boundary(boundary)
        if self.augment:
            x_extra = self.seghead_extra(temp)

        if self.detail and self.augment:
            return [x_extra,x_], x_boundary
        elif self.detail and not self.augment:
            return x_, x_boundary
        elif not self.detail and self.augment:
            return [x_extra,x_]

        else:
            return x_

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


def DualResNet_imagenet(cfg, pretrained=False):

    is_detail = cfg.LOSS.USE_DETAIL_LOSS

    is_augment = cfg.LOSS.USE_AUGMENT

    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=cfg.DATASET.NUM_CLASSES, planes=32, spp_planes=128, head_planes=64, augment=is_augment, detail=is_detail)
    if pretrained:
        root = os.path.abspath(os.path.join(os.getcwd()))


    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=cfg.DATASET.NUM_CLASSES, planes=32, spp_planes=128, head_planes=64, augment=is_augment, detail=is_detail)
    if pretrained:
        root = os.path.abspath(os.getcwd())
        pretrained_path = os.path.join(root,cfg.MODEL.PRETRAINED)
        pretrained_state = torch.load(pretrained_path, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        model.load_state_dict(model_dict, strict = False)
    return model

def get_seg_model(cfg, **kwargs):

    model = DualResNet_imagenet(cfg, pretrained=True)
    return model


if __name__ == '__main__':
    x = torch.rand(4, 3, 800, 800)
    net = DualResNet_imagenet(pretrained=False)
    y = net(x)
    print(y.shape)