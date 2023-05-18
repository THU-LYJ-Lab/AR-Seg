import torch
import torch.nn as nn

import torch
from torch import nn
import torch.nn.functional as F

# import model.resnet as models
from model.extractors import resnet18, resnet50, resnet101, resnet152
from model.attention import *

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, feat_dim=2048 ,use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(PSPNet, self).__init__()
        assert layers in [18, 50, 101, 152]
        assert feat_dim % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        if layers == 50:
            resnet = resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = resnet101(pretrained=pretrained)
        elif layers == 18:
            resnet = resnet18(pretrained=pretrained)
        else:
            resnet = resnet152(pretrained=pretrained)
        
        if layers==18:
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = feat_dim
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(feat_dim//2, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        n,c,h,w = x.size()
        # assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        # h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        # w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            # main_loss = self.criterion(x, y)
            # aux_loss = self.criterion(aux, y)
            # return x.max(1)[1], main_loss, aux_loss
            return x, aux
        else:
            return x,

class PSPNetWithFuse(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, feat_dim=2048 ,use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255),
                 pretrained=True,  attention_type='local', atten_k=7):
        super(PSPNetWithFuse, self).__init__()
        assert layers in [18, 50, 101, 152]
        assert feat_dim % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        if layers == 50:
            resnet = resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = resnet101(pretrained=pretrained)
        elif layers == 18:
            resnet = resnet18(pretrained=pretrained)
        else:
            resnet = resnet152(pretrained=pretrained)
        
        if layers==18:
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = feat_dim
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2

        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )

        self.final_conv = self.cls[-1]

        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(feat_dim//2, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
        
        self.middle_dim = 512
        if attention_type == 'local':
            self.fuse_attention = MyAttention(self.middle_dim, kH=atten_k, kW=atten_k)

    def forward(self, x, mode='normal', ref_p=None):
        if mode == 'normal':
            n,c,h,w = x.size()
            # assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
            # h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
            # w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x_tmp = self.layer3(x)
            x = self.layer4(x_tmp)
            if self.use_ppm:
                x = self.ppm(x)
            p = self.cls[:-1](x)
            x = self.cls[-1:](p)
            if self.zoom_factor != 1:
                x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        elif mode=='merge':
            n,c,h,w = x.size()

            x_tmp, p = self.forward_phase1(x)

            x, p = self.forward_phase2(p, ref_p)

            # return x, out_cls, out_p

        # if self.training:
        aux = self.aux(x_tmp)
        if self.zoom_factor != 1:
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
        # main_loss = self.criterion(x, y)
        # aux_loss = self.criterion(aux, y)
        # return x.max(1)[1], main_loss, aux_loss
        return x, aux, p
        # else:
        #     return x, aux, p
    
    def forward_phase1(self, x):
        n,c,h,w = x.size()

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        p = self.cls[:-1](x)

        return x_tmp, p

    def forward_phase2(self, p, ref_p):
        # import pdb; pdb.set_trace()
        N, C, H, W = ref_p.shape

        # import pdb; pdb.set_trace()
        p = self.fuse_attention(ref_p, p)
        # p = ref_p

        out = self.final_conv(p)
        # out = F.interpolate(out, (H,W), mode='bilinear', align_corners=True)

        # out = self.final_logsoftmax(out)

        return out, p
        

        

        
if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    input = torch.rand(4, 3, 360, 480).cuda()
    ref_p = torch.rand(4, 512, 90, 120).cuda()
    model = PSPNetWithFuse(layers=18, bins=(1, 2, 3, 6), dropout=0.1, classes=19, zoom_factor=8, feat_dim=512, use_ppm=True, pretrained=True).cuda()
    # model.eval()
    print(model)
    output = model(input,mode='merge',ref_p = ref_p)
    print(len(output))
    print('PSPNet', output[0].size(), output[1].size(), output[2].size())