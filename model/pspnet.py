import torch
torch.manual_seed(233)

from torch import nn
from torch.nn import functional as F

import model.extractors as extractors

from model.attention import *

import time


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, input_channel=3, n_classes=18, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained, input_channel=input_channel)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        # self.final = nn.Sequential(
        #     nn.Conv2d(64, n_classes, kernel_size=1),
        #     nn.LogSoftmax()
        # )

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        self.final_logsoftmax = nn.LogSoftmax()

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        N,C,H,W = x.shape

        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        # import pdb; pdb.set_trace()
        out = self.final_conv(p)
        out = F.interpolate(out, (H,W), mode='bilinear', align_corners=True)

        out = self.final_logsoftmax(out)

        return out, self.classifier(auxiliary), p


class PSPNetWithFuse(nn.Module):
    def __init__(self, input_channel=3, n_classes=18, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True, attention_type='local', atten_k=7):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained, input_channel=input_channel)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        # self.final = nn.Sequential(
        #     nn.Conv2d(64, n_classes, kernel_size=1),
        #     nn.LogSoftmax()
        # )

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        self.final_logsoftmax = nn.LogSoftmax()

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

        self.middle_dim = 64
        self.attention_type = attention_type

        if attention_type == 'global':
            self.fuse_attention = MyAttentionGlobal(self.middle_dim, kScale=atten_k)
        elif attention_type == 'local':
            self.fuse_attention = MyAttention(self.middle_dim, kH=atten_k, kW=atten_k)
        elif attention_type == 'localNoGroup':
            self.fuse_attention = MyAttentionNoGroup(self.middle_dim, kH=atten_k, kW=atten_k)
        elif attention_type == 'local1':
            self.fuse_attention = MyAttentionV1(self.middle_dim, kH=atten_k, kW=atten_k)
        elif attention_type == 'local2':
            self.fuse_attention = MyAttentionV2(self.middle_dim, kH=atten_k, kW=atten_k)
        elif attention_type == 'local3':
            self.fuse_attention = MyAttentionV3(self.middle_dim, kH=atten_k, kW=atten_k)
        elif attention_type == 'local4':
            self.fuse_attention = MyAttentionV4(self.middle_dim, kH=atten_k, kW=atten_k)
        elif attention_type == 'local5':
            self.fuse_attention = MyAttentionV5(self.middle_dim, kH=atten_k, kW=atten_k)
        elif attention_type == 'local6':
            self.fuse_attention = MyAttentionV6(self.middle_dim, kH=atten_k, kW=atten_k)
        elif attention_type == 'localNew':
            self.fuse_attention = MyAttentionLocalNew(self.middle_dim, kH=atten_k, kW=atten_k)
        elif attention_type == 'no':
            self.fuse_attention = MyAttentionNo(self.middle_dim)
        elif attention_type == 'upsample':
            self.fuse_attention = MyAttentionUpsample(self.middle_dim)
        elif attention_type == 'globalNoGroup':
            self.fuse_attention = MyAttentionGlobalNoGroup(self.middle_dim)
        elif attention_type == 'conv':
            self.fuse_attention = ConvFusion(self.middle_dim)
        elif attention_type == 'localOnly':
            self.fuse_attention = MyAttentionLocalOnly(self.middle_dim, kH=atten_k, kW=atten_k)

    def forward(self, x, mode='normal', ref_p=None):
        # import pdb; pdb.set_trace()
        if mode=='normal':
            N,C,H,W = x.shape

            f, class_f = self.feats(x) 
            p = self.psp(f)
            p = self.drop_1(p)

            p = self.up_1(p)
            p = self.drop_2(p)

            p = self.up_2(p)
            p = self.drop_2(p)

            p = self.up_3(p)
            p = self.drop_2(p)

            auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

            # import pdb; pdb.set_trace()
            out = self.final_conv(p)
            out = F.interpolate(out, (H,W), mode='bilinear', align_corners=True)

            out = self.final_logsoftmax(out)

            return out, self.classifier(auxiliary), p
        elif mode=='merge':
            out_cls, out_p = self.forward_phase1(x)

            out, out_p = self.forward_phase2(out_p, ref_p)

            return out, out_cls, out_p
  
    def forward_phase1(self, x):
        # import pdb; pdb.set_trace()
        N,C,H,W = x.shape

        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return self.classifier(auxiliary), p

    def forward_phase2(self, p, ref_p):
        # import pdb; pdb.set_trace()
        N, C, H, W = ref_p.shape

        p = self.fuse_attention(ref_p, p)
        # p = ref_p

        out = self.final_conv(p)
        out = F.interpolate(out, (H,W), mode='bilinear', align_corners=True)

        out = self.final_logsoftmax(out)

        return out, p
        

class PSPNetWithFuseV2(nn.Module):
    def __init__(self, input_channel=3, n_classes=18, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True, attention_type='local', atten_k=7):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained, input_channel=input_channel)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        # self.final = nn.Sequential(
        #     nn.Conv2d(64, n_classes, kernel_size=1),
        #     nn.LogSoftmax()
        # )

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        self.final_logsoftmax = nn.LogSoftmax()

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

        self.middle_dim = 512
        self.attention_type = attention_type

        if attention_type == 'local':
            self.fuse_attention = MyAttention(self.middle_dim, kH=atten_k, kW=atten_k)

    def forward(self, x, mode='normal', ref_p=None):
        # import pdb; pdb.set_trace()
        if mode=='normal':
            N,C,H,W = x.shape

            f, class_f = self.feats(x) 
            p = self.psp(f)
            p = self.drop_1(p)

            p = self.up_1(p)
            p = self.drop_2(p)

            p = self.up_2(p)
            p = self.drop_2(p)

            p = self.up_3(p)
            p = self.drop_2(p)

            auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

            # import pdb; pdb.set_trace()
            out = self.final_conv(p)
            out = F.interpolate(out, (H,W), mode='bilinear', align_corners=True)

            out = self.final_logsoftmax(out)

            return out, self.classifier(auxiliary), f
        elif mode=='merge':
            out_cls, out_p = self.forward_phase1(x)

            out, out_p = self.forward_phase2(out_p, ref_p)

            return out, out_cls, out_p


    
    def forward_phase1(self, x):
        # import pdb; pdb.set_trace()
        N,C,H,W = x.shape

        f, class_f = self.feats(x) 

        # import pdb; pdb.set_trace()

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return self.classifier(auxiliary), f

    def forward_phase2(self, p, ref_p):
        # import pdb; pdb.set_trace()
        N, C, H, W = ref_p.shape

        f = self.fuse_attention(ref_p, p)

        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
   
        # p = ref_p

        out = self.final_conv(p)
        out = F.interpolate(out, (H,W), mode='bilinear', align_corners=True)

        out = self.final_logsoftmax(out)

        return out, f
        

class PSPNetWithFuseV3(nn.Module):
    def __init__(self, input_channel=3, n_classes=18, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True, attention_type='local', atten_k=7):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained, input_channel=input_channel)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        # self.final = nn.Sequential(
        #     nn.Conv2d(64, n_classes, kernel_size=1),
        #     nn.LogSoftmax()
        # )

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        self.final_logsoftmax = nn.LogSoftmax()

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

        self.middle_dim = 64
        self.attention_type = attention_type

        if attention_type == 'local':
            self.fuse_attention = MyAttention(self.middle_dim, kH=atten_k, kW=atten_k)

    def forward(self, x, mode='normal', ref_p=None):
        # import pdb; pdb.set_trace()
        if mode=='normal':
            N,C,H,W = x.shape

            x = self.feats.conv1(x)
            x = self.feats.bn1(x)
            x = self.feats.relu(x)
            mid_output = self.feats.maxpool(x)

            x = self.feats.layer1(mid_output)
            x = self.feats.layer2(x)
            class_f = self.feats.layer3(x)
            f = self.feats.layer4(class_f)

            p = self.psp(f)
            p = self.drop_1(p)

            p = self.up_1(p)
            p = self.drop_2(p)

            p = self.up_2(p)
            p = self.drop_2(p)

            p = self.up_3(p)
            p = self.drop_2(p)

            auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

            # import pdb; pdb.set_trace()
            out = self.final_conv(p)
            out = F.interpolate(out, (H,W), mode='bilinear', align_corners=True)

            out = self.final_logsoftmax(out)

            return out, self.classifier(auxiliary), mid_output
        elif mode=='merge':
            out_p = self.forward_phase1(x)

            out_p = out_p[0]

            out, out_cls, out_p = self.forward_phase2(out_p, ref_p)

            return out, out_cls, out_p


    
    def forward_phase1(self, x):
        # import pdb; pdb.set_trace()
        N,C,H,W = x.shape

        x = self.feats.conv1(x)
        x = self.feats.bn1(x)
        x = self.feats.relu(x)
        f = self.feats.maxpool(x)

        
        return [f]

    def forward_phase2(self, p, ref_p):
        # import pdb; pdb.set_trace()
        N, C, H, W = ref_p.shape

        mid_output = self.fuse_attention(ref_p, p)
        

        x = self.feats.layer1(mid_output)
        x = self.feats.layer2(x)
        class_f = self.feats.layer3(x)
        f = self.feats.layer4(class_f)

        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        # p = ref_p

        out = self.final_conv(p)
        out = F.interpolate(out, (H,W), mode='bilinear', align_corners=True)

        out = self.final_logsoftmax(out)

        return out, self.classifier(auxiliary), mid_output
        



if __name__ == "__main__":
    # model = PSPNet(sizes=(1, 2, 3, 6), n_classes=12, psp_size=512, deep_features_size=256, backend='resnet18')
    # from thop import profile
    # input = torch.randn(1, 3, 224, 224)
    # macs, params = profile(model, inputs=(input, ))

    # from thop import clever_format
    # macs, params = clever_format([macs, params], "%.3f")
    # print(model)

    model = PSPNet(sizes=(1, 2, 3, 6), n_classes=12, psp_size=512, deep_features_size=256, backend='resnet18').cuda()
    size = [720, 960]
    input = torch.randn(1, 3, size[0], size[1]).cuda()

    time_list = []
    for i in range(20):
        start = time.time()
        output = model(input)
        time_list.append(time.time()-start)
    # print(time_list)
    print(size, np.array(time_list).mean())
    
