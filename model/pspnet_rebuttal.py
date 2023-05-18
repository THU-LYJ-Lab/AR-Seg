import torch
torch.manual_seed(233)

from torch import nn
from torch.nn import functional as F

import model.extractors as extractors

from model.attention import *


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

        self.fusion_conv = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

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
            # f = self.fusion_conv(torch.cat(ref_p, dim=1))
            f = (ref_p[0] + ref_p[1]) /2 
            p = self.psp(f)
            p = self.drop_1(p)

            p = self.up_1(p)
            p = self.drop_2(p)

            p = self.up_2(p)
            p = self.drop_2(p)

            p = self.up_3(p)
            p = self.drop_2(p)

            # import pdb; pdb.set_trace()
            out = self.final_conv(p)

            out = self.final_logsoftmax(out)

            return out 


    




if __name__ == "__main__":
    model = PSPNet(sizes=(1, 2, 3, 6), n_classes=12, psp_size=512, deep_features_size=256, backend='resnet18')
    from thop import profile
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input, ))

    from thop import clever_format
    macs, params = clever_format([macs, params], "%.3f")
    print(model)
