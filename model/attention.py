''' Define the sublayers in encoder/decoder layer '''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from localAttention import (similar_forward,
                            similar_backward,
                            weighting_forward,
                            weighting_backward_ori,
                            weighting_backward_weight)

class similarFunction(Function):
    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW):
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = (kH, kW)
        output = similar_forward(x_ori, x_loc, kH, kW)

        return output

    @staticmethod
    #@once_differentiable
    def backward(ctx, grad_outputs):
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = similar_backward(x_loc, grad_outputs, kH, kW, True)
        grad_loc = similar_backward(x_ori, grad_outputs, kH, kW, False)

        return grad_ori, grad_loc, None, None


class weightingFunction(Function):
    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW):
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = (kH, kW)
        output = weighting_forward(x_ori, x_weight, kH, kW)

        return output

    @staticmethod
    #@once_differentiable
    def backward(ctx, grad_outputs):
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = weighting_backward_ori(x_weight, grad_outputs, kH, kW)
        grad_weight = weighting_backward_weight(x_ori, grad_outputs, kH, kW)

        return grad_ori, grad_weight, None, None

f_similar = similarFunction.apply
f_weighting = weightingFunction.apply

def f_similar_cpu(query, key, kH, kW, key_uf):
    unfold = nn.Unfold(kernel_size=(kH, kW), padding=(kH//2, kW//2))
    # N, C, kH*kW, H, W
    key_uf = unfold(key).view(key.shape[0],key.shape[1],kH*kW, key.shape[-2], key.shape[-1])
    # N, 1, C, H, W
    query_uf = torch.unsqueeze(query,1)

    query_uf = query_uf.transpose(2,3).transpose(3,4).transpose(1,2).transpose(2,3)
    key_uf = key_uf.transpose(2,3).transpose(3,4).transpose(1,2).transpose(2,3)
    # N, H, W, 1, kH*kW
    import time
    start = time.time()
    weight = torch.matmul(
        query_uf, key_uf[0,0,0]
        )
    print(time.time() - start)
    import pdb; pdb.set_trace()
    # N, H, W, kH*kW
    return weight[:,:,:,0,:]

def f_weighting_cpu(value, weight, kH, kW):
    # import pdb; pdb.set_trace()
    unfold = nn.Unfold(kernel_size=(kH, kW), padding=(kH//2, kW//2))
    # N, C, kH*kW, H, W
    value_uf = unfold(value).view(value.shape[0],value.shape[1],kH*kW, value.shape[-2], value.shape[-1])
    # N, 1, kH*kW, H, W 
    weight_uf = torch.unsqueeze(weight,1).transpose(3,4).transpose(2,3)
    # N, C, kH*kW, H, W
    weight = value_uf * weight_uf
    # N, C, H, W
    return weight.sum(dim=2)

class MyAttention_dup(nn.Module):
    def __init__(self, feat_dim, kW, kH):
        super(MyAttention_dup, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        
        # self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)#, groups=feat_dim)
        # self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting

        self.softmax = nn.Softmax(dim=3)

        self.kW = kW
        self.kH = kH

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, hr_feat, lr_feat):
        # # - hr_feat [N, C, H, W]
        # # - lr_feat [N, C, h, w]

        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)
        # hr_feat = F.interpolate(hr_feat,(h,w),mode='bilinear', align_corners=True)

        # hr_value = self.hr_value_conv(hr_feat)
        hr_value = hr_feat
        hr_key = self.hr_key_conv(hr_feat)
        lr_query = self.lr_query_conv(lr_feat)

        weight = f_similar(lr_query, hr_key, self.kH, self.kW)
        # print(weight.shape)
        # import pdb; pdb.set_trace()
        # weight = F.softmax(weight, dim=3)
        weight = self.softmax(weight)
        attention_result = f_weighting(hr_value, weight, self.kH, self.kW)


        result = lr_feat + attention_result
        # result = attention_result

        return result
        
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params


class MyAttention(nn.Module):
    def __init__(self, feat_dim, kW, kH):
        super(MyAttention, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        
        self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim)
        # self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting

        self.softmax = nn.Softmax(dim=3)

        self.kW = kW
        self.kH = kH

        self.init_weight()

        # self.placeholder = torch.empty(1, feat_dim, kH*kW, 720, 960)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, hr_feat, lr_feat):
        # # - hr_feat [N, C, H, W]
        # # - lr_feat [N, C, h, w]

        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)
        # hr_feat = F.interpolate(hr_feat,(h,w),mode='bilinear', align_corners=True)

        hr_value = self.hr_value_conv(hr_feat)
        # hr_value = hr_feat
        hr_key = self.hr_key_conv(hr_feat)
        lr_query = self.lr_query_conv(lr_feat)

        weight = f_similar(lr_query, hr_key, self.kH, self.kW)
        # weight = f_similar_cpu(lr_query, hr_key, self.kH, self.kW, self.placeholder)
        # print(weight.shape)
        # weight = F.softmax(weight, dim=3)
        weight = self.softmax(weight)
        # np.save('./image_test_result/test-038-weight.npy',weight.cpu().detach().numpy())
        # import pdb; pdb.set_trace()

        attention_result = f_weighting(hr_value, weight, self.kH, self.kW)


        result = lr_feat + attention_result
        # result = attention_result

        return result
        
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params

class MyAttentionNoGroup(nn.Module):
    def __init__(self, feat_dim, kW, kH):
        super(MyAttentionNoGroup, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1) # computation can be reduced by setting groups=feat_dim
        
        self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)
        # self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting

        self.softmax = nn.Softmax(dim=3)

        self.kW = kW
        self.kH = kH

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, hr_feat, lr_feat):
        # # - hr_feat [N, C, H, W]
        # # - lr_feat [N, C, h, w]

        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)
        # hr_feat = F.interpolate(hr_feat,(h,w),mode='bilinear', align_corners=True)

        hr_value = self.hr_value_conv(hr_feat)
        # hr_value = hr_feat
        hr_key = self.hr_key_conv(hr_feat)
        lr_query = self.lr_query_conv(lr_feat)

        weight = f_similar(lr_query, hr_key, self.kH, self.kW)
        # print(weight.shape)
        # import pdb; pdb.set_trace()
        # weight = F.softmax(weight, dim=3)
        weight = self.softmax(weight)
        attention_result = f_weighting(hr_value, weight, self.kH, self.kW)


        result = lr_feat + attention_result
        # result = attention_result

        return result
        
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params

class MyAttentionLocalOnly(nn.Module):
    def __init__(self, feat_dim, kW, kH):
        super(MyAttentionLocalOnly, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        
        self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim)
        # self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting

        self.softmax = nn.Softmax(dim=3)

        self.kW = kW
        self.kH = kH

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, hr_feat, lr_feat):
        # # - hr_feat [N, C, H, W]
        # # - lr_feat [N, C, h, w]

        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)
        # hr_feat = F.interpolate(hr_feat,(h,w),mode='bilinear', align_corners=True)

        hr_value = self.hr_value_conv(hr_feat)
        # hr_value = hr_feat
        hr_key = self.hr_key_conv(hr_feat)
        lr_query = self.lr_query_conv(lr_feat)

        weight = f_similar(lr_query, hr_key, self.kH, self.kW)
        # print(weight.shape)
        # import pdb; pdb.set_trace()
        # weight = F.softmax(weight, dim=3)
        weight = self.softmax(weight)
        attention_result = f_weighting(hr_value, weight, self.kH, self.kW)


        # result = lr_feat + attention_result
        result = attention_result

        return result
        
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params


class MyAttentionV2(nn.Module):
    def __init__(self, feat_dim, kW, kH):
        super(MyAttentionV2, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=8) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=8) # computation can be reduced by setting groups=feat_dim
        
        # self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim)
        # self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting

        self.softmax = nn.Softmax(dim=3)

        self.kW = kW
        self.kH = kH

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, hr_feat, lr_feat):
        # # - hr_feat [N, C, H, W]
        # # - lr_feat [N, C, h, w]

        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)
        # hr_feat = F.interpolate(hr_feat,(h,w),mode='bilinear', align_corners=True)

        # hr_value = self.hr_value_conv(hr_feat)
        hr_value = hr_feat
        hr_key = self.hr_key_conv(hr_feat)
        lr_query = self.lr_query_conv(lr_feat)

        weight = f_similar(lr_query, hr_key, self.kH, self.kW)
        # print(weight.shape)
        # import pdb; pdb.set_trace()
        # weight = F.softmax(weight, dim=3)
        weight = self.softmax(weight)
        attention_result = f_weighting(hr_value, weight, self.kH, self.kW)


        result = lr_feat + attention_result
        # result = attention_result

        return result
        
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params


class MyAttentionV3(nn.Module):
    def __init__(self, feat_dim, kW, kH):
        super(MyAttentionV3, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=8) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=8) # computation can be reduced by setting groups=feat_dim
        
        self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=8)
        # self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting

        self.softmax = nn.Softmax(dim=3)

        self.kW = kW
        self.kH = kH

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, hr_feat, lr_feat):
        # # - hr_feat [N, C, H, W]
        # # - lr_feat [N, C, h, w]

        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)
        # hr_feat = F.interpolate(hr_feat,(h,w),mode='bilinear', align_corners=True)

        hr_value = self.hr_value_conv(hr_feat)
        # hr_value = hr_feat
        hr_key = self.hr_key_conv(hr_feat)
        lr_query = self.lr_query_conv(lr_feat)

        weight = f_similar(lr_query, hr_key, self.kH, self.kW)
        # print(weight.shape)
        # import pdb; pdb.set_trace()
        # weight = F.softmax(weight, dim=3)
        weight = self.softmax(weight)
        attention_result = f_weighting(hr_value, weight, self.kH, self.kW)


        result = lr_feat + attention_result
        # result = attention_result

        return result
        
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params

class MyAttentionV4(nn.Module):
    def __init__(self, feat_dim, kW, kH):
        super(MyAttentionV4, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        
        self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim)
        # self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting

        self.softmax = nn.Softmax(dim=3)

        self.kW = kW
        self.kH = kH

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, hr_feat, lr_feat):
        # # - hr_feat [N, C, H, W]
        # # - lr_feat [N, C, h, w]

        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        scale_factor = 4

        lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)
        # hr_feat = F.interpolate(hr_feat,(h,w),mode='bilinear', align_corners=True)

        hr_value = self.hr_value_conv(hr_feat)
        hr_value = F.interpolate(hr_value, (H//scale_factor, W//scale_factor), mode='bilinear', align_corners=True)
        # hr_value = hr_feat
        hr_key = self.hr_key_conv(hr_feat)
        hr_key = F.interpolate(hr_key, (H//scale_factor, W//scale_factor), mode='bilinear', align_corners=True)
        lr_query = self.lr_query_conv(lr_feat)

        attention_result = torch.zeros_like(lr_feat)

        for i in range(scale_factor):
            for j in range(scale_factor):
                weight = f_similar(lr_query[...,i::scale_factor,j::scale_factor], hr_key, self.kH, self.kW)
                # print(weight.shape)
                # import pdb; pdb.set_trace()
                # weight = F.softmax(weight, dim=3)
                weight = self.softmax(weight)
                attention_result[...,i::scale_factor,j::scale_factor] = f_weighting(hr_value, weight, self.kH, self.kW)


        result = lr_feat + attention_result
        # result = attention_result

        return result
        
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params

class MyAttentionV5(nn.Module):
    def __init__(self, feat_dim, kW, kH):
        super(MyAttentionV5, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        
        self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim)
        # self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting

        self.softmax = nn.Softmax(dim=3)

        self.kW = kW
        self.kH = kH

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, hr_feat, lr_feat):
        # # - hr_feat [N, C, H, W]
        # # - lr_feat [N, C, h, w]

        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        scale_factor = 2

        lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)
        # hr_feat = F.interpolate(hr_feat,(h,w),mode='bilinear', align_corners=True)

        hr_value = self.hr_value_conv(hr_feat)
        hr_value = F.interpolate(hr_value, (H//scale_factor, W//scale_factor), mode='bilinear', align_corners=True)
        # hr_value = hr_feat
        hr_key = self.hr_key_conv(hr_feat)
        hr_key = F.interpolate(hr_key, (H//scale_factor, W//scale_factor), mode='bilinear', align_corners=True)
        lr_query = self.lr_query_conv(lr_feat)

        attention_result = torch.zeros_like(lr_feat)

        for i in range(scale_factor):
            for j in range(scale_factor):
                weight = f_similar(lr_query[...,i::scale_factor,j::scale_factor], hr_key, self.kH, self.kW)
                # print(weight.shape)
                # import pdb; pdb.set_trace()
                # weight = F.softmax(weight, dim=3)
                weight = self.softmax(weight)
                attention_result[...,i::scale_factor,j::scale_factor] = f_weighting(hr_value, weight, self.kH, self.kW)


        result = lr_feat + attention_result
        # result = attention_result

        return result
        
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params

class MyAttentionV6(nn.Module):
    def __init__(self, feat_dim, kW, kH):
        super(MyAttentionV6, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        
        self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim)
        # self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting

        self.softmax = nn.Softmax(dim=3)

        self.kW = kW
        self.kH = kH

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, hr_feat, lr_feat):
        # # - hr_feat [N, C, H, W]
        # # - lr_feat [N, C, h, w]

        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        scale_factor = 1

        lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)
        # hr_feat = F.interpolate(hr_feat,(h,w),mode='bilinear', align_corners=True)

        hr_value = self.hr_value_conv(hr_feat)
        hr_value = F.interpolate(hr_value, (H//scale_factor, W//scale_factor), mode='bilinear', align_corners=True)
        # hr_value = hr_feat
        hr_key = self.hr_key_conv(hr_feat)
        hr_key = F.interpolate(hr_key, (H//scale_factor, W//scale_factor), mode='bilinear', align_corners=True)
        lr_query = self.lr_query_conv(lr_feat)

        attention_result = torch.zeros_like(lr_feat)

        for i in range(scale_factor):
            for j in range(scale_factor):
                weight = f_similar(lr_query[...,i::scale_factor,j::scale_factor], hr_key, self.kH, self.kW)
                # print(weight.shape)
                # import pdb; pdb.set_trace()
                # weight = F.softmax(weight, dim=3)
                weight = self.softmax(weight)
                attention_result[...,i::scale_factor,j::scale_factor] = f_weighting(hr_value, weight, self.kH, self.kW)


        result = lr_feat + attention_result
        # result = attention_result

        return result
        
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params




class MyAttentionLocalNew(nn.Module):
    def __init__(self, feat_dim, kW, kH):
        super(MyAttentionLocalNew, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        
        # self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)#, groups=feat_dim)
        # self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting

        self.kW = kW
        self.kH = kH

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    

    def forward(self, hr_feat, lr_feat):
        # # - hr_feat [N, C, H, W]
        # # - lr_feat [N, C, h, w]

        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        scaled_kH = int(self.kH*(H/h))
        scaled_kW = int(self.kW*(W/w))

        downsample_hr_feat = F.interpolate(hr_feat,(h,w), mode='bilinear', align_corners=True)

        # hr_value = self.hr_value_conv(hr_feat)
        hr_value = hr_feat

        downsample_hr_key = self.hr_key_conv(downsample_hr_feat)

        lr_query = self.lr_query_conv(lr_feat)

        # import pdb; pdb.set_trace()
        weight = f_similar(lr_query, downsample_hr_key, self.kH, self.kW).transpose(2,3).transpose(1,2).view(N,self.kH*self.kW,-1).transpose(1,2)
        weight = F.upsample(weight,size=(scaled_kH*scaled_kW)).transpose(1,2).view(N,scaled_kH*scaled_kW,h,w)
        weight = F.interpolate(weight,(H,W), mode='bilinear', align_corners=True).transpose(1,2).transpose(2,3)
        # print(weight.shape)
        # import pdb; pdb.set_trace()
        weight = F.softmax(weight, dim=3)

        attention_result = f_weighting(hr_value, weight, scaled_kH, scaled_kW)

        lr_feat = F.interpolate(lr_feat,(H,W), mode='bilinear', align_corners=True)
        result = lr_feat + attention_result

        return result
        
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params



class MyAttentionGlobal(nn.Module):
    def __init__(self, feat_dim, kScale=16):
        super(MyAttentionGlobal, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        
        self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim)#, groups=feat_dim)
        self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting
        self.kScale=kScale

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    
    def forward(self, hr_feat, lr_feat):
        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)

        hr_feat = self.hr_value_conv(hr_feat)
        hr_value = F.interpolate(hr_feat, (H//self.kScale, W//self.kScale), mode='bilinear', align_corners=True)
        hr_value = hr_value.transpose(1,2).transpose(2,3).view(N, -1, C).transpose(0,1)

        hr_key = self.hr_key_conv(hr_feat)
        hr_key = F.interpolate(hr_key, (H//self.kScale, W//self.kScale), mode='bilinear', align_corners=True)

        hr_key = hr_key.transpose(1,2).transpose(2,3).view(N, -1, C).transpose(0,1)

        lr_query = self.lr_query_conv(lr_feat)
        lr_query = lr_query.transpose(1,2).transpose(2,3).view(N, -1, C).transpose(0,1)

        # print(hr_key.shape, lr_query.shape)
        # exit(0)

        attention_result, attention_weight = self.attention(lr_query, hr_key, hr_value)
        # import pdb; pdb.set_trace()
        attention_result = attention_result.transpose(0,1).view(N,H,W,C).transpose(2,3).transpose(1,2)


        result = lr_feat + attention_result

        return result
    
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params

# class MyAttentionLocalOnly(nn.Module):
#     def __init__(self, feat_dim, kW, kH):
#         super(MyAttentionLocalOnly, self).__init__()

#         self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
#         self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim

#         self.kW = kW
#         self.kH = kH

#         self.init_weight()

#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)

#     def forward(self, hr_feat, lr_feat):
#         # # - hr_feat [N, C, H, W]
#         # # - lr_feat [N, C, h, w]

#         N, C, H, W = hr_feat.shape
#         N, C, h, w = lr_feat.shape 

#         lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)

#         hr_value = hr_feat
#         hr_key = self.hr_key_conv(hr_feat)
#         lr_query = self.lr_query_conv(lr_feat)

#         weight = f_similar(lr_query, hr_key, self.kH, self.kW)
#         weight = F.softmax(weight, -1)
#         attention_result = f_weighting(hr_value, weight, self.kH, self.kW)


#         result = attention_result

#         return result
        
#     def get_params(self):
#         wd_params, nowd_params = [], []
#         for name, module in self.named_modules():
#             if isinstance(module, (nn.Linear, nn.Conv2d)):
#                 # print("1", name, module)
#                 wd_params.append(module.weight)
#                 if not module.bias is None:
#                     # print("2", name, module)
#                     nowd_params.append(module.bias)
#             elif isinstance(module, nn.MultiheadAttention):
#                 # print("3", name, module)
#                 # nowd_params += list(module.parameters())
#                 ## The params in attention modules are nn.Linear
#                 continue
#         return wd_params, nowd_params


class MyAttentionGlobalOnly(nn.Module):
    def __init__(self, feat_dim):
        super(MyAttentionGlobalOnly, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        
        # self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)#, groups=feat_dim)
        self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    
    def forward(self, hr_feat, lr_feat):
        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)

        hr_value = F.interpolate(hr_feat, (H//16, W//16), mode='bilinear', align_corners=True)
        hr_value = hr_value.transpose(1,2).transpose(2,3).view(N, -1, C).transpose(0,1)

        hr_key = self.hr_key_conv(hr_feat)
        hr_key = F.interpolate(hr_key, (H//16, W//16), mode='bilinear', align_corners=True)

        hr_key = hr_key.transpose(1,2).transpose(2,3).view(N, -1, C).transpose(0,1)

        lr_query = self.lr_query_conv(lr_feat)
        lr_query = lr_query.transpose(1,2).transpose(2,3).view(N, -1, C).transpose(0,1)

        # print(hr_key.shape, lr_query.shape)
        # exit(0)

        attention_result, attention_weight = self.attention(lr_query, hr_key, hr_value)
        # import pdb; pdb.set_trace()
        attention_result = attention_result.transpose(0,1).view(N,H,W,C).transpose(2,3).transpose(1,2)


        result = attention_result

        return result
    
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params


class MyAttentionGlobalNoGroup(nn.Module):
    def __init__(self, feat_dim):
        super(MyAttentionGlobalNoGroup, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim//4, kernel_size=3, padding=1) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim//4, kernel_size=3, padding=1) # computation can be reduced by setting groups=feat_dim
        
        self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim//4, kernel_size=3, padding=1)#, groups=feat_dim)
        self.value_trans_conv = nn.Conv2d(feat_dim//4, feat_dim, kernel_size=1)

        self.attention = nn.MultiheadAttention(embed_dim=feat_dim//4, num_heads=1)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    
    def forward(self, hr_feat, lr_feat):
        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)

        # import pdb; pdb.set_trace()

        hr_value = self.hr_value_conv(hr_feat)
        hr_value = F.interpolate(hr_value, (H//16, W//16), mode='bilinear', align_corners=True)
        hr_value = hr_value.transpose(1,2).transpose(2,3).view(N, -1, C//4).transpose(0,1)

        hr_key = self.hr_key_conv(hr_feat)
        hr_key = F.interpolate(hr_key, (H//16, W//16), mode='bilinear', align_corners=True)

        hr_key = hr_key.transpose(1,2).transpose(2,3).view(N, -1, C//4).transpose(0,1)

        lr_query = self.lr_query_conv(lr_feat)
        lr_query = lr_query.transpose(1,2).transpose(2,3).view(N, -1, C//4).transpose(0,1)

        # import pdb; pdb.set_trace()
        attention_result, attention_weight = self.attention(lr_query, hr_key, hr_value)
        attention_result = attention_result.transpose(0,1).view(N,H,W,C//4).transpose(2,3).transpose(1,2)


        result = lr_feat + self.value_trans_conv(attention_result)

        return result
    
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params


class MyAttentionSelf(nn.Module):
    def __init__(self, feat_dim):
        super(MyAttentionSelf, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        
        # self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)#, groups=feat_dim)
        self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    
    def forward(self, hr_feat, lr_feat):
        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)

        lr_value = F.interpolate(lr_feat, (H//16, W//16), mode='bilinear', align_corners=True)
        lr_value = lr_value.transpose(1,2).transpose(2,3).view(N, -1, C).transpose(0,1)

        lr_key = self.hr_key_conv(lr_feat)
        lr_key = F.interpolate(lr_key, (H//16, W//16), mode='bilinear', align_corners=True)

        lr_key = lr_key.transpose(1,2).transpose(2,3).view(N, -1, C).transpose(0,1)

        lr_query = self.lr_query_conv(lr_feat)
        lr_query = lr_query.transpose(1,2).transpose(2,3).view(N, -1, C).transpose(0,1)

        # print(hr_key.shape, lr_query.shape)
        # exit(0)

        attention_result, attention_weight = self.attention(lr_query, lr_key, lr_value)
        # import pdb; pdb.set_trace()
        attention_result = attention_result.transpose(0,1).view(N,H,W,C).transpose(2,3).transpose(1,2)


        result = lr_feat + attention_result

        return result
    
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params


class MyAttentionNo(nn.Module):
    def __init__(self, feat_dim):
        super(MyAttentionNo, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        
        # self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)#, groups=feat_dim)
        self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    
    def forward(self, hr_feat, lr_feat):
        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        # import pdb; pdb.set_trace()
        # lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)


        return lr_feat
    
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params


class MyAttentionUpsample(nn.Module):
    def __init__(self, feat_dim):
        super(MyAttentionUpsample, self).__init__()

        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        self.hr_key_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim) # computation can be reduced by setting groups=feat_dim
        
        # self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)#, groups=feat_dim)
        self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=1)
        # self.f_similar = f_similar
        # self.f_weighting = f_weighting

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    
    def forward(self, hr_feat, lr_feat):
        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        # import pdb; pdb.set_trace()
        lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)


        return lr_feat
    
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # print("1", name, module)
                wd_params.append(module.weight)
                if not module.bias is None:
                    # print("2", name, module)
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # print("3", name, module)
                # nowd_params += list(module.parameters())
                ## The params in attention modules are nn.Linear
                continue
        return wd_params, nowd_params


class ConvFusion(nn.Module):
    def __init__(self, feat_dim):
        super(ConvFusion, self).__init__()

        # self.lr_refine_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1) # computation can be reduced by setting groups=feat_dim
        self.fusion_conv = nn.Conv2d(2 * feat_dim, feat_dim, kernel_size=3, padding=1) # computation can be reduced by setting groups=feat_dim

        self.init_weight()
    
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    
    def forward(self, hr_feat, lr_feat):
        N, C, H, W = hr_feat.shape
        N, C, h, w = lr_feat.shape 

        # import pdb; pdb.set_trace()
        upsample_lr_feat = F.interpolate(lr_feat,(H,W),mode='bilinear', align_corners=True)

        # refined_lr_feat = self.lr_refine_conv(upsample_lr_feat)

        result = self.fusion_conv(torch.cat([upsample_lr_feat, hr_feat], dim=1))


        return result


if __name__ == '__main__':
    net = MyAttention(feat_dim=128, kW=5, kH=5).cuda()
    input = torch.rand((1,128,90,120)).cuda()
    lr_input = torch.rand((1,128,68,90)).cuda()
    print(net(input, lr_input).shape)