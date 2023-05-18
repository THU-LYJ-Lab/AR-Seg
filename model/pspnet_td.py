import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_td import resnet18,resnet34,resnet50
import random
import logging
import pdb
import os
# from .transformer import Encoding, Attention
from model.attention import *

up_kwargs = {'mode': 'bilinear', 'align_corners': True}
logger = logging.getLogger("ptsemseg")

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=1, stride=1, padding=0, norm_layer=None, bias=True, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = bias)
        self.norm_layer = norm_layer
        if norm_layer is not None:
            self.bn = norm_layer(out_chan, activation='leaky_relu')
  
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        
        if self.norm_layer is not None:
            x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class Encoding(nn.Module):
    def __init__(self, d_model, d_k, d_v, norm_layer=None, dropout=0.1):
        super(Encoding, self).__init__()

        self.norm_layer = norm_layer

        self.d_k = d_k
        self.d_v = d_v

        self.w_vs = nn.Sequential(ConvBNReLU(d_model, d_v, ks=1, stride=1, padding=0, norm_layer=None))

        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=3, padding=0)

    def forward(self, fea, pre=None, start=None):

        n_,c_,h_,w_ = fea.size()

        d_k, d_v = self.d_k, self.d_v

        if pre:
            raise NotImplementedError

        else:
            v = self.w_vs(fea).view(n_, d_v, h_, w_)
            return v

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.norm_layer)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class PSPNet(nn.Module):
    """
    """
    def __init__(self,
                 input_channel=3,
                 nclass=21,
                 norm_layer=nn.BatchNorm2d,
                 backend='resnet101',
                 dilated=True,
                 aux=True,
                 multi_grid=True,
                 loss_fn=None,
                 path_num=1,
                 mdl_path = None,
                 teacher = None
                 ):
        super(PSPNet, self).__init__()

        self.psp_path = mdl_path
        self.loss_fn = loss_fn
        self.path_num = path_num
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.nclass = nclass

        # copying modules from pretrained models
        self.backend = backend
        assert(backend == 'resnet50' or backend == 'resnet34' or backend == 'resnet18')
        assert(path_num == 1)

        if backend == 'resnet18':
            ResNet_ = resnet18
            deep_base = False
            self.expansion = 1
        elif backend == 'resnet34':
            ResNet_ = resnet34
            deep_base = False
            self.expansion = 1
        elif backend == 'resnet50':
            ResNet_ = resnet50
            deep_base = True
            self.expansion = 4
        else:
            raise RuntimeError("Four branch model only support ResNet18 amd ResNet34")

        self.pretrained = ResNet_(pretrained=True, dilated=dilated, multi_grid=multi_grid,
                                           deep_base=deep_base, norm_layer=norm_layer)
        # bilinear upsample options

        # self.psp =  PyramidPooling(512*self.expansion, norm_layer, self._up_kwargs, path_num=2, pid=0)

        # self.enc = Encoding(512*self.expansion,64,512*self.expansion,norm_layer)



        self.head = FCNHead_out(512*self.expansion*1, nclass, norm_layer, chn_down=4)

        if aux:
            self.auxlayer = FCNHead(256*self.expansion, nclass, norm_layer)
            
        #self.pretrained_init_2p()
        self.pretrained_init()
        self.get_params()


    
    def forward(self, f_img):
        '''
        :param f_img: [t-3, t-2, t-1, t]
        '''
        
        _, _, h, w = f_img.size()

        c3_1,c4_1 = self.pretrained(f_img)

        # z1 = self.psp(c4_1)

        # v1 = self.enc(z1, pre=False)



        out1 = self.head(c4_1)

        outputs1 = F.interpolate(out1, (h, w), **self._up_kwargs)
        
        if self.training:
            #############Knowledge-distillation###########
        
            auxout1 = self.auxlayer(c3_1)        
            auxout1 = F.interpolate(auxout1, (h, w), **self._up_kwargs)
            return outputs1, auxout1
        else:
            return outputs1, 0
        


    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (Encoding, PyramidPooling, FCNHead, Layer_Norm)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

    def pretrained_init(self):
        if self.psp_path is not None:
            if os.path.isfile(self.psp_path):
                logger.info("Initializaing sub networks with pretrained '{}'".format(self.psp_path))
                print("Initializaing sub networks with pretrained '{}'".format(self.psp_path))
                model_state = torch.load(self.psp_path)
                self.load_state_dict(model_state, strict=True)
                # backend_state, psp_state, head_state1, head_state2, _, _, auxlayer_state = split_psp_dict(model_state,2)
                # self.pretrained.load_state_dict(backend_state, strict=True)
                # self.psp.load_state_dict(psp_state, strict=True)
                # self.head.load_state_dict(head_state1, strict=False)
                # self.auxlayer.load_state_dict(auxlayer_state, strict=True)
            else:
                logger.info("No pretrained found at '{}'".format(self.psp_path))


class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.norm_layer = norm_layer
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   norm_layer(out_channels),
                                   nn.ReLU(True))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs
        self.init_weight()

    def forward(self, x):
        n, c, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)

        # x = x[:, self.pid*c//self.path_num:(self.pid+1)*c//self.path_num]
        # feat1 = feat1[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]
        # feat2 = feat2[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]
        # feat3 = feat3[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]
        # feat4 = feat4[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]

        return torch.cat((x, feat1, feat2, feat3, feat4), 1)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Sequential):
                for lz in ly:
                    if isinstance(lz, nn.Conv2d):
                        nn.init.kaiming_normal_(lz.weight, a=1)
                        if not lz.bias is None: nn.init.constant_(lz.bias, 0)
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.norm_layer)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class FCNHead_out(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs={}, chn_down=4):
        super(FCNHead_out, self).__init__()

        inter_channels = in_channels // chn_down

        self._up_kwargs = up_kwargs
        self.norm_layer = norm_layer
        self.conv5 = nn.Sequential(PyramidPooling(in_channels=in_channels, norm_layer=norm_layer, up_kwargs=up_kwargs),
                                   nn.Conv2d(in_channels*2, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))
        self.init_weight()

    def forward(self, x):
        return self.conv5(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Sequential):
                for lz in ly:
                    if isinstance(lz, nn.Conv2d):
                        nn.init.kaiming_normal_(lz.weight, a=1)
                        if not lz.bias is None: nn.init.constant_(lz.bias, 0)
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.norm_layer)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs={}, chn_down=4):
        super(FCNHead, self).__init__()

        inter_channels = in_channels // chn_down

        self._up_kwargs = up_kwargs
        self.norm_layer = norm_layer
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))
        self.init_weight()

    def forward(self, x):
        return self.conv5(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Sequential):
                for lz in ly:
                    if isinstance(lz, nn.Conv2d):
                        nn.init.kaiming_normal_(lz.weight, a=1)
                        if not lz.bias is None: nn.init.constant_(lz.bias, 0)
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.norm_layer)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class Layer_Norm(nn.Module):
    def __init__(self, shape):
        super(Layer_Norm, self).__init__()
        self.ln = nn.LayerNorm(shape)

    def forward(self, x):
        return self.ln(x)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (nn.LayerNorm)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


### uils functions
from collections import OrderedDict

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def split_psp_dict(state_dict, path_num =None):
    """Split a PSPNet model into different part
       :param state_dict is the loaded DataParallel model_state
    """
    model_state = convert_state_dict(state_dict)

    backbone_state = OrderedDict()
    psp_state = OrderedDict()
    head_state1 = OrderedDict()
    head_state2 = OrderedDict()
    head_state3 = OrderedDict()
    head_state4 = OrderedDict()
    auxlayer_state = OrderedDict()

    for k, v in model_state.items():
        s_k = k.split('.')
        if s_k[0] == 'pretrained':
            backbone_state['.'.join(s_k[1:])] = v

        if s_k[0] == 'head':
            pk = s_k[1:]
            if pk[1] == '0':
                psp_state['.'.join(pk[2:])] = v
            else:
                pk[1] = str(int(pk[1]) - 1)
                if pk[1] == '0':  #Shift channel params
                    o_c, i_c, h_, w_ = v.size()
                    shifted_param_l = []
                    step1 = i_c//2//path_num
                    step2 = i_c//8//path_num
                    for id in range(path_num):
                        idx1 = range(id*step1,id*step1+step1)
                        idx2 = range(i_c*4//8+id*step2,i_c*4//8+id*step2+step2)
                        idx3 = range(i_c*5//8+id*step2,i_c*5//8+id*step2+step2)
                        idx4 = range(i_c*6//8+id*step2,i_c*6//8+id*step2+step2)
                        idx5 = range(i_c*7//8+id*step2,i_c*7//8+id*step2+step2)
                        shifted_param_l.append(v[:,idx1,:,:])
                        shifted_param_l.append(v[:,idx2,:,:])
                        shifted_param_l.append(v[:,idx3,:,:])
                        shifted_param_l.append(v[:,idx4,:,:])
                        shifted_param_l.append(v[:,idx5,:,:])
                    v1 = torch.cat(shifted_param_l[:5], dim=1)
                    v2 = torch.cat(shifted_param_l[5:10], dim=1)
                    if path_num==2:
                        v3 = torch.cat(shifted_param_l[:5], dim=1)
                        v4 = torch.cat(shifted_param_l[5:10], dim=1)
                    elif path_num==4:
                        v3 = torch.cat(shifted_param_l[10:15], dim=1)
                        v4 = torch.cat(shifted_param_l[15:20], dim=1)
                    else:
                        raise RuntimeError("Only support 2 or 4 path")

                    head_state1['.'.join(pk)] =v1
                    head_state2['.'.join(pk)] =v2
                    head_state3['.'.join(pk)] =v3
                    head_state4['.'.join(pk)] =v4
                else:
                    head_state1['.'.join(pk)] = v
                    head_state2['.'.join(pk)] = v
                    head_state3['.'.join(pk)] = v
                    head_state4['.'.join(pk)] = v

        if s_k[0] == 'auxlayer':
            auxlayer_state['.'.join(s_k[1:])] = v

    return backbone_state, psp_state, head_state1, head_state2, head_state3, head_state4, auxlayer_state

