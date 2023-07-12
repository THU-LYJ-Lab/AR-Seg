import os
import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms

from PIL import Image

from dataset.transform import *

import json

import numpy as np
np.random.seed(233)

import os.path as osp

def pil_loader(data_path, label_path):
    """Loads a sample and label image given their path as PIL images.
    Keyword arguments:
    - data_path (``string``): The filepath to the image.
    - label_path (``string``): The filepath to the ground-truth image.
    Returns the image and the label as PIL images.
    """
    data = Image.open(data_path)
    label = Image.open(label_path)

    return data, label


## The dataset for training HR model on cityscapes
class CityScapes(Dataset):
    def __init__(self, rootpth, model_type, cropsize=(640, 480), mode='train', 
    randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255

        ## Load dataset info file
        with open('./dataset/cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'leftImg8bit', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            if '_gtFine_leftImg8bit' in im_names[0]:
                names = [el.replace('_gtFine_leftImg8bit.png', '') for el in im_names]
            else:
                names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        print('self.len', self.mode, self.len)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        ## pre-processing

        ## We use pre-trained BiseNet as the HR model, which is trained with another set of mean and std values
        ## BiseNet: https://github.com/CoinCheung/BiSeNet/
        if model_type == 'bisenet':
            self.mean =(0.3257, 0.3690, 0.3223)
            self.std = (0.2112, 0.2148, 0.2115)
        elif model_type == 'pspnet':
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.Normalize((0.3257, 0.3690, 0.3223), (0.2112, 0.2148, 0.2115)),
            transforms.Normalize(self.mean, self.std)
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.4,
                contrast = 0.4,
                saturation = 0.4),
            HorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomScale(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            RandomCrop(cropsize)
            ])


    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        # print(fn)
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)
        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)
        # print(label.shape, label.max(), label)
        label = self.convert_labels(label)
        # print(label.shape, label)
        # import pdb; pdb.set_trace()
        return img, label, self.gen_label_existence(label)


    def __len__(self):
        return self.len


    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label
    
    def gen_label_existence(self, label):
        existence_list = torch.zeros((19,))
        label_unique = np.unique(label)

        for label in label_unique:
            if label == self.ignore_lb:
                continue
            existence_list[label] = 1
        
        return existence_list

class CityScapesWithFlow(Dataset):
    def __init__(self, rootpth, model_type, cropsize=(640, 480), mode='train', 
    randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), 
    ref_gap=12, flow_path = '/mnt/nvme1n1/hyb/data/cityscapes-sequence/5M-GOP12/MVmap_GOP12_dist_11',
    ref_path = './data/cityscapes/leftImg8bit_sequence/',
    *args, **kwargs):
        super(CityScapesWithFlow, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255

        with open('./dataset/cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'leftImg8bit', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            if '_gtFine_leftImg8bit' in im_names[0]:
                names = [el.replace('_gtFine_leftImg8bit.png', '') for el in im_names]
            else:
                names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        print('self.len', self.mode, self.len)
        # print(imgnames, gtnames)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        if model_type == 'bisenet':
            self.mean =(0.3257, 0.3690, 0.3223)
            self.std = (0.2112, 0.2148, 0.2115)
        elif model_type == 'pspnet':
            self.mean = (0.485, 0.456, 0.406)
            # self.mean = (0.406, 0.456, 0.485)
            self.std = (0.229, 0.224, 0.225)

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Normalize(self.mean, self.std),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.4,
                contrast = 0.4,
                saturation = 0.4),
            HorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomScale(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            RandomCrop(cropsize)
            ])
        
        self.trans_train_color = pairColorJitter(
            brightness = 0.5,
            contrast = 0.5,
            saturation = 0.5
        )

        self.trans_train_homo = pairCompose([
            pairOFHorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            pairOFRandomScaleV2(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            pairOFRandomCrop(cropsize)
        ])
        
        self.ref_gap = ref_gap
        # self.ref_path = os.path.join(rootpth, 'leftImg8bit_sequence')
        self.ref_path = ref_path

        self.flow_pth = flow_path


    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)

        fn_parse = fn.split('_')
        fn_idx = int(fn_parse[-1])
        ref_idx = fn_idx - (self.ref_gap - 1)
        ref_fn_parse = fn_parse[:-1] + ["%06d"%ref_idx, "leftImg8bit.png"]
        ref_fn = '_'.join(ref_fn_parse)
        fn_scene = fn_parse[0]

        ref_impth = os.path.join(self.ref_path,self.mode,fn_scene,ref_fn)
        ref_img = Image.open(ref_impth)

        ref_label = label

        ref_flow_pth = os.path.join(self.flow_pth, self.mode, fn_scene, fn+'_gtFine_leftImg8bit.bin')

        f = open(ref_flow_pth)
        flow = np.fromfile(f,np.dtype(np.short)).reshape(1024,2048,2)/4
        f.close()
        flow_map = flow

        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            ref_im_lb = dict(im = ref_img, lb = ref_label)
            im_lb, ref_im_lb = self.trans_train_color(im_lb, ref_im_lb)
            
            ref_im_lb = dict(im = ref_im_lb['im'], lb = flow)
            im_lb, ref_im_lb = self.trans_train_homo(im_lb, ref_im_lb)

            img, label = im_lb['im'], im_lb['lb']
            ref_img, flow_map = ref_im_lb['im'], ref_im_lb['lb']

        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)
        # print(label.shape, label.max(), label)
        label = self.convert_labels(label)
        # print(label.shape, label)
        # import pdb; pdb.set_trace()

        ref_img = self.to_tensor(ref_img)
        return img, label, self.gen_label_existence(label), ref_img, flow_map


    def __len__(self):
        return self.len


    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label
    
    def gen_label_existence(self, label):
        existence_list = torch.zeros((19,))
        label_unique = np.unique(label)

        for label in label_unique:
            if label == self.ignore_lb:
                continue
            existence_list[label] = 1
        
        return existence_list


if __name__ == "__main__":
    dataset = CityScapesWithFlow("./data/cityscapes/",mode="train", ref_gap=12)
    for i, data in enumerate(dataset):
        print(i)
    import pdb; pdb.set_trace()