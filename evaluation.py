#!/usr/bin/python
# -*- encoding: utf-8 -*-

# from cityscapes import CityScapes
from dataset.camvid import CamVid, CamVidWithFlow
from dataset.cityscapes import CityScapes, CityScapesWithFlow

from model.pspnet import PSPNet, PSPNetWithFuse
from model.pspnet_semseg import PSPNetWithFuse as PSPNetWithFuse_Cityscapes 
from model.bisenet import BiSeNetV1, BiSeNetV1WithFuse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable

import os
import logging
import numpy as np
from tqdm import tqdm

models = {
    'camvid-psp18': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=12, psp_size=512, deep_features_size=256, backend='resnet18'),
    'camvid-bise18': lambda: BiSeNetV1(n_classes=12, backend='resnet18'), 
    'cityscapes-psp18': lambda: PSPNetWithFuse_Cityscapes(bins=(1, 2, 3, 6), classes=19, feat_dim=512, layers=18),
    'cityscapes-bise18': lambda: BiSeNetV1(n_classes=19, backend='resnet18')
}

models_fuse = {
    'camvid-psp18': lambda: PSPNetWithFuse(sizes=(1, 2, 3, 6), n_classes=12, psp_size=512, deep_features_size=256, backend='resnet18', atten_k=7),
    'camvid-bise18': lambda: BiSeNetV1WithFuse(n_classes=12, backend='resnet18'),
    'cityscapes-psp18': lambda: PSPNetWithFuse_Cityscapes(bins=(1, 2, 3, 6), classes=19, feat_dim=512, layers=18),
    'cityscapes-bise18': lambda: BiSeNetV1WithFuse(n_classes=19, backend='resnet18')
}

def build_network(snapshot, backend):
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        if backend == 'cityscapes-psp18':
            net.load_state_dict(torch.load(snapshot), strict=False)
        else:
            net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot loaded from {}".format(snapshot))
    net = net.cuda()
    return net

def build_network_fuse(snapshot, backend):
    backend = backend.lower()
    net = models_fuse[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot loaded from {}".format(snapshot))
    net = net.cuda()
    return net

def warpFeature(feature, flow):
    B, C, H, W = feature.shape

    flow = flow.permute(0,3,1,2)
    # assert B==1

    xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
    yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
    xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    grid = torch.cat((xx ,yy) ,1).float()

    # import pdb; pdb.set_trace()

    if feature.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flow

    # scale grid to [-1,1]
    vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
    vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0

    vgrid = vgrid.permute(0 ,2 ,3 ,1).float()
    # import pdb; pdb.set_trace()
    output = F.grid_sample(feature, vgrid)

    return output


class EvalConstRes(object):

    def __init__(self, scale=0.5, ignore_label=255):
        self.ignore_label = ignore_label
        self.scale = scale

    def __call__(self, net, dl, n_classes):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))

        sample_iou_list = []
        for i, (imgs, label, _) in diter:

            N, H, W = label.shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]

            imgs = imgs.cuda()

            N, C, H, W = imgs.size()
            new_hw = [int(H*self.scale), int(W*self.scale)]

            imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)
            
            logits = net(imgs)[0]

            # print(logits.shape)
  
            logits = F.interpolate(logits, size=size,
                    mode='bilinear', align_corners=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes).float()
            

        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        # import pdb; pdb.set_trace()

        # plt.plot(np.array(sample_iou_list))
        # plt.savefig('decoded.png')
        # np.save("decoded-mIoU.npy", np.array(sample_iou_list))

        return miou.item()



class EvalAlterRes(object):

    def __init__(self, scale=0.5, ignore_label=255):
        self.ignore_label = ignore_label
        self.scale = scale

    def __call__(self, highres_net, net, dl, n_classes):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label, _, ref_imgs, flow) in diter:
            ## Prepare the data
            N, H, W = label.shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]

            imgs = imgs.cuda()

            flow = flow.cuda()

            ## Calculate the reference feature
            ref_out = highres_net(ref_imgs.cuda())
            highres_ref_p = ref_out[-1]

            ## resize the motion vectors to fit the feature shape
            flow = flow.transpose(2,3).transpose(1,2)
            flow = flow * highres_ref_p.shape[-2] / flow.shape[-2]
            flow = F.interpolate(flow, [highres_ref_p.shape[-2], highres_ref_p.shape[-1]], mode='bilinear', align_corners=True)
            flow = flow.transpose(1,2).transpose(2,3)
            
            ## warp the reference feature
            highres_ref_p = warpFeature(highres_ref_p, flow)

            ## calculate the low resolution feature with CReFF
            N, C, H, W = imgs.size()
            new_hw = [int(H*self.scale), int(W*self.scale)]
            imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)

            phase1_out = net.module.forward_phase1(imgs)
            out_p = phase1_out[-1]
            
            out, _ = net.module.forward_phase2(out_p, highres_ref_p)

            

            logits = out

            # print(logits.shape)
  
            logits = F.interpolate(logits, size=size,
                    mode='bilinear', align_corners=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes).float()
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        # import pdb; pdb.set_trace()
        return miou.item()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluation.')
    parser.add_argument('--mode', type=int, nargs='+', default=[1,1,1], help='Eval or not with HR / LR / AR')
    parser.add_argument('--result_dir', type=str, default='./evaluation-result',help='directory to store the evaluation output txt')
    parser.add_argument('--ckpt_root', type=str, default='./checkpoints', help='root directory to load the checkpoints')

    parser.add_argument('--data_root', type=str, default='./data', help='directory to load the dataset')
    
    parser.add_argument('--dataset', type=str, default='camvid', help='dataset name')
    parser.add_argument('--backbone', type=str, default='psp18', help='backbone name')
    
    parser.add_argument('--GOP', type=int, default=12, help='GOP size')
    parser.add_argument('--test_scale', type=float, default=0.5, help='low resolution scale')
    
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    ## GOP size, in order to set loop number and find the data folder
    GOP = args.GOP

    ## bitrate, in order to find the data folder
    if args.dataset == 'camvid':
        bitrate='3M'
        n_class = 12
    elif args.dataset == 'cityscapes':
        bitrate='5M'
        n_class = 19
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))

    ## low resolution scale list
    # scale_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    scale_list = [args.test_scale]

    mode_list = {
        'HR': args.mode[0],
        'LR': args.mode[1],
        'AR': args.mode[2]
    }

    ## 1.0x HR model checkpoint 
    highres_dir = os.path.join(args.ckpt_root, f'{args.dataset}-{args.backbone}', 'HR')
    highres_snapshot = os.path.join(highres_dir, os.listdir(highres_dir)[0])

    if mode_list['HR']:
        ## Test 1.0x model
        backend = f'{args.dataset}-{args.backbone}'
        highres_net = build_network(highres_snapshot, backend)
        highres_net.eval()

        mIoU_list = []

        for ref_gap in range(1, GOP+1):
            data_path =os.path.join(args.data_root, f'{args.dataset}-sequence', '%s-GOP%d/decoded_GOP%d_dist_%d/'%(bitrate, GOP,GOP,ref_gap-1))
            
            if args.dataset == 'camvid':
                test_ds = CamVid(data_path, mode='test')
            elif args.dataset == 'cityscapes':
                test_ds = CityScapes(data_path, model_type=f'{args.backbone[:-2]}net', mode='val')

            test_loader = DataLoader(test_ds,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 8,
                            pin_memory = False,
                            drop_last = True)

            evaluator = EvalConstRes(scale=1.0, ignore_label=255)

            mIOU = evaluator(net=highres_net, dl=test_loader, n_classes=n_class)

            print(ref_gap, 'HR', '1.0x', mIOU)
            mIoU_list.append(mIOU)

        print(mIoU_list)
        print(np.array(mIoU_list).mean())
        mIoU_list.append(np.array(mIoU_list).mean())
        np.savetxt(os.path.join(args.result_dir,
                        f"{args.dataset}-{args.backbone}-1.0x-resolution-exp-GOP{GOP}-{bitrate}-evaluation.txt"
                        ), 
                   np.array(mIoU_list)
        )


    for test_scale in scale_list:
        if mode_list['AR']:
            mIoU_list = []

            # get the AR segmentation model
            model_dir = os.path.join(args.ckpt_root, f'{args.dataset}-{args.backbone}', 'AR')
            # model_list = os.listdir("./exp/pspnet18-camvid/rebuttal/local_7x7_1")
            model_list = [x for x in os.listdir(model_dir) if x.split('_')[2]==str(test_scale)]

            lowres_snapshot = os.path.join(model_dir, model_list[0])

            for ref_gap in range(1, GOP+1):
                ## data path to load the decoded annotated frames
                data_path =os.path.join(args.data_root, f'{args.dataset}-sequence', 
                                '%s-GOP%d/decoded_GOP%d_dist_%d/'%(bitrate,GOP,GOP,ref_gap-1)
                            )   

                if ref_gap > 1:
                    ## create test dataset, loading the motion vectors and decoded reference frames
                    flow_path = os.path.join(args.data_root, f'{args.dataset}-sequence','%s-GOP%d/MVmap_GOP%d_dist_%d/'%(bitrate,GOP, GOP, ref_gap-1) )
                    ref_path = os.path.join(args.data_root, f'{args.dataset}-sequence', '%s-GOP%d/frames/'%(bitrate,GOP))
                    if args.dataset == 'camvid':  
                        test_ds = CamVidWithFlow(data_path, mode='test',load_pair=True, ref_gap=ref_gap, flow_path=flow_path,ref_path=ref_path)
                    elif args.dataset == 'cityscapes':
                        test_ds = CityScapesWithFlow(data_path, model_type=f'{args.backbone[:-2]}net',mode='val', ref_gap=ref_gap, flow_path=flow_path, ref_path=ref_path)
                    
                ## If ref_gap == 0, we just test with the high-resolution model, which utilize another type of test dataset.
                else:
                    if args.dataset == 'camvid':
                        test_ds = CamVid(data_path, mode='test')
                    elif args.dataset == 'cityscapes':
                        test_ds = CityScapes(data_path, model_type=f'{args.backbone[:-2]}net', mode='val')

                test_loader = DataLoader(test_ds,
                                batch_size = 1,
                                shuffle = False,
                                num_workers = 4,
                                pin_memory = False,
                                drop_last = True)
                
                ## build the model of high and low resolution branch
                backend = f'{args.dataset}-{args.backbone}'
                highres_net = build_network(highres_snapshot, backend)
                highres_net.eval()

                lowres_net = build_network_fuse(lowres_snapshot, backend)
                lowres_net.eval()

                ## set the low resolution scale: 0.5
                ## For debugging
                # evaluator = MscEvalPair(scale=test_scale, ignore_label=255)

                if ref_gap > 1:
                    ## create the evaluator with motion vector
                    evaluator = EvalAlterRes(scale=test_scale, ignore_label=255)
                else:
                    ## create the evaluator that only utilizes high-resolution model
                    evaluator = EvalConstRes(scale=1.0, ignore_label=255)
                

                # mIOU = evaluator(highres_net=highres_net, net=lowres_net, dl=test_loader, n_classes=12,
                #                     start_point_list=test_ds.test_start_point, ref_gap=2)

                ## Different  usage of different evaluator
                if ref_gap > 1:
                    mIOU = evaluator(highres_net=highres_net, net=lowres_net, dl=test_loader, n_classes=n_class)
                else:
                    mIOU = evaluator(net=highres_net, dl=test_loader, n_classes=n_class)

                print(ref_gap, 'AR', test_scale, mIOU)
                mIoU_list.append(mIOU)
            
            print(mIoU_list)
            print(np.array(mIoU_list).mean())
            mIoU_list.append(np.array(mIoU_list).mean())

            np.savetxt(os.path.join(args.result_dir,
                                 f"{args.dataset}-{args.backbone}-AR-{test_scale}x-resolution-exp-GOP{GOP}-{bitrate}-evaluation.txt"
                            ), 
                        np.array(mIoU_list)
            )


        ####################################################################
        ####################################################################
        if mode_list['LR']:
            ## Test LR model

            ## Get the LR segmentation model 
            model_dir = os.path.join(args.ckpt_root, f'{args.dataset}-{args.backbone}', 'LR')
            # model_list = os.listdir("./exp/pspnet18-camvid/rebuttal/local_7x7_1")
            model_list = [x for x in os.listdir(model_dir) if x.split('_')[2]==str(test_scale)]
            
            lowres_snapshot = os.path.join(model_dir, model_list[0])

                
            backend = f'{args.dataset}-{args.backbone}'
            lowres_net = build_network_fuse(lowres_snapshot, backend)
            lowres_net.eval()

            mIoU_list = []

            for ref_gap in range(1, GOP+1):
                data_path =os.path.join(args.data_root, f'{args.dataset}-sequence',
                                '%s-GOP%d/decoded_GOP%d_dist_%d/'%(bitrate,GOP,GOP,ref_gap-1)
                            )
                
                if args.dataset == 'camvid':
                    test_ds = CamVid(data_path, mode='test')
                elif args.dataset == 'cityscapes':
                    test_ds = CityScapes(data_path, model_type=f'{args.backbone[:-2]}net', mode='val')

                test_loader = DataLoader(test_ds,
                                batch_size = 1,
                                shuffle = False,
                                num_workers = 4,
                                pin_memory = False,
                                drop_last = True)

                evaluator = EvalConstRes(scale=test_scale, ignore_label=255)

                mIOU = evaluator(net=lowres_net, dl=test_loader, n_classes=n_class)

                print(ref_gap, 'LR', test_scale, mIOU)
                mIoU_list.append(mIOU)

            print(mIoU_list)
            print(np.array(mIoU_list).mean())
            mIoU_list.append(np.array(mIoU_list).mean())
            np.savetxt(os.path.join(args.result_dir,
                            f"{args.dataset}-{args.backbone}-{test_scale}x-resolution-exp-GOP{GOP}-{bitrate}-evaluation.txt"
                            ), 
                        np.array(mIoU_list)
            )
