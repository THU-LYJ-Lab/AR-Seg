import os

import torch
# this_seed = 233

this_seed = 689
torch.manual_seed(689)

from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
import click
import numpy as np
np.random.seed(689)

import random
random.seed(689)

from model.bisenet import BiSeNetV1WithFuse, OhemCELoss

from dataset.camvid import CamVid, CamVidWithFlow

from dataset.cityscapes import CityScapes, CityScapesWithFlow

from evaluation import EvalConstRes, warpFeature, EvalAlterRes

import torch.nn.functional as F

from model.warmup_scheduler import GradualWarmupScheduler


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_network(models, snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        # _, epoch = os.path.basename(snapshot).split('_')
        # epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot),strict=False)
        print("Snapshot loaded from {}".format(snapshot))
    net = net.cuda()
    return net, epoch


def load_decoder(net, path):
    high_res_state = torch.load(path)
    new_state = {}
    new_state['weight'] = high_res_state['module.final_conv.weight']
    new_state['bias'] = high_res_state['module.final_conv.bias']
    # print(new_state['bias'])
    net.module.final_conv.load_state_dict(new_state)
    
    return net

@click.command()
@click.option('--data-path', type=str, help='Path to dataset folder')
@click.option('--sequence-path', type=str, help='Path to sequence folder')
@click.option('--models-path', type=str, default=None, help='Path for storing model snapshots')
@click.option('--backend', type=str, default='resnet34', help='Feature extractor')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--crop_x', type=int, default=256, help='Horizontal random crop size')
@click.option('--crop_y', type=int, default=256, help='Vertical random crop size')
@click.option('--batch-size', type=int, default=16)
@click.option('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
@click.option('--epochs', type=int, default=20, help='Number of training epochs to run')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--start-lr', type=float, default=0.001)
@click.option('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')
@click.option('--scale', type=float, default=1.0, help='scale param for augmentation')
@click.option('--feat_loss', type=str, default=None, help='type of feature loss')
@click.option('--atten_type', type=str, default='local', help='type of feature loss')
@click.option('--atten_k', type=int, default=7, help='type of feature loss')
@click.option('--stage1_epoch', type=int, default=50, help='Length of the first stage of training: no fusion')
@click.option('--ref_gap', type=int, default=2, help='The length of reference GOP.')
@click.option('--bitrate', type=int, default=3, help='bitrate of dataset.')
@click.option('--with_motion', type=int, default=0)
@click.option('--model_type', type=str, default='pspnet', help='model that we apply')
@click.option('--dataset', type=str, default='camvid', help='dataset')
@click.option('--fuse_version', type=int, default=1, help='Fusion version with different CReFF locations')

def train(data_path, sequence_path, models_path, backend, snapshot, crop_x, crop_y, batch_size, alpha, epochs, start_lr, milestones, gpu, scale, feat_loss, atten_type, stage1_epoch, ref_gap, bitrate, with_motion, model_type, dataset, fuse_version, atten_k):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # data_path = os.path.abspath(os.path.expanduser(data_path))
    # models_path = os.path.abspath(os.path.expanduser(models_path))
    os.makedirs(models_path, exist_ok=True)
    
    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''
    train_loader, class_weights, n_images = None, None, None

    if dataset == 'camvid':
        cropsize = [960, 720]
        randomscale = (0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
        # randomscale = (1.0,)
        train_ds = CamVid(data_path, cropsize=cropsize, mode='train', randomscale=randomscale, load_pair=True, ref_gap=ref_gap)

        val_ds = CamVid(data_path, mode='val', load_pair=True, ref_gap=ref_gap)

        if with_motion:
            train_ds = CamVidWithFlow(data_path, cropsize=cropsize, mode='train', randomscale=randomscale, 
                                    load_pair=True, 
                                    ref_gap=ref_gap,
                                    flow_path=os.path.join(sequence_path, '%dM-GOP%d/MVmap_GOP%d_dist_%d/'%(bitrate,ref_gap, ref_gap, ref_gap-1)))
                                    # ref_path='/mnt/nvme1n1/hyb/data/camvid-sequence/1M-GOP12/frames/')

            val_ds = CamVidWithFlow(data_path, mode='val', load_pair=True, ref_gap=ref_gap,
                                    flow_path=os.path.join(sequence_path, '%dM-GOP%d/MVmap_GOP%d_dist_%d/'%(bitrate, ref_gap, ref_gap,ref_gap-1)))
                                    # ref_path='/mnt/nvme1n1/hyb/data/camvid-sequence/1M-GOP12/frames/')
        
            val_ds_stage1 = CamVid(data_path, mode='val')

        class_num = 12
        train_workers = 8
        val_workers = 4
    elif dataset == 'cityscapes':
        cropsize = [512, 1024]
        randomscale = (0.5, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0)

        if with_motion:
            train_ds = CityScapesWithFlow(data_path, model_type=model_type,cropsize=cropsize, mode='train', randomscale=randomscale, 
                                    ref_gap=ref_gap,
                                    flow_path=os.path.join(sequence_path,'%dM-GOP%d/MVmap_GOP%d_dist_%d/'%(bitrate,ref_gap, ref_gap, ref_gap-1)))
            val_ds = CityScapesWithFlow(data_path, model_type=model_type,mode='val', ref_gap=ref_gap,
                                    flow_path=os.path.join(sequence_path,'%dM-GOP%d/MVmap_GOP%d_dist_%d/'%(bitrate, ref_gap, ref_gap,ref_gap-1)))
            val_ds_stage1 = CityScapes(data_path, model_type=model_type,mode='val')
        else:
            raise NotImplementedError
        
        class_num = 19
        train_workers = 16
        val_workers = 4

    train_loader = DataLoader(train_ds,
                    batch_size = batch_size,
                    shuffle = True,
                    num_workers = train_workers,
                    worker_init_fn=seed_worker,
                    pin_memory = False,
                    drop_last = True)
       

    val_loader = DataLoader(val_ds,
                    batch_size = 1,
                    shuffle = False,
                    num_workers = val_workers,
                    worker_init_fn=seed_worker,
                    pin_memory = False,
                    drop_last = True)

    val_loader_stage1 = DataLoader(val_ds_stage1,
                    batch_size = 1,
                    shuffle = False,
                    num_workers = 4,
                    worker_init_fn=seed_worker,
                    pin_memory = False,
                    drop_last = True)
    
    # import pdb; pdb.set_trace()
    if model_type == 'pspnet':
        if dataset == 'camvid':
            if fuse_version == 1:
                from model.pspnet import PSPNetWithFuse as PSPNet
            elif fuse_version == 2:
                from model.pspnet import PSPNetWithFuseV2 as PSPNet
            elif fuse_version == 3:
                from model.pspnet import PSPNetWithFuseV3 as PSPNet
            psp_models = {
                'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=class_num, psp_size=512, deep_features_size=256, backend='squeezenet'),
                'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=class_num, psp_size=1024, deep_features_size=512, backend='densenet'),
                'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=class_num, psp_size=512, deep_features_size=256, backend='resnet18', atten_k=atten_k, attention_type=atten_type),
                'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=class_num, psp_size=512, deep_features_size=256, backend='resnet34'),
                'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=class_num, psp_size=2048, deep_features_size=1024, backend='resnet50'),
                'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=class_num, psp_size=2048, deep_features_size=1024, backend='resnet101'),
                'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=class_num, psp_size=2048, deep_features_size=1024, backend='resnet152')
            }
            net, starting_epoch = build_network(psp_models, snapshot, backend)

            ckpt_dir = "./exp/pspnet18-camvid/scale1.0_epoch100_pure"
            models = os.listdir(ckpt_dir)
            models.sort(key=lambda x: int(x.split('_')[-2].split('.')[0]))
                    
            if feat_loss:
                if backend=='resnet18':
                    net = load_decoder(net, os.path.join(ckpt_dir, models[-1]))
                else:
                    raise NotImplementedError

            if backend=='resnet18':
                highres_net, _  = build_network(psp_models, os.path.join(ckpt_dir, models[-1]), backend)
            else:
                raise NotImplementedError
        
        elif dataset == 'cityscapes':
            from model.pspnet_semseg import PSPNetWithFuse as PSPNet
            psp_models = {
                'resnet18': lambda: PSPNet(bins=(1, 2, 3, 6), classes=class_num, feat_dim=512, layers=18),
            }
            
            ## The HR model is trained with PSPNet model, 
            ## here we need to load it as a PSPNetWithFuse model, so we use additional scropts to convert it.
            pretrained_pspnet = 'cityscapes_pretrained/converted_pspnet18_semseg.pth'
            net, starting_epoch = build_network(psp_models, snapshot, backend)
            # net, starting_epoch = build_network(psp_models, pretrained_pspnet, backend)

            if feat_loss:
                net = load_decoder(net, pretrained_pspnet)

            highres_net, _  = build_network(psp_models, pretrained_pspnet, backend)

            # raise NotImplementedError
    
    elif model_type == 'bisenet':
        bise_models = {
            'resnet18': lambda: BiSeNetV1WithFuse(n_classes=class_num, backend='resnet18'),
            'resnet34': lambda: BiSeNetV1WithFuse(n_classes=class_num, backend='resnet34'),
        }

        if dataset == 'camvid':
            net, starting_epoch = build_network(bise_models, snapshot, backend)

            ckpt_dir = "./exp/bisenet18-camvid/scale1.0_epoch100_pure"
            models = os.listdir(ckpt_dir)
            models.sort(key=lambda x: int(x.split('_')[-2].split('.')[0]))
                
            if feat_loss:
                net = load_decoder(net, os.path.join(ckpt_dir, models[-1]))

            highres_net, _  = build_network(bise_models, os.path.join(ckpt_dir, models[-1]), backend)

        elif dataset == 'cityscapes':
            pretrained_bisenet = './cityscapes_pretrained/converted_bisenet.pth'
            highres_net, _  = build_network(bise_models,pretrained_bisenet, backend)

            lowres_snapshot = pretrained_bisenet
            net, starting_epoch = build_network(bise_models, lowres_snapshot, backend)
            if feat_loss:
                net = load_decoder(net, pretrained_bisenet)
            
    
    # scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')])

    if not snapshot:
        # import pdb; pdb.set_trace()
        if feat_loss:
            for param in net.module.final_conv.parameters():
                param.requires_grad = False

        optimizer = optim.Adam(net.parameters(), lr=start_lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs*(len(train_ds) // batch_size + 1))

        if dataset=='cityscapes':
            optimizer = optim.SGD(net.parameters(), lr=start_lr, momentum=0.9,weight_decay=5e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs*(len(train_ds) // batch_size + 1))


    else:
        max_iter = epochs*(len(train_ds) // batch_size + 1)
        warmup_start_lr = 1e-5
        warmup_steps = 500
        optimizer = optim.Adam(net.parameters(), lr=warmup_start_lr)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_iter)
        scheduler = GradualWarmupScheduler(optimizer, start_lr/warmup_start_lr, warmup_steps, cosine_scheduler)

    if with_motion:
        evaluator_stage2 = EvalAlterRes(scale=scale, ignore_label=255)
    else:
        raise NotImplementedError('Phase 2 should be trained with motion vectors.')

    evaluator_stage1 = EvalConstRes(scale=scale, ignore_label=255)
    
    max_mIoU = 0.0

    for epoch in range(starting_epoch, starting_epoch + epochs):
        if model_type == 'pspnet':
            seg_criterion = nn.NLLLoss(weight=class_weights, ignore_index=255)
            cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
            if dataset == 'cityscapes':
                seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
        elif model_type == 'bisenet':
            seg_criterion = OhemCELoss(0.7, ignore_lb=255)

        if feat_loss == 'mse':
            feat_criterion = nn.MSELoss()
        elif feat_loss == "KL":
            feat_criterion = nn.KLDivLoss(log_target=True, reduction='mean')

        epoch_losses = []
        train_iterator = tqdm(train_loader, total=len(train_ds) // batch_size + 1)
        net.train()
        highres_net.eval()
        steps = 0
        for data_item  in train_iterator:
            if with_motion:
                x, y, y_cls, ref_x, flow = data_item
            else:
                x, y, y_cls, ref_x = data_item
            # import pdb; pdb.set_trace()
            # print(net.module.final_conv.bias)

            steps += batch_size
            optimizer.zero_grad()
            
            with torch.no_grad():
                if model_type == 'pspnet':
                    _, _, highres_p = highres_net(Variable(x).cuda())
                    # print(x.shape, highres_p.shape)
                    if epoch >= stage1_epoch:
                        _, _, highres_ref_p = highres_net(Variable(ref_x).cuda())
                    # import pdb; pdb.set_trace()
                elif model_type == 'bisenet':
                    _, _, _,  highres_p = highres_net(Variable(x).cuda())
                    if epoch >= stage1_epoch:
                        _, _, _, highres_ref_p = highres_net(Variable(ref_x).cuda())
                else:
                    raise NotImplementedError

            downsample_highres_p = F.interpolate(highres_p, [int(cropsize[1]*scale), int(cropsize[0]*scale)], mode='bilinear', align_corners=True)

            x = F.interpolate(x, [int(cropsize[1]*scale), int(cropsize[0]*scale)], mode='bilinear', align_corners=True)
            x, y, y_cls = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).cuda()

            

            if epoch >= stage1_epoch:
                if with_motion:
                    flow = flow.cuda()

                    flow = flow.transpose(2,3).transpose(1,2)
                    flow = flow * highres_ref_p.shape[-2] / flow.shape[-2]
                    flow = F.interpolate(flow, [highres_ref_p.shape[-2], highres_ref_p.shape[-1]], mode='nearest')
                    flow = flow.transpose(1,2).transpose(2,3)
                    
                    highres_ref_p = warpFeature(highres_ref_p, flow)

                if model_type == 'pspnet':
                    out, out_cls, out_p = net(x, mode='merge', ref_p = highres_ref_p)
                elif model_type == 'bisenet':
                    out, out_feat16, out_feat32, out_p = net(x, mode='merge', ref_p = highres_ref_p)
                # out_cls, out_p = net.module.forward_phase1(x)

                # out, out_p = net.module.forward_phase2(out_p, highres_ref_p)
                # import pdb; pdb.set_trace()
            else:
                if model_type == 'pspnet':
                    out, out_cls, out_p = net(x, mode='normal')
                    
                elif model_type == 'bisenet':
                    out, out_feat16, out_feat32, out_p = net(x, mode='normal')
                else:
                    raise NotImplementedError

            if model_type == 'pspnet':
                out_p = F.interpolate(out_p, [highres_p.shape[-2], highres_p.shape[-1]], mode='bilinear', align_corners=True)
                out = F.interpolate(out, [cropsize[1], cropsize[0]], mode='bilinear', align_corners=True)
                if dataset == 'camvid':
                    seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
                    loss = seg_loss + alpha * cls_loss
                elif dataset == 'cityscapes':
                    out_cls = F.interpolate(out_cls, [cropsize[1], cropsize[0]], mode='bilinear', align_corners=True)
                    loss = seg_criterion(out,y) + seg_criterion(out_cls,y)*0.4
            else:
                out_p = F.interpolate(out_p, [highres_p.shape[-2], highres_p.shape[-1]], mode='bilinear', align_corners=True)
                out = F.interpolate(out, [cropsize[1], cropsize[0]], mode='bilinear', align_corners=True)
                out_feat16 = F.interpolate(out_feat16, [cropsize[1], cropsize[0]], mode='bilinear', align_corners=True)
                out_feat32 = F.interpolate(out_feat32, [cropsize[1], cropsize[0]], mode='bilinear', align_corners=True)

                seg_loss, seg_loss16, seg_loss32 = seg_criterion(out, y), seg_criterion(out_feat16, y), seg_criterion(out_feat32, y)

                loss = seg_loss + seg_loss16 + seg_loss32

            
            if feat_loss == 'mse':
                # import pdb; pdb.set_trace()
                # print(highres_p.shape, out_p.shape)
                # exit(0)
                loss2 = feat_criterion(highres_p, out_p)
                # print(loss2, loss)
                # exit(0)
                loss = loss2 + loss
            elif feat_loss == 'KL':
                loss2 = feat_criterion(highres_p, out_p)
                loss = loss2 + loss
            
            # epoch_losses.append(loss.data[0])
            epoch_losses.append(loss.item())

            status = '[{0}] loss = {1:0.5f} avg = {2:0.5f}, LR = {3:0.7f}'.format(
                epoch + 1, loss.item(), np.mean(epoch_losses), scheduler.get_last_lr()[0])
            train_iterator.set_description(status)
            loss.backward()
            optimizer.step()

            scheduler.step()

        # import pdb; pdb.set_trace()
        
        net.eval()

        if epoch >= stage1_epoch:
            mIOU = evaluator_stage2(highres_net, net, val_loader, class_num)
        else:
            mIOU = evaluator_stage1(net, val_loader_stage1, class_num)
        print("epoch %d: val mIoU %.4f, max mIoU %.4f"%(epoch, mIOU, max_mIoU))

        # print(seg_loss, cls_loss, feat_criterion(highres_p, out_p))
        # exit(0)

        if mIOU > max_mIoU:
            max_mIoU = mIOU
            torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", backend, str(scale), str(epoch + 1), '.pth'])))

        train_loss = np.mean(epoch_losses)

        

        
if __name__ == '__main__':
    train()
