import os

import torch
torch.manual_seed(233)

from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
import click
import numpy as np
np.random.seed(233)

import random
random.seed(233)

from model.bisenet import BiSeNetV1, OhemCELoss

from dataset.camvid import CamVid
from dataset.cityscapes import CityScapes

from evaluation import EvalConstRes

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
        net.load_state_dict(torch.load(snapshot))
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
@click.option('--models-path', type=str, help='Path for storing model snapshots')
@click.option('--backend', type=str, default='resnet34', help='Feature extractor')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--batch-size', type=int, default=16)
@click.option('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
@click.option('--epochs', type=int, default=20, help='Number of training epochs to run')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--start-lr', type=float, default=0.001)
@click.option('--scale', type=float, default=1.0, help='scale param for augmentation')
@click.option('--feat_loss', type=str, default=None, help='type of feature loss')
@click.option('--dataset', type=str, default='camvid', help='dataset')
@click.option('--model_type', type=str, default='pspnet', help='model that we apply')


def train(data_path, models_path, backend, snapshot, batch_size, alpha, epochs, start_lr, gpu, scale, feat_loss, dataset, model_type):
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
    train_loader, class_weights = None, None

    if dataset == 'camvid':
        cropsize = [960, 720]
        randomscale = (0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
        # randomscale = (1.0,)
        train_ds = CamVid(data_path, cropsize=cropsize, mode='train', randomscale=randomscale)
        val_ds = CamVid(data_path, mode='val')
        class_num = 12

        train_workers = 8
        val_workers = 4
    elif dataset == 'cityscapes':
        # cropsize = [1024, 512]
        # cropsize = [1536, 768]
        cropsize = [512, 1024]
        randomscale = (0.5, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0)
        # randomscale = (0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5)

        # Pretend that all images from the data loader are 0.75x images.
        # In our experiment, we only train & test models with resolutions lower than 0.75
        base_scale = 1.0
        scale = scale/base_scale

        # randomscale = (1.0,)
        train_ds = CityScapes(data_path, model_type=model_type, cropsize=cropsize, mode='train', randomscale=randomscale)
        val_ds = CityScapes(data_path, model_type=model_type, mode='val')
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
    
    
    if model_type == 'pspnet':
        from model.pspnet import PSPNet
        if dataset == 'camvid':
            models = {
                'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=class_num, psp_size=512, deep_features_size=256, backend='squeezenet'),
                'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=class_num, psp_size=1024, deep_features_size=512, backend='densenet'),
                'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=class_num, psp_size=512, deep_features_size=256, backend='resnet18'),
                'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=class_num, psp_size=512, deep_features_size=256, backend='resnet34'),
                'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=class_num, psp_size=2048, deep_features_size=1024, backend='resnet50'),
                'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=class_num, psp_size=2048, deep_features_size=1024, backend='resnet101'),
                'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), n_classes=class_num, psp_size=2048, deep_features_size=1024, backend='resnet152')
            }

            net, starting_epoch = build_network(models, snapshot, backend)
        elif dataset == 'cityscapes':
            from model.pspnet_semseg import PSPNet
            models = {
                'resnet18': lambda: PSPNet(bins=(1, 2, 3, 6), classes=class_num, feat_dim=512, layers=18),
                'resnet50': lambda: PSPNet(bins=(1, 2, 3, 6), classes=class_num, feat_dim=2048, layers=50),
            }
            net, starting_epoch = build_network(models, snapshot, backend)

    elif model_type =='bisenet':
        starting_epoch = 0
        if backend == 'resnet18' or backend == 'resnet34':
            net = BiSeNetV1(n_classes=class_num, backend=backend)
            net = nn.DataParallel(net)
            net = net.cuda()
        else:
            raise NotImplementedError


    if feat_loss:
        if dataset == 'camvid':
            if model_type == 'pspnet':
                net = load_decoder(net, "./exp/pspnet18-camvid/scale1.0_epoch100_pure/PSPNet_resnet18_1.0_92_.pth")
                highres_net, _  = build_network(models, "./exp/pspnet18-camvid/scale1.0_epoch100_pure/PSPNet_resnet18_1.0_92_.pth", backend)
            else:
                raise NotImplementedError

    if not snapshot:
        # import pdb; pdb.set_trace()
        if feat_loss:
            if model_type == 'pspnet':
                for param in net.module.final_conv.parameters():
                    param.requires_grad = False
            else:
                raise NotImplementedError

        optimizer = optim.Adam(net.parameters(), lr=start_lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs*(len(train_ds) // batch_size + 1))

        if dataset=='cityscapes':
            optimizer = optim.SGD(net.parameters(), lr=start_lr, momentum=0.9,weight_decay=5e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs*(len(train_ds) // batch_size + 1))
            # scheduler = PolynomialLRDecay(optimizer, max_decay_steps=epochs*(len(train_ds) // batch_size + 1), end_learning_rate=0.0001, power=0.9)
    else:
        max_iter = epochs*(len(train_ds) // batch_size + 1)
        warmup_start_lr = 1e-5
        warmup_steps = 500
        optimizer = optim.Adam(net.parameters(), lr=warmup_start_lr)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_iter)
        scheduler = GradualWarmupScheduler(optimizer, start_lr/warmup_start_lr, warmup_steps, cosine_scheduler)

    evaluator = EvalConstRes(scale=scale, ignore_label=255)
    
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

        if feat_loss:
            if model_type == 'pspnet':
                highres_net.eval()
            else:
                raise NotImplementedError
        steps = 0
        for x, y, y_cls in train_iterator:
            # import pdb; pdb.set_trace()
            # print(net.module.final_conv.bias)

            steps += batch_size
            optimizer.zero_grad()
            
            with torch.no_grad():
                if feat_loss:
                    if model_type == 'pspnet':
                        _, _, highres_p = highres_net(Variable(x).cuda())

                        downsample_highres_p = F.interpolate(highres_p, [int(cropsize[1]*scale), int(cropsize[0]*scale)], mode='bilinear', align_corners=True)
                    else:
                        raise NotImplementedError

            x = F.interpolate(x, [int(cropsize[1]*scale), int(cropsize[0]*scale)], mode='bilinear', align_corners=True)
            x, y, y_cls = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).cuda()

            if model_type == 'pspnet':
                if dataset=='camvid':
                    out, out_cls, out_p = net(x)
                    out = F.interpolate(out, [cropsize[1], cropsize[0]], mode='bilinear', align_corners=True)
                    seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
                    loss = seg_loss + alpha * cls_loss
                elif dataset=='cityscapes':
                    out, aux = net(x)
                    # import pdb; pdb.set_trace()
                    out = F.interpolate(out, [cropsize[1], cropsize[0]], mode='bilinear', align_corners=True)
                    aux = F.interpolate(aux, [cropsize[1], cropsize[0]], mode='bilinear', align_corners=True)
                    loss = seg_criterion(out,y) + seg_criterion(aux,y)*0.4
            
            elif model_type == 'bisenet':
                out, out_feat16, out_feat32, out_p = net(x)
                out = F.interpolate(out, [cropsize[1], cropsize[0]], mode='bilinear', align_corners=True)
                out_feat16 = F.interpolate(out_feat16, [cropsize[1], cropsize[0]], mode='bilinear', align_corners=True)
                out_feat32 = F.interpolate(out_feat32, [cropsize[1], cropsize[0]], mode='bilinear', align_corners=True)

                seg_loss, seg_loss16, seg_loss32 = seg_criterion(out, y), seg_criterion(out_feat16, y), seg_criterion(out_feat32, y)

                loss = seg_loss + seg_loss16 + seg_loss32
            else:
                raise NotImplementedError

            if feat_loss == 'mse':
                # import pdb; pdb.set_trace()
                loss2 = feat_criterion(downsample_highres_p, out_p)
                loss = loss2 + loss
            elif feat_loss == 'KL':
                loss2 = feat_criterion(downsample_highres_p, out_p)
                loss = loss2 + loss
            
            # epoch_losses.append(loss.data[0])
            epoch_losses.append(loss.item())

            if dataset=='cityscapes':
                status = '[{0}] loss = {1:0.5f} avg = {2:0.5f}, LR = {3:0.7f}'.format(
                    epoch + 1, loss.item(), np.mean(epoch_losses), scheduler.get_lr()[0])
            else:
                status = '[{0}] loss = {1:0.5f} avg = {2:0.5f}, LR = {3:0.7f}'.format(
                    epoch + 1, loss.item(), np.mean(epoch_losses), scheduler.get_last_lr()[0])
            train_iterator.set_description(status)
            loss.backward()
            optimizer.step()

            scheduler.step()

        net.eval()
        mIOU = evaluator(net, val_loader, class_num)
        print("epoch %d: val mIoU %.4f, max mIoU %.4f"%(epoch, mIOU, max_mIoU))
        if mIOU > max_mIoU:
            max_mIoU = mIOU
            torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", backend, str(scale), str(epoch + 1), '.pth'])))

        train_loss = np.mean(epoch_losses)

        

        
if __name__ == '__main__':
    train()
