import argparse
import torch
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert model format")
    parser.add_argument('--backbone', type=str, default='psp18', help='backbone type')
    
    args = parser.parse_args()
    
    dst_dir = 'cityscapes_pretrained'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    if args.backbone == 'psp18':
        orig_dict_dir = './exp/pspnet18-cityscapes/scale1.0_epoch200_pure_bs8_0.5-2.0-aug-512x1024-lr-0.01-semsegPSP/'
        models = os.listdir(orig_dict_dir)
        ## sort models by epoch
        models.sort(key=lambda x: int(x.split('_')[-2].split('.')[0]))
        orig_dict = torch.load(os.path.join(orig_dict_dir, models[-1]))

        orig_dict['module.final_conv.weight'] = orig_dict['module.cls.4.weight']
        orig_dict['module.final_conv.bias'] = orig_dict['module.cls.4.bias']

        torch.save(orig_dict, os.path.join(dst_dir, 'converted_pspnet18_semseg.pth'))
        
    elif args.backbone == 'bise18':
        
        orig_dict = torch.load(os.path.join(dst_dir, './BiseNet_v1_city.pth'))
        new_dict = {}

        for key in orig_dict.keys():
            new_dict['module.'+key] = orig_dict[key]

            if key.startswith('conv_out.conv_out'):
                # print(key)
                end_parse = key.split('.')[2:]
                total_parse = "module.final_conv".split('.') + end_parse
                print(key,'.'.join(total_parse))
                new_dict['.'.join(total_parse)] = orig_dict[key]

            elif key.startswith('conv_out.conv'):
                # print(key)
                end_parse = key.split('.')[2:]
                total_parse = "module.feat_conv_out".split('.') + end_parse
                print(key,'.'.join(total_parse))
                new_dict['.'.join(total_parse)] = orig_dict[key]
        
        torch.save(new_dict, os.path.join(dst_dir, 'converted_bisenet.pth'))
        
    else:
        raise NotImplementedError('backbone type {} is not implemented'.format(args.backbone))
    