import os
import torch
import torch.utils.data as data

import torchvision.transforms as transforms

from PIL import Image

from dataset.transform import *

import numpy as np
np.random.seed(233)


scene_length_info = {
    '0001TP': {
        'encoded_start_idx': 31,
        'encoded_end_idx': 3721,
        'dataset_start_idx': 6690,
        'dataset_end_idx': 10380,
    },
    '0006R0': {
        'encoded_start_idx': 932,
        'encoded_end_idx': 3932,
        'dataset_start_idx': 930,
        'dataset_end_idx': 3930,
    },
    '0016E5': {
        'encoded_start_idx': 392,
        'encoded_end_idx': 8642,
        'dataset_start_idx': 390,
        'dataset_end_idx': 8640,
    },
    'Seq05VD': {
        'encoded_start_idx': 32,
        'encoded_end_idx': 5102,
        'dataset_start_idx': 30,
        'dataset_end_idx': 5100,
    }
}

def get_files(folder, name_filter=None, extension_filter=None):
    """Helper function that returns the list of files in a specified folder
    with a specified extension.
    Keyword arguments:
    - folder (``string``): The path to a folder.
    - name_filter (```string``, optional): The returned files must contain
    this substring in their filename. Default: None; files are not filtered.
    - extension_filter (``string``, optional): The desired file extension.
    Default: None; files are not filtered
    """
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    # Filename filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files that do not
    # contain "name_filter"
    if name_filter is None:
        # This looks hackish...there is probably a better way
        name_cond = lambda filename: True
    else:
        name_cond = lambda filename: name_filter in filename

    # Extension filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files whose extension
    # is not "extension_filter"
    if extension_filter is None:
        # This looks hackish...there is probably a better way
        ext_cond = lambda filename: True
    else:
        ext_cond = lambda filename: filename.endswith(extension_filter)

    filtered_files = []

    # Explore the directory tree to get files that contain "name_filter" and
    # with extension "extension_filter"
    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            if name_cond(file) and ext_cond(file):
                full_path = os.path.join(path, file)
                filtered_files.append(full_path)

    return filtered_files

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

def pil_loader_single(data_path):
    """Loads a sample and label image given their path as PIL images.
    Keyword arguments:
    - data_path (``string``): The filepath to the image.
    - label_path (``string``): The filepath to the ground-truth image.
    Returns the image and the label as PIL images.
    """
    data = Image.open(data_path)

    return data

class CamVid(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.
    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.
    """
    # Training dataset root folders
    train_folder = 'train'
    train_lbl_folder = 'train_labels_with_ignored'

    # Validation dataset root folders
    val_folder = 'val'
    val_lbl_folder = 'val_labels_with_ignored'

    # Test dataset root folders
    test_folder = 'test'
    test_lbl_folder = 'test_labels_with_ignored'

    # Images extension
    img_extension = '.png'

    # Default encoding for pixel value, class name, and class color
    _cmap = {
        0: (128, 128, 128),    # sky
        1: (128, 0, 0),        # building
        2: (192, 192, 128),    # column_pole
        3: (128, 64, 128),     # road
        4: (0, 0, 192),        # sidewalk
        5: (128, 128, 0),      # Tree
        6: (192, 128, 128),    # SignSymbol
        7: (64, 64, 128),      # Fence
        8: (64, 0, 128),       # Car
        9: (64, 64, 0),        # Pedestrian
        10: (0, 128, 192),     # Bicyclist
        11: (0, 0, 0)}         # Void
    
    _color2label = dict(zip(_cmap.values(), _cmap.keys()))

    def __init__(self,
                 root_dir,
                 mode='train',
                 cropsize=(640, 480),
                 randomscale=(0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5),
                #  transform=None,
                #  label_transform=None,
                 loader=pil_loader,
                 load_pair=False,
                 ref_gap=5,
                 ref_path = '/mnt/nvme1n1/hyb/data/camvid-sequence/frames/'):

        self.root_dir = root_dir
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        # self.transform = transform
        # self.label_transform = label_transform
        self.loader = loader
        self.load_pair = load_pair
        self.ref_gap = ref_gap
        self.ref_path = ref_path
        
        ## The frame idx gap in CamVid dataset
        self.data_frame_idx_gap = [2,30]

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.39068785, 0.40521392, 0.41434407), (0.29652068, 0.30514979, 0.30080369)),
            ])
        if not self.load_pair:
            self.trans_train = Compose([
                ColorJitter(
                    brightness = 0.5,
                    contrast = 0.5,
                    saturation = 0.5),
                HorizontalFlip(),
                # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
                RandomScale(randomscale),
                # RandomScale((0.125, 1)),
                # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
                # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
                RandomCrop(cropsize)
                ])
        else:
            self.trans_train = pairCompose([
                pairColorJitter(
                    brightness = 0.5,
                    contrast = 0.5,
                    saturation = 0.5),
                pairHorizontalFlip(),
                # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
                pairRandomScale(randomscale),
                # RandomScale((0.125, 1)),
                # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
                # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
                pairRandomCrop(cropsize)
                ])

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = get_files(
                os.path.join(root_dir, self.train_folder),
                extension_filter=self.img_extension)

            self.train_labels = get_files(
                os.path.join(root_dir, self.train_lbl_folder),
                extension_filter=self.img_extension)

            ## This image corresponds to the second frame of sequence Seq05VD, 
            ## which does not fit large ref_gap, so we omit this sample during training phase
            if len(self.train_data) != len(self.train_labels):
                # import pdb; pdb.set_trace()
                ignore_name = 'Seq05VD_f00000'
                bool_list = [(ignore_name in x) for x in self.train_labels]
                idx = np.where(bool_list)[0][0]
                del[self.train_labels[idx]]

            # self.train_start_point = self.find_start_point(self.train_data)
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = get_files(
                os.path.join(root_dir, self.val_folder),
                extension_filter=self.img_extension)

            self.val_labels = get_files(
                os.path.join(root_dir, self.val_lbl_folder),
                extension_filter=self.img_extension)

            # self.val_start_point = self.find_start_point(self.val_data)
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.img_extension)

            self.test_labels = get_files(
                os.path.join(root_dir, self.test_lbl_folder),
                extension_filter=self.img_extension)

            # self.test_start_point = self.find_start_point(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset
        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.
        """
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[index]   

        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
            
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
           

        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)
        # print(index, data_path)
        
        # print(index, ref_index)
        if self.load_pair:
            seq_name = os.path.basename(data_path).split('_')[0]
            data_frame_idx = self.getDatasetFrameIdx(os.path.basename(data_path), seq_name)

            decoded_frame_idx = data_frame_idx - scene_length_info[seq_name]['dataset_start_idx'] + scene_length_info[seq_name]['encoded_start_idx']

            ref_decoded_frame_idx = decoded_frame_idx - (self.ref_gap-1)

            decoded_basename = self.getDecodedBaseName(ref_decoded_frame_idx, seq_name)

            ref_data_path = os.path.join(self.ref_path, seq_name, decoded_basename)

            # print(data_path, ref_data_path)

            ref_img = Image.open(ref_data_path)

            ## ref label placeholder
            ref_label = label
        

        if self.mode == 'train' or self.mode == 'trainval':
            if not self.load_pair:
                im_lb = dict(im = img, lb = label)
                im_lb = self.trans_train(im_lb)
                img, label = im_lb['im'], im_lb['lb']
            else:
                im_lb = dict(im = img, lb = label)
                ref_im_lb = dict(im = ref_img, lb = ref_label)
                im_lb, ref_im_lb = self.trans_train(im_lb, ref_im_lb)
                # print(im_lb, ref_im_lb)
                img, label = im_lb['im'], im_lb['lb']
                ref_img, ref_label = ref_im_lb['im'], ref_im_lb['lb']

        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)
        # label = self.convert_labels(label)

        if self.load_pair:
            ref_img = self.to_tensor(ref_img)
            ref_label = np.array(ref_label).astype(np.int64)[np.newaxis, :]
        
        if not self.load_pair:
            # print(data_path)
            existence_list = self.gen_label_existence(label)
            return img, label, existence_list
        else:
            existence_list = self.gen_label_existence(label)
            ref_existence_list = self.gen_label_existence(ref_label)
            return img, label, existence_list, ref_img

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
    
    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label
    

    def gen_label_existence(self, label):
        existence_list = torch.zeros((12,))
        label_unique = np.unique(label)

        for label in label_unique:
            if label == 255:
                continue
            existence_list[label] = 1
        
        return existence_list

    def find_start_point(self, file_list):
        base_name_list = [os.path.basename(x) for x in file_list]

        frame_idx_list = []
        seq_name_list = []
        for name in base_name_list:
            seq_name = name.split('_')[0]
            seq_name_list.append(seq_name)

            if seq_name == '0001TP' or seq_name == '0016E5':
                frame_idx = int(name.split('_')[-1][:-4])
            elif seq_name == '0006R0' or seq_name == 'Seq05VD':
                frame_idx = int(name.split('_')[-1][:-4][1:])
            else:
                print("Unknown Sequence name!")
                exit(1)

            frame_idx_list.append(frame_idx)
            
            # print(name, seq_name, frame_idx)
        
        frame_idx_list = np.array(frame_idx_list)
        frame_idx_gap = np.ediff1d(frame_idx_list)
        middle_frame_ind = [False] + [(x in self.data_frame_idx_gap) for x in frame_idx_gap]
        start_frame_ind = np.invert(middle_frame_ind)

        start_point_list = np.array(range(len(base_name_list)))[start_frame_ind]

        start_point_dict = {}
        start_point_idx = 0
        for idx in range(len(base_name_list)):
            if start_point_idx+1 < len(start_point_list) and idx >= start_point_list[start_point_idx+1]:
                start_point_idx +=1 

            this_start_point = start_point_list[start_point_idx]
            start_point_dict[idx] = this_start_point

        return start_point_dict

    def getDatasetFrameIdx(self, basename, seq_name):
        if seq_name == "0001TP" or seq_name == "0016E5":
            frame_idx = int(basename.split('_')[1][:-4])
        elif seq_name == '0006R0' or seq_name == "Seq05VD":
            frame_idx = int(basename.split('_')[1][1:-4])

        return frame_idx
    
    def getDecodedBaseName(self, frame_idx, seq_name):
        # if seq_name == "0001TP":
        #     basename = "_".join([seq_name, "%06d.png"%frame_idx])
        # elif seq_name == "0006R0" or seq_name == "Seq05VD":
        #     basename = "_".join([seq_name, "f%05d.png"%frame_idx])
        # elif seq_name == '0016E5':
        #     basename = "_".join([seq_name, "%05d.png"%frame_idx])

        basename = "_".join([seq_name, "%06d.png"%frame_idx])
        
        return basename


class CamVidWithFlow(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.
    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.
    """
    # Training dataset root folders
    train_folder = 'train'
    train_lbl_folder = 'train_labels_with_ignored'

    # Validation dataset root folders
    val_folder = 'val'
    val_lbl_folder = 'val_labels_with_ignored'

    # Test dataset root folders
    test_folder = 'test'
    test_lbl_folder = 'test_labels_with_ignored'

    # Images extension
    img_extension = '.png'

    # Default encoding for pixel value, class name, and class color
    _cmap = {
        0: (128, 128, 128),    # sky
        1: (128, 0, 0),        # building
        2: (192, 192, 128),    # column_pole
        3: (128, 64, 128),     # road
        4: (0, 0, 192),        # sidewalk
        5: (128, 128, 0),      # Tree
        6: (192, 128, 128),    # SignSymbol
        7: (64, 64, 128),      # Fence
        8: (64, 0, 128),       # Car
        9: (64, 64, 0),        # Pedestrian
        10: (0, 128, 192),     # Bicyclist
        11: (0, 0, 0)}         # Void
    
    _color2label = dict(zip(_cmap.values(), _cmap.keys()))

    def __init__(self,
                 root_dir,
                 mode='train',
                 cropsize=(640, 480),
                 randomscale=(0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5),
                #  transform=None,
                #  label_transform=None,
                 loader=pil_loader,
                 load_pair=False,
                 ref_gap=5,
                 ref_path = '/mnt/nvme1n1/hyb/data/camvid-sequence/frames/',
                 flow_path = '/mnt/nvme1n1/hyb/data/camvid-sequence/MVmap/'):

        self.root_dir = root_dir
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        # self.transform = transform
        # self.label_transform = label_transform
        self.loader = loader
        self.load_pair = load_pair
        self.ref_gap = ref_gap
        self.ref_path = ref_path
        self.flow_path = flow_path
        
        ## The frame idx gap in CamVid dataset
        self.data_frame_idx_gap = [2,30]

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.39068785, 0.40521392, 0.41434407), (0.29652068, 0.30514979, 0.30080369)),
            ])

        self.trans_train_color = pairColorJitter(
            brightness = 0.5,
            contrast = 0.5,
            saturation = 0.5
        )
        self.trans_train_homo = pairCompose([
            pairOFHorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            # pairOFRandomScale(randomscale),
            pairOFRandomScaleV2(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            pairOFRandomCrop(cropsize)
            ])

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = get_files(
                os.path.join(root_dir, self.train_folder),
                extension_filter=self.img_extension)

            self.train_labels = get_files(
                os.path.join(root_dir, self.train_lbl_folder),
                extension_filter=self.img_extension)

            ## This image corresponds to the second frame of sequence Seq05VD, 
            ## which does not fit large ref_gap, so we omit this sample during training phase
            # if len(self.train_data) != len(self.train_labels):
            # import pdb; pdb.set_trace()
            ignore_name = 'Seq05VD_f00000'
            bool_list = [(ignore_name in x) for x in self.train_labels]
            idx = np.where(bool_list)[0][0]
            del[self.train_labels[idx]]
            bool_list = [(ignore_name in x) for x in self.train_data]
            idx = np.where(bool_list)[0][0]
            del[self.train_data[idx]]

            # self.train_start_point = self.find_start_point(self.train_data)
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = get_files(
                os.path.join(root_dir, self.val_folder),
                extension_filter=self.img_extension)

            self.val_labels = get_files(
                os.path.join(root_dir, self.val_lbl_folder),
                extension_filter=self.img_extension)

            # self.val_start_point = self.find_start_point(self.val_data)
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            # print(os.path.join(root_dir, self.test_folder))
            self.test_data = get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.img_extension)

            self.test_labels = get_files(
                os.path.join(root_dir, self.test_lbl_folder),
                extension_filter=self.img_extension)

            # self.test_start_point = self.find_start_point(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset
        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.
        """
        # index = 38
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]   

        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
            
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
           

        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)
        # print(data_path)
        # print(data_path)
        
        # print(index, ref_index)
        if self.load_pair:
            seq_name = os.path.basename(data_path).split('_')[0]
            data_frame_idx = self.getDatasetFrameIdx(os.path.basename(data_path), seq_name)

            decoded_frame_idx = data_frame_idx - scene_length_info[seq_name]['dataset_start_idx'] + scene_length_info[seq_name]['encoded_start_idx']

            ref_decoded_frame_idx = decoded_frame_idx - (self.ref_gap-1)

            decoded_basename = self.getDecodedBaseName(ref_decoded_frame_idx, seq_name)

            ref_data_path = os.path.join(self.ref_path, seq_name, decoded_basename)

            # print(data_path, ref_data_path)

            ref_img = Image.open(ref_data_path)

            ## ref label placeholder
            ref_label = label

            f = open(os.path.join(self.flow_path, seq_name, os.path.basename(data_path)[:-4]+'.bin'))
            flow = np.fromfile(f,np.dtype(np.short)).reshape(720,960,2)/4
            f.close()

            # warped_img = self.warp_img(ref_img, flow)
            # # cv2.imwrite("ref_img.png", np.array(ref_img).astype(np.uint8)[...,::-1])
            # cv2.imwrite("./image_test_result/test-%03d-true_img.png"%index, np.array(img).astype(np.uint8)[...,::-1])
            # cv2.imwrite("./image_test_result/test-%03d-warpped_img.png"%index, np.array(warped_img).astype(np.uint8)[...,::-1])
            # # exit(0)

            flow_map = flow

        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            ref_im_lb = dict(im = ref_img, lb = ref_label)
            im_lb, ref_im_lb = self.trans_train_color(im_lb, ref_im_lb)
            
            ref_im_lb = dict(im = ref_im_lb['im'], lb = flow)
            im_lb, ref_im_lb = self.trans_train_homo(im_lb, ref_im_lb)

            # print(im_lb, ref_im_lb)
            img, label = im_lb['im'], im_lb['lb']
            ref_img, flow_map = ref_im_lb['im'], ref_im_lb['lb']

        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)
        # label = self.convert_labels(label)

        if self.load_pair:
            ref_img = self.to_tensor(ref_img)
        
        if not self.load_pair:
            # print(data_path)
            existence_list = self.gen_label_existence(label)
            return img, label, existence_list
        else:
            existence_list = self.gen_label_existence(label)
            return img, label, existence_list, ref_img, flow_map

    def warp_img(self, ref_img, flow):

        max_abs = 150
        mat = flow

        mat = (mat + max_abs)/2/max_abs

        mat[mat<0] = 0
        mat[mat>1] = 1

        mat = mat*255

        cv2.imwrite(os.path.join("proxy_x.png"),mat[...,0].astype(np.uint8))
        cv2.imwrite(os.path.join("proxy_y.png"),mat[...,1].astype(np.uint8))   

        ref_img = np.array(ref_img)

        h, w = flow.shape[:2]
        # import pdb; pdb.set_trace()
        # flow = -flow
        flow = np.concatenate([
            (flow[:,:,0] + np.arange(w))[...,np.newaxis], 
            (flow[:,:,1] + np.arange(h)[:,np.newaxis])[...,np.newaxis]
            ],axis=2)
        # import pdb; pdb.set_trace()
        result_img = cv2.remap(ref_img, flow.astype(np.float32), None, cv2.INTER_LINEAR)

        return result_img

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
    
    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label
    

    def gen_label_existence(self, label):
        existence_list = torch.zeros((12,))
        label_unique = np.unique(label)

        for label in label_unique:
            if label == 255:
                continue
            existence_list[label] = 1
        
        return existence_list

    def find_start_point(self, file_list):
        base_name_list = [os.path.basename(x) for x in file_list]

        frame_idx_list = []
        seq_name_list = []
        for name in base_name_list:
            seq_name = name.split('_')[0]
            seq_name_list.append(seq_name)

            if seq_name == '0001TP' or seq_name == '0016E5':
                frame_idx = int(name.split('_')[-1][:-4])
            elif seq_name == '0006R0' or seq_name == 'Seq05VD':
                frame_idx = int(name.split('_')[-1][:-4][1:])
            else:
                print("Unknown Sequence name!")
                exit(1)

            frame_idx_list.append(frame_idx)
            
            # print(name, seq_name, frame_idx)
        
        frame_idx_list = np.array(frame_idx_list)
        frame_idx_gap = np.ediff1d(frame_idx_list)
        middle_frame_ind = [False] + [(x in self.data_frame_idx_gap) for x in frame_idx_gap]
        start_frame_ind = np.invert(middle_frame_ind)

        start_point_list = np.array(range(len(base_name_list)))[start_frame_ind]

        start_point_dict = {}
        start_point_idx = 0
        for idx in range(len(base_name_list)):
            if start_point_idx+1 < len(start_point_list) and idx >= start_point_list[start_point_idx+1]:
                start_point_idx +=1 

            this_start_point = start_point_list[start_point_idx]
            start_point_dict[idx] = this_start_point

        return start_point_dict

    def getDatasetFrameIdx(self, basename, seq_name):
        if seq_name == "0001TP" or seq_name == "0016E5":
            frame_idx = int(basename.split('_')[1][:-4])
        elif seq_name == '0006R0' or seq_name == "Seq05VD":
            frame_idx = int(basename.split('_')[1][1:-4])

        return frame_idx
    
    def getDecodedBaseName(self, frame_idx, seq_name):
        # if seq_name == "0001TP":
        #     basename = "_".join([seq_name, "%06d.png"%frame_idx])
        # elif seq_name == "0006R0" or seq_name == "Seq05VD":
        #     basename = "_".join([seq_name, "f%05d.png"%frame_idx])
        # elif seq_name == '0016E5':
        #     basename = "_".join([seq_name, "%05d.png"%frame_idx])
        
        basename = "_".join([seq_name, "%06d.png"%frame_idx])

        return basename



class CamVidWithBiFlow(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.
    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.
    """
    # Training dataset root folders
    train_folder = 'train'
    train_lbl_folder = 'train_labels_with_ignored'

    # Validation dataset root folders
    val_folder = 'val'
    val_lbl_folder = 'val_labels_with_ignored'

    # Test dataset root folders
    test_folder = 'test'
    test_lbl_folder = 'test_labels_with_ignored'

    # Images extension
    img_extension = '.png'

    # Default encoding for pixel value, class name, and class color
    _cmap = {
        0: (128, 128, 128),    # sky
        1: (128, 0, 0),        # building
        2: (192, 192, 128),    # column_pole
        3: (128, 64, 128),     # road
        4: (0, 0, 192),        # sidewalk
        5: (128, 128, 0),      # Tree
        6: (192, 128, 128),    # SignSymbol
        7: (64, 64, 128),      # Fence
        8: (64, 0, 128),       # Car
        9: (64, 64, 0),        # Pedestrian
        10: (0, 128, 192),     # Bicyclist
        11: (0, 0, 0)}         # Void
    
    _color2label = dict(zip(_cmap.values(), _cmap.keys()))

    def __init__(self,
                 root_dir,
                 mode='train',
                 cropsize=(640, 480),
                 randomscale=(0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5),
                #  transform=None,
                #  label_transform=None,
                 loader=pil_loader,
                 load_pair=False,
                 ref_gap=6,
                 ref_path = '/mnt/nvme1n1/hyb/data/camvid-sequence/frames/',
                 flow_path = '/mnt/nvme1n1/hyb/data/camvid-sequence/MVmap/'):

        self.root_dir = root_dir
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        # self.transform = transform
        # self.label_transform = label_transform
        self.loader = loader
        self.load_pair = load_pair
        self.ref_gap = ref_gap
        self.ref_path = ref_path
        self.flow_path = flow_path
        
        ## The frame idx gap in CamVid dataset
        self.data_frame_idx_gap = [2,30]

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.39068785, 0.40521392, 0.41434407), (0.29652068, 0.30514979, 0.30080369)),
            ])

        self.trans_train_color = tripleColorJitter(
            brightness = 0.5,
            contrast = 0.5,
            saturation = 0.5
        )
        self.trans_train_homo = tripleCompose([
            tripleOFHorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            tripleOFRandomScaleV2(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            tripleOFRandomCrop(cropsize)
            ])

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = get_files(
                os.path.join(root_dir, self.train_folder),
                extension_filter=self.img_extension)

            self.train_labels = get_files(
                os.path.join(root_dir, self.train_lbl_folder),
                extension_filter=self.img_extension)

            ## This image corresponds to the second frame of sequence Seq05VD, 
            ## which does not fit large ref_gap, so we omit this sample during training phase
            # if len(self.train_data) != len(self.train_labels):
            if True:
                # import pdb; pdb.set_trace()
                ignore_name = 'Seq05VD_f00000'
                bool_list = [(ignore_name in x) for x in self.train_labels]
                idx = np.where(bool_list)[0][0]
                del[self.train_labels[idx]]
                bool_list = [(ignore_name in x) for x in self.train_data]
                idx = np.where(bool_list)[0][0]
                del[self.train_data[idx]]

            # self.train_start_point = self.find_start_point(self.train_data)
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = get_files(
                os.path.join(root_dir, self.val_folder),
                extension_filter=self.img_extension)

            self.val_labels = get_files(
                os.path.join(root_dir, self.val_lbl_folder),
                extension_filter=self.img_extension)

            # self.val_start_point = self.find_start_point(self.val_data)
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            # print(os.path.join(root_dir, self.test_folder))
            self.test_data = get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.img_extension)

            self.test_labels = get_files(
                os.path.join(root_dir, self.test_lbl_folder),
                extension_filter=self.img_extension)

            # self.test_start_point = self.find_start_point(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset
        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.
        """
        # index = 38
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]   

        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
            
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
           

        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)
        # print(data_path)
        # print(data_path)
        
        # print(index, ref_index)
        if self.load_pair:
            seq_name = os.path.basename(data_path).split('_')[0]
            data_frame_idx = self.getDatasetFrameIdx(os.path.basename(data_path), seq_name)

            decoded_frame_idx = data_frame_idx - scene_length_info[seq_name]['dataset_start_idx'] + scene_length_info[seq_name]['encoded_start_idx']

            ref_decoded_frame_idx = decoded_frame_idx - self.ref_gap
            decoded_basename = self.getDecodedBaseName(ref_decoded_frame_idx, seq_name)
            ref_data_path = os.path.join(self.ref_path, seq_name, decoded_basename)

            ref_decoded_frame_idx2 = decoded_frame_idx + (12-self.ref_gap)
            decoded_basename2 = self.getDecodedBaseName(ref_decoded_frame_idx2, seq_name)
            ref_data_path2 = os.path.join(self.ref_path, seq_name, decoded_basename2)

            # print(decoded_frame_idx, data_path, ref_data_path, ref_data_path2)

            ref_img = Image.open(ref_data_path)

            # print(data_path, ref_data_path)

            ref_img2 = Image.open(ref_data_path2)

            ## ref label placeholder
            ref_label = label

            f = open(os.path.join(self.flow_path, seq_name, os.path.basename(data_path)[:-4]+'_last.bin'))
            flow = np.fromfile(f,np.dtype(np.short)).reshape(720,960,2)/4
            f.close()

            f = open(os.path.join(self.flow_path, seq_name, os.path.basename(data_path)[:-4]+'_next.bin'))
            flow2 = np.fromfile(f,np.dtype(np.short)).reshape(720,960,2)/4
            f.close()

            # warped_img = self.warp_img(ref_img, flow)
            # # cv2.imwrite("ref_img.png", np.array(ref_img).astype(np.uint8)[...,::-1])
            # cv2.imwrite("./image_test_result/test-%03d-true_img.png"%index, np.array(img).astype(np.uint8)[...,::-1])
            # cv2.imwrite("./image_test_result/test-%03d-warpped_img.png"%index, np.array(warped_img).astype(np.uint8)[...,::-1])
            # # exit(0)

            flow_map = flow
            flow_map2 = flow2

        ## Add another reference frame and augmentation
        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            ref_im_lb = dict(im = ref_img, lb = ref_label)
            ref_im_lb2 = dict(im = ref_img2, lb = ref_label)
            im_lb, ref_im_lb, ref_im_lb2 = self.trans_train_color(im_lb, ref_im_lb, ref_im_lb2)
            
            ref_im_lb = dict(im = ref_im_lb['im'], lb = flow)
            ref_im_lb2 = dict(im = ref_im_lb2['im'], lb = flow2)
            im_lb, ref_im_lb, ref_im_lb2 = self.trans_train_homo(im_lb, ref_im_lb, ref_im_lb2)

            # print(im_lb, ref_im_lb)
            img, label = im_lb['im'], im_lb['lb']
            ref_img, flow_map = ref_im_lb['im'], ref_im_lb['lb']
            ref_img2, flow_map2 = ref_im_lb2['im'], ref_im_lb2['lb']

        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)
        # label = self.convert_labels(label)

        if self.load_pair:
            ref_img = self.to_tensor(ref_img)
            ref_img2 = self.to_tensor(ref_img2)
        
        if not self.load_pair:
            # print(data_path)
            existence_list = self.gen_label_existence(label)
            return img, label, existence_list
        else:
            existence_list = self.gen_label_existence(label)
            return img, label, existence_list, ref_img, flow_map, ref_img2, flow_map2

    def warp_img(self, ref_img, flow):

        max_abs = 150
        mat = flow

        mat = (mat + max_abs)/2/max_abs

        mat[mat<0] = 0
        mat[mat>1] = 1

        mat = mat*255

        cv2.imwrite(os.path.join("proxy_x.png"),mat[...,0].astype(np.uint8))
        cv2.imwrite(os.path.join("proxy_y.png"),mat[...,1].astype(np.uint8))   

        ref_img = np.array(ref_img)

        h, w = flow.shape[:2]
        # import pdb; pdb.set_trace()
        # flow = -flow
        flow = np.concatenate([
            (flow[:,:,0] + np.arange(w))[...,np.newaxis], 
            (flow[:,:,1] + np.arange(h)[:,np.newaxis])[...,np.newaxis]
            ],axis=2)
        # import pdb; pdb.set_trace()
        result_img = cv2.remap(ref_img, flow.astype(np.float32), None, cv2.INTER_LINEAR)

        return result_img

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
    
    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label
    

    def gen_label_existence(self, label):
        existence_list = torch.zeros((12,))
        label_unique = np.unique(label)

        for label in label_unique:
            if label == 255:
                continue
            existence_list[label] = 1
        
        return existence_list

    def find_start_point(self, file_list):
        base_name_list = [os.path.basename(x) for x in file_list]

        frame_idx_list = []
        seq_name_list = []
        for name in base_name_list:
            seq_name = name.split('_')[0]
            seq_name_list.append(seq_name)

            if seq_name == '0001TP' or seq_name == '0016E5':
                frame_idx = int(name.split('_')[-1][:-4])
            elif seq_name == '0006R0' or seq_name == 'Seq05VD':
                frame_idx = int(name.split('_')[-1][:-4][1:])
            else:
                print("Unknown Sequence name!")
                exit(1)

            frame_idx_list.append(frame_idx)
            
            # print(name, seq_name, frame_idx)
        
        frame_idx_list = np.array(frame_idx_list)
        frame_idx_gap = np.ediff1d(frame_idx_list)
        middle_frame_ind = [False] + [(x in self.data_frame_idx_gap) for x in frame_idx_gap]
        start_frame_ind = np.invert(middle_frame_ind)

        start_point_list = np.array(range(len(base_name_list)))[start_frame_ind]

        start_point_dict = {}
        start_point_idx = 0
        for idx in range(len(base_name_list)):
            if start_point_idx+1 < len(start_point_list) and idx >= start_point_list[start_point_idx+1]:
                start_point_idx +=1 

            this_start_point = start_point_list[start_point_idx]
            start_point_dict[idx] = this_start_point

        return start_point_dict

    def getDatasetFrameIdx(self, basename, seq_name):
        if seq_name == "0001TP" or seq_name == "0016E5":
            frame_idx = int(basename.split('_')[1][:-4])
        elif seq_name == '0006R0' or seq_name == "Seq05VD":
            frame_idx = int(basename.split('_')[1][1:-4])

        return frame_idx
    
    def getDecodedBaseName(self, frame_idx, seq_name):
        # if seq_name == "0001TP":
        #     basename = "_".join([seq_name, "%06d.png"%frame_idx])
        # elif seq_name == "0006R0" or seq_name == "Seq05VD":
        #     basename = "_".join([seq_name, "f%05d.png"%frame_idx])
        # elif seq_name == '0016E5':
        #     basename = "_".join([seq_name, "%05d.png"%frame_idx])
        
        basename = "_".join([seq_name, "%06d.png"%frame_idx])

        return basename




class CamVidWithFlowTest(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.
    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.
    """

    # Images extension
    img_extension = '.png'

    # Default encoding for pixel value, class name, and class color
    _cmap = {
        0: (128, 128, 128),    # sky
        1: (128, 0, 0),        # building
        2: (192, 192, 128),    # column_pole
        3: (128, 64, 128),     # road
        4: (0, 0, 192),        # sidewalk
        5: (128, 128, 0),      # Tree
        6: (192, 128, 128),    # SignSymbol
        7: (64, 64, 128),      # Fence
        8: (64, 0, 128),       # Car
        9: (64, 64, 0),        # Pedestrian
        10: (0, 128, 192),     # Bicyclist
        11: (0, 0, 0)}         # Void
    
    _color2label = dict(zip(_cmap.values(), _cmap.keys()))

    def __init__(self,
                 root_dir,
                 mode='train',
                 cropsize=(640, 480),
                 randomscale=(0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5),
                #  transform=None,
                #  label_transform=None,
                 loader=pil_loader_single,
                 load_pair=False,
                 ref_gap=5,
                 ref_path = '/mnt/nvme1n1/hyb/data/camvid-sequence/frames/',
                 flow_path = '/mnt/nvme1n1/hyb/data/camvid-sequence/MVmap/'):

        self.root_dir = root_dir
        assert mode in ('test')
        self.mode = mode
        print('self.mode', self.mode)
        # self.transform = transform
        # self.label_transform = label_transform
        self.loader = loader
        self.load_pair = load_pair
        self.ref_gap = ref_gap
        self.ref_path = ref_path
        self.flow_path = flow_path
        
        ## The frame idx gap in CamVid dataset
        self.data_frame_idx_gap = [2,30]

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.39068785, 0.40521392, 0.41434407), (0.29652068, 0.30514979, 0.30080369)),
            ])

        self.trans_train_color = pairColorJitter(
            brightness = 0.5,
            contrast = 0.5,
            saturation = 0.5
        )
        self.trans_train_homo = pairCompose([
            pairOFHorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            # pairOFRandomScale(randomscale),
            pairOFRandomScaleV2(randomscale),   
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            pairOFRandomCrop(cropsize)
            ])

        if self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            # print(os.path.join(root_dir, self.test_folder))
            self.test_data = get_files(
                os.path.join(root_dir),
                extension_filter=self.img_extension)

            # self.test_start_point = self.find_start_point(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset
        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.
        """
        if index < 0:
            return 0
        if self.mode.lower() == 'test':
            data_path = self.test_data[index]
           

        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img = self.loader(data_path)
        # print(data_path)
        # print(data_path)
        
        # print(index, ref_index)
        if self.load_pair:
            decoded_frame_idx = int(os.path.basename(data_path)[:-4])

            ref_decoded_frame_idx = (decoded_frame_idx//self.ref_gap)*self.ref_gap
            
            ref_data_path = os.path.join(self.ref_path, '%05d.png'%ref_decoded_frame_idx)

            # print(data_path, ref_data_path)

            ref_img = Image.open(ref_data_path)

            f = open(os.path.join(self.flow_path, os.path.basename(data_path)[:-4]+'.bin'))
            flow = np.fromfile(f,np.dtype(np.short)).reshape(720,960,2)/4
            f.close()

            # warped_img = self.warp_img(ref_img, flow)
            # # cv2.imwrite("ref_img.png", np.array(ref_img).astype(np.uint8)[...,::-1])
            # cv2.imwrite("./image_test_result/test-%03d-true_img.png"%index, np.array(img).astype(np.uint8)[...,::-1])
            # cv2.imwrite("./image_test_result/test-%03d-warpped_img.png"%index, np.array(warped_img).astype(np.uint8)[...,::-1])
            # # exit(0)

            flow_map = flow

        img = self.to_tensor(img)
        label = 0
        # label = self.convert_labels(label)

        if self.load_pair:
            ref_img = self.to_tensor(ref_img)
        
        if not self.load_pair:
            # print(data_path)
            existence_list = 0
            return img, label, existence_list
        else:
            existence_list = 0
            return img, label, existence_list, ref_img, flow_map

    def warp_img(self, ref_img, flow):

        max_abs = 150
        mat = flow

        mat = (mat + max_abs)/2/max_abs

        mat[mat<0] = 0
        mat[mat>1] = 1

        mat = mat*255

        cv2.imwrite(os.path.join("proxy_x.png"),mat[...,0].astype(np.uint8))
        cv2.imwrite(os.path.join("proxy_y.png"),mat[...,1].astype(np.uint8))   

        ref_img = np.array(ref_img)

        h, w = flow.shape[:2]
        # import pdb; pdb.set_trace()
        # flow = -flow
        flow = np.concatenate([
            (flow[:,:,0] + np.arange(w))[...,np.newaxis], 
            (flow[:,:,1] + np.arange(h)[:,np.newaxis])[...,np.newaxis]
            ],axis=2)
        # import pdb; pdb.set_trace()
        result_img = cv2.remap(ref_img, flow.astype(np.float32), None, cv2.INTER_LINEAR)

        return result_img

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
    
    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label
    

    def gen_label_existence(self, label):
        existence_list = torch.zeros((12,))
        label_unique = np.unique(label)

        for label in label_unique:
            if label == 255:
                continue
            existence_list[label] = 1
        
        return existence_list

    def find_start_point(self, file_list):
        base_name_list = [os.path.basename(x) for x in file_list]

        frame_idx_list = []
        seq_name_list = []
        for name in base_name_list:
            seq_name = name.split('_')[0]
            seq_name_list.append(seq_name)

            if seq_name == '0001TP' or seq_name == '0016E5':
                frame_idx = int(name.split('_')[-1][:-4])
            elif seq_name == '0006R0' or seq_name == 'Seq05VD':
                frame_idx = int(name.split('_')[-1][:-4][1:])
            else:
                print("Unknown Sequence name!")
                exit(1)

            frame_idx_list.append(frame_idx)
            
            # print(name, seq_name, frame_idx)
        
        frame_idx_list = np.array(frame_idx_list)
        frame_idx_gap = np.ediff1d(frame_idx_list)
        middle_frame_ind = [False] + [(x in self.data_frame_idx_gap) for x in frame_idx_gap]
        start_frame_ind = np.invert(middle_frame_ind)

        start_point_list = np.array(range(len(base_name_list)))[start_frame_ind]

        start_point_dict = {}
        start_point_idx = 0
        for idx in range(len(base_name_list)):
            if start_point_idx+1 < len(start_point_list) and idx >= start_point_list[start_point_idx+1]:
                start_point_idx +=1 

            this_start_point = start_point_list[start_point_idx]
            start_point_dict[idx] = this_start_point

        return start_point_dict

    def getDatasetFrameIdx(self, basename, seq_name):
        if seq_name == "0001TP" or seq_name == "0016E5":
            frame_idx = int(basename.split('_')[1][:-4])
        elif seq_name == '0006R0' or seq_name == "Seq05VD":
            frame_idx = int(basename.split('_')[1][1:-4])

        return frame_idx
    
    def getDecodedBaseName(self, frame_idx, seq_name):
        # if seq_name == "0001TP":
        #     basename = "_".join([seq_name, "%06d.png"%frame_idx])
        # elif seq_name == "0006R0" or seq_name == "Seq05VD":
        #     basename = "_".join([seq_name, "f%05d.png"%frame_idx])
        # elif seq_name == '0016E5':
        #     basename = "_".join([seq_name, "%05d.png"%frame_idx])
        
        basename = "_".join([seq_name, "%06d.png"%frame_idx])

        return basename



class CamVidwithCUmap(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.
    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.
    """
    # Training dataset root folders
    train_folder = 'train'
    train_lbl_folder = 'train_labels_with_ignored'

    # Validation dataset root folders
    val_folder = 'val'
    val_lbl_folder = 'val_labels_with_ignored'

    # Test dataset root folders
    test_folder = 'test'
    test_lbl_folder = 'test_labels_with_ignored'

    # Images extension
    img_extension = '.png'

    # CU map location
    CUmap_dir = 'CUmap'

    # Default encoding for pixel value, class name, and class color
    _cmap = {
        0: (128, 128, 128),    # sky
        1: (128, 0, 0),        # building
        2: (192, 192, 128),    # column_pole
        3: (128, 64, 128),     # road
        4: (0, 0, 192),        # sidewalk
        5: (128, 128, 0),      # Tree
        6: (192, 128, 128),    # SignSymbol
        7: (64, 64, 128),      # Fence
        8: (64, 0, 128),       # Car
        9: (64, 64, 0),        # Pedestrian
        10: (0, 128, 192),     # Bicyclist
        11: (0, 0, 0)}         # Void
    
    _color2label = dict(zip(_cmap.values(), _cmap.keys()))

    def __init__(self,
                 root_dir,
                 mode='train',
                 cropsize=(640, 480),
                 randomscale=(0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5),
                #  transform=None,
                #  label_transform=None,
                 loader=pil_loader,):
        self.root_dir = root_dir
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        # self.transform = transform
        # self.label_transform = label_transform
        self.loader = loader

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.39068785, 0.40521392, 0.41434407), (0.29652068, 0.30514979, 0.30080369)),
            ])
        
        self.CUmap_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.75, 0.25),
        ]) 

        self.color_trans = ColorJitter(
                            brightness = 0.5,
                            contrast = 0.5,
                            saturation = 0.5)

        self.homo_trans = Compose([
            HorizontalFlip(),
            RandomScale(randomscale),
            RandomCrop(cropsize)
        ])
        
        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = get_files(
                os.path.join(root_dir, self.train_folder),
                extension_filter=self.img_extension)

            self.train_labels = get_files(
                os.path.join(root_dir, self.train_lbl_folder),
                extension_filter=self.img_extension)

        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = get_files(
                os.path.join(root_dir, self.val_folder),
                extension_filter=self.img_extension)

            self.val_labels = get_files(
                os.path.join(root_dir, self.val_lbl_folder),
                extension_filter=self.img_extension)

        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.img_extension)

            self.test_labels = get_files(
                os.path.join(root_dir, self.test_lbl_folder),
                extension_filter=self.img_extension)

        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset
        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.
        """
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]
    

        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
    

        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]

        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
        
        img, label = self.loader(data_path, label_path)

        CUmap_sec = os.path.basename(data_path).split('_')[0]

        if CUmap_sec == '0016E5':
            # print(data_path)
            CUmap_name_list = os.path.basename(data_path).split('_')
            CUmap_name_list[1] = 'f'+ CUmap_name_list[1]
            CUmap_name = '_'.join(CUmap_name_list)
            CUmap_path = os.path.join(self.root_dir, self.CUmap_dir, CUmap_sec, CUmap_name)
        elif CUmap_sec == 'Seq05VD':
            # print(data_path)
            CUmap_name_list = os.path.basename(data_path).split('_')
            CUmap_name_list[1] = CUmap_name_list[1][1:]
            CUmap_name = '_'.join(CUmap_name_list)
            CUmap_path = os.path.join(self.root_dir, self.CUmap_dir, CUmap_sec, CUmap_name)
        else:
            CUmap_path = os.path.join(self.root_dir, self.CUmap_dir, CUmap_sec,os.path.basename(data_path))

        CUmap, _ = self.loader(CUmap_path, label_path)

        # import pdb; pdb.set_trace()
        # CUmap_array = np.array(CUmap)
        # # CUmap_array = (127 + (CUmap_array+1)/2*128).astype(np.uint8)
        # import cv2
        # cv2.imwrite("CUmap.png", CUmap_array)
        # cv2.imwrite("image.png", np.array(img))

        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            im_lb = self.color_trans(im_lb)
            img = im_lb['im']
            img.putalpha(CUmap)
            # import pdb; pdb.set_trace()
            im_lb = dict(im = img, lb = label)
            im_lb = self.homo_trans(im_lb)
            img, label = im_lb['im'], im_lb['lb']

            CUmap = img.split()[-1]
            # import pdb; pdb.set_trace()
            img = Image.merge('RGB', img.split()[:-1])
        
        # import pdb; pdb.set_trace()
        # CUmap_array = np.array(CUmap)
        # # CUmap_array = (127 + (CUmap_array+1)/2*128).astype(np.uint8)
        # import cv2
        # cv2.imwrite("CUmap.png", cv2.resize(CUmap_array,(288,216)))
        # cv2.imwrite("image.png", cv2.resize(np.array(img),(288,216)))


        CUmap = self.CUmap_to_tensor(CUmap)
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)
        # label = self.convert_labels(label)
        
        img = torch.cat([img,CUmap],dim=0)

        existence_list = self.gen_label_existence(label)
        return img, label, existence_list


    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
    
    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label
    

    def gen_label_existence(self, label):
        existence_list = torch.zeros((12,))
        label_unique = np.unique(label)

        for label in label_unique:
            if label == 255:
                continue
            existence_list[label] = 1
        
        return existence_list

    def find_start_point(self, file_list):
        base_name_list = [os.path.basename(x) for x in file_list]
        seq_name_list = [x.split('_')[0] for x in base_name_list]

        start_point_dict = {}

        for idx, seq_name in enumerate(seq_name_list):
            if not seq_name in start_point_dict.keys():
                start_point_dict[seq_name] = idx
        
        return start_point_dict


class CamVidwithCUmapSingleBranch(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.
    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.
    """
    # Training dataset root folders
    train_folder = 'train'
    train_lbl_folder = 'train_labels_with_ignored'

    # Validation dataset root folders
    val_folder = 'val'
    val_lbl_folder = 'val_labels_with_ignored'

    # Test dataset root folders
    test_folder = 'test'
    test_lbl_folder = 'test_labels_with_ignored'

    # Images extension
    img_extension = '.png'

    # CU map location
    CUmap_dir = 'CUmap'

    # Default encoding for pixel value, class name, and class color
    _cmap = {
        0: (128, 128, 128),    # sky
        1: (128, 0, 0),        # building
        2: (192, 192, 128),    # column_pole
        3: (128, 64, 128),     # road
        4: (0, 0, 192),        # sidewalk
        5: (128, 128, 0),      # Tree
        6: (192, 128, 128),    # SignSymbol
        7: (64, 64, 128),      # Fence
        8: (64, 0, 128),       # Car
        9: (64, 64, 0),        # Pedestrian
        10: (0, 128, 192),     # Bicyclist
        11: (0, 0, 0)}         # Void
    
    _color2label = dict(zip(_cmap.values(), _cmap.keys()))

    def __init__(self,
                 root_dir,
                 mode='train',
                 cropsize=(640, 480),
                 randomscale=(0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5),
                #  transform=None,
                #  label_transform=None,
                 loader=pil_loader,):
        self.root_dir = root_dir
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        # self.transform = transform
        # self.label_transform = label_transform
        self.loader = loader

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.39068785, 0.40521392, 0.41434407), (0.29652068, 0.30514979, 0.30080369)),
            ])
        
        self.CUmap_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.75, 0.25),
        ]) 

        self.color_trans = ColorJitter(
                            brightness = 0.5,
                            contrast = 0.5,
                            saturation = 0.5)

        self.homo_trans = Compose([
            HorizontalFlip(),
            RandomScale(randomscale),
            RandomCrop(cropsize)
        ])
        
        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = get_files(
                os.path.join(root_dir, self.train_folder),
                extension_filter=self.img_extension)

            self.train_labels = get_files(
                os.path.join(root_dir, self.train_lbl_folder),
                extension_filter=self.img_extension)

        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = get_files(
                os.path.join(root_dir, self.val_folder),
                extension_filter=self.img_extension)

            self.val_labels = get_files(
                os.path.join(root_dir, self.val_lbl_folder),
                extension_filter=self.img_extension)

        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.img_extension)

            self.test_labels = get_files(
                os.path.join(root_dir, self.test_lbl_folder),
                extension_filter=self.img_extension)

        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset
        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.
        """
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]
    

        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
    

        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]

        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
        
        img, label = self.loader(data_path, label_path)

        CUmap_sec = os.path.basename(data_path).split('_')[0]

        if CUmap_sec == '0016E5':
            # print(data_path)
            CUmap_name_list = os.path.basename(data_path).split('_')
            CUmap_name_list[1] = 'f'+ CUmap_name_list[1]
            CUmap_name = '_'.join(CUmap_name_list)
            CUmap_path = os.path.join(self.root_dir, self.CUmap_dir, CUmap_sec, CUmap_name)
        elif CUmap_sec == 'Seq05VD':
            # print(data_path)
            CUmap_name_list = os.path.basename(data_path).split('_')
            CUmap_name_list[1] = CUmap_name_list[1][1:]
            CUmap_name = '_'.join(CUmap_name_list)
            CUmap_path = os.path.join(self.root_dir, self.CUmap_dir, CUmap_sec, CUmap_name)
        else:
            CUmap_path = os.path.join(self.root_dir, self.CUmap_dir, CUmap_sec,os.path.basename(data_path))

        CUmap, _ = self.loader(CUmap_path, label_path)

        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            im_lb = self.color_trans(im_lb)
            img = im_lb['im']
            img.putalpha(CUmap)
            # import pdb; pdb.set_trace()
            im_lb = dict(im = img, lb = label)
            im_lb = self.homo_trans(im_lb)
            img, label = im_lb['im'], im_lb['lb']

            CUmap = img.split()[-1]
            # import pdb; pdb.set_trace()
            img = Image.merge('RGB', img.split()[:-1])
        
        CUmap = self.CUmap_to_tensor(CUmap)
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)
        # label = self.convert_labels(label)
        
        # img = torch.cat([img,CUmap],dim=0)

        existence_list = self.gen_label_existence(label)
        return img, CUmap, label, existence_list


    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
    
    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label
    

    def gen_label_existence(self, label):
        existence_list = torch.zeros((12,))
        label_unique = np.unique(label)

        for label in label_unique:
            if label == 255:
                continue
            existence_list[label] = 1
        
        return existence_list

    def find_start_point(self, file_list):
        base_name_list = [os.path.basename(x) for x in file_list]
        seq_name_list = [x.split('_')[0] for x in base_name_list]

        start_point_dict = {}

        for idx, seq_name in enumerate(seq_name_list):
            if not seq_name in start_point_dict.keys():
                start_point_dict[seq_name] = idx
        
        return start_point_dict



if __name__ == "__main__":
    dataset = CamVidWithFlow("./data/CamVid/",mode="train",load_pair=True, ref_gap=2)
    print(dataset[23])
    import pdb; pdb.set_trace()