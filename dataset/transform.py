#!/usr/bin/python
# -*- encoding: utf-8 -*-


from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import random
random.seed(233)
import numpy as np
np.random.seed(233)
import cv2


class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im=im, lb=lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
                im = im.crop(crop),
                lb = lb.crop(crop)
                    )


class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = lb.transpose(Image.FLIP_LEFT_RIGHT),
                    )


class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales
        # print('scales: ', scales)

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        # scale = np.random.uniform(min(self.scales), max(self.scales))
        w, h = int(W * scale), int(H * scale)
        return dict(im = im.resize((w, h), Image.BILINEAR),
                    lb = lb.resize((w, h), Image.NEAREST),
                )


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(im = im,
                    lb = lb,
                )


class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W*ratio), int(H*ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb

class pairRandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb, ref_im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        ref_im = ref_im_lb['im']
        ref_lb = ref_im_lb['lb']

        if (W, H) == (w, h): 
            return dict(im=im, lb=lb), dict(im=ref_im, lb=ref_lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
            ref_im = ref_im.resize((w, h), Image.BILINEAR)
            ref_lb = ref_lb.resize((w, h), Image.NEAREST)

        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
                im = im.crop(crop),
                lb = lb.crop(crop)
                    ), \
               dict(
                im = ref_im.crop(crop),
                lb = ref_lb.crop(crop)
                    )


class pairHorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb, ref_im_lb):
        if random.random() > self.p:
            return im_lb, ref_im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']

            ref_im = ref_im_lb['im']
            ref_lb = ref_im_lb['lb']

            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = lb.transpose(Image.FLIP_LEFT_RIGHT),
                    ), \
                   dict(im = ref_im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = ref_lb.transpose(Image.FLIP_LEFT_RIGHT),
                    )


class pairRandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales
        # print('scales: ', scales)

    def __call__(self, im_lb, ref_im_lb):
        im = im_lb['im']
        lb = im_lb['lb']

        ref_im = ref_im_lb['im']
        ref_lb = ref_im_lb['lb']

        W, H = im.size
        scale = random.choice(self.scales)
        # scale = np.random.uniform(min(self.scales), max(self.scales))
        w, h = int(W * scale), int(H * scale)

        return dict(im = im.resize((w, h), Image.BILINEAR),
                    lb = lb.resize((w, h), Image.NEAREST),
                ), \
               dict(im = ref_im.resize((w, h), Image.BILINEAR),
                    lb = ref_lb.resize((w, h), Image.NEAREST),
                ),


class pairOFRandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb, ref_im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        ref_im = ref_im_lb['im']
        ref_lb = ref_im_lb['lb']

        if (W, H) == (w, h): 
            return dict(im=im, lb=lb), dict(im=ref_im, lb=ref_lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
            ref_im = ref_im.resize((w, h), Image.BILINEAR)

            ref_lb = np.concatenate([ref_lb, ref_lb[...,0:1]], axis=-1)
            ref_lb = cv2.resize(ref_lb,dsize=(w,h),interpolation=cv2.INTER_NEAREST)
            ref_lb = ref_lb[...,:2]

        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H

        ref_lb = ref_lb[crop[1]:crop[3], crop[0]:crop[2],:]
        return dict(
                im = im.crop(crop),
                lb = lb.crop(crop)
                    ), \
               dict(
                im = ref_im.crop(crop),
                lb = ref_lb
                    )

class pairOFHorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb, ref_im_lb):
        if random.random() > self.p:
            return im_lb, ref_im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']

            ref_im = ref_im_lb['im']
            ref_lb = ref_im_lb['lb']

            ref_lb[...,0] = -ref_lb[...,0]
            ref_lb = np.fliplr(ref_lb)

            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = lb.transpose(Image.FLIP_LEFT_RIGHT),
                    ), \
                   dict(im = ref_im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = ref_lb,
                    )


class pairOFRandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales
        # print('scales: ', scales)

    def __call__(self, im_lb, ref_im_lb):
        im = im_lb['im']
        lb = im_lb['lb']

        ref_im = ref_im_lb['im']
        ref_lb = ref_im_lb['lb']

        W, H = im.size
        scale = random.choice(self.scales)
        # scale = np.random.uniform(min(self.scales), max(self.scales))
        w, h = int(W * scale), int(H * scale)

        ref_lb = np.concatenate([ref_lb, ref_lb[...,0:1]], axis=-1)
        ref_lb = cv2.resize(ref_lb,dsize=(w,h),interpolation=cv2.INTER_NEAREST)
        ref_lb = ref_lb[...,:2]

        return dict(im = im.resize((w, h), Image.BILINEAR),
                    lb = lb.resize((w, h), Image.NEAREST),
                ), \
               dict(im = ref_im.resize((w, h), Image.BILINEAR),
                    lb = ref_lb,
                ),


class pairOFRandomScaleV2(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales
        # print('scales: ', scales)

    def __call__(self, im_lb, ref_im_lb):
        im = im_lb['im']
        lb = im_lb['lb']

        ref_im = ref_im_lb['im']
        ref_lb = ref_im_lb['lb']

        W, H = im.size
        scale = random.choice(self.scales)
        # scale = np.random.uniform(min(self.scales), max(self.scales))
        w, h = int(W * scale), int(H * scale)

        ref_lb = np.concatenate([ref_lb, ref_lb[...,0:1]], axis=-1)
        ref_lb = cv2.resize(ref_lb,dsize=(w,h),interpolation=cv2.INTER_NEAREST)
        ref_lb = ref_lb[...,:2]
        ref_lb = ref_lb * scale
        # print(scale, self.scales)

        return dict(im = im.resize((w, h), Image.BILINEAR),
                    lb = lb.resize((w, h), Image.NEAREST),
                ), \
               dict(im = ref_im.resize((w, h), Image.BILINEAR),
                    lb = ref_lb,
                ),




class pairColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb, ref_im_lb):
        im = im_lb['im']
        lb = im_lb['lb']

        ref_im = ref_im_lb['im']
        ref_lb = ref_im_lb['lb']

        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)

        ref_im = ImageEnhance.Brightness(ref_im).enhance(r_brightness)
        ref_im = ImageEnhance.Contrast(ref_im).enhance(r_contrast)
        ref_im = ImageEnhance.Color(ref_im).enhance(r_saturation)

        return dict(im = im,
                    lb = lb,
                ), \
               dict(im = ref_im,
                    lb = ref_lb,
                )

class pairCompose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb, ref_im_lb):
        for comp in self.do_list:
            im_lb, ref_im_lb = comp(im_lb, ref_im_lb)
        return im_lb, ref_im_lb



class tripleRandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb, ref_im_lb, ref_im_lb2):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        ref_im = ref_im_lb['im']
        ref_lb = ref_im_lb['lb']

        ref_im2 = ref_im_lb2['im']
        ref_lb2 = ref_im_lb2['lb']

        if (W, H) == (w, h): 
            return dict(im=im, lb=lb), dict(im=ref_im, lb=ref_lb), dict(im=ref_im2, lb=ref_lb2)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
            ref_im = ref_im.resize((w, h), Image.BILINEAR)
            ref_lb = ref_lb.resize((w, h), Image.NEAREST)

            ref_im2 = ref_im2.resize((w, h), Image.BILINEAR)
            ref_lb2 = ref_lb2.resize((w, h), Image.NEAREST)

        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
                im = im.crop(crop),
                lb = lb.crop(crop)
                    ), \
               dict(
                im = ref_im.crop(crop),
                lb = ref_lb.crop(crop)
                    ), \
               dict(
                im = ref_im2.crop(crop),
                lb = ref_lb2.crop(crop)
                    )


class tripleHorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb, ref_im_lb, ref_im_lb2):
        if random.random() > self.p:
            return im_lb, ref_im_lb, ref_im_lb2
        else:
            im = im_lb['im']
            lb = im_lb['lb']

            ref_im = ref_im_lb['im']
            ref_lb = ref_im_lb['lb']

            ref_im2 = ref_im_lb2['im']
            ref_lb2 = ref_im_lb2['lb']

            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = lb.transpose(Image.FLIP_LEFT_RIGHT),
                    ), \
                   dict(im = ref_im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = ref_lb.transpose(Image.FLIP_LEFT_RIGHT),
                    ), \
                   dict(im = ref_im2.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = ref_lb2.transpose(Image.FLIP_LEFT_RIGHT),
                    )


class tripleRandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales
        # print('scales: ', scales)

    def __call__(self, im_lb, ref_im_lb, ref_im_lb2):
        im = im_lb['im']
        lb = im_lb['lb']

        ref_im = ref_im_lb['im']
        ref_lb = ref_im_lb['lb']

        ref_im2 = ref_im_lb2['im']
        ref_lb2 = ref_im_lb2['lb']

        W, H = im.size
        scale = random.choice(self.scales)
        # scale = np.random.uniform(min(self.scales), max(self.scales))
        w, h = int(W * scale), int(H * scale)

        return dict(im = im.resize((w, h), Image.BILINEAR),
                    lb = lb.resize((w, h), Image.NEAREST),
                ), \
               dict(im = ref_im.resize((w, h), Image.BILINEAR),
                    lb = ref_lb.resize((w, h), Image.NEAREST),
                ), \
               dict(im = ref_im2.resize((w, h), Image.BILINEAR),
                    lb = ref_lb2.resize((w, h), Image.NEAREST),
                ),


class tripleOFRandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb, ref_im_lb, ref_im_lb2):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        ref_im = ref_im_lb['im']
        ref_lb = ref_im_lb['lb']

        ref_im2 = ref_im_lb2['im']
        ref_lb2 = ref_im_lb2['lb']

        if (W, H) == (w, h): 
            return dict(im=im, lb=lb), dict(im=ref_im, lb=ref_lb), dict(im=ref_im2, lb=ref_lb2)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
            ref_im = ref_im.resize((w, h), Image.BILINEAR)

            ref_lb = np.concatenate([ref_lb, ref_lb[...,0:1]], axis=-1)
            ref_lb = cv2.resize(ref_lb,dsize=(w,h),interpolation=cv2.INTER_NEAREST)
            ref_lb = ref_lb[...,:2]

            ref_im2 = ref_im2.resize((w, h), Image.BILINEAR)

            ref_lb2 = np.concatenate([ref_lb2, ref_lb2[...,0:1]], axis=-1)
            ref_lb2 = cv2.resize(ref_lb2,dsize=(w,h),interpolation=cv2.INTER_NEAREST)
            ref_lb2 = ref_lb2[...,:2]

        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H

        ref_lb = ref_lb[crop[1]:crop[3], crop[0]:crop[2],:]
        ref_lb2 = ref_lb2[crop[1]:crop[3], crop[0]:crop[2],:]

        return dict(
                im = im.crop(crop),
                lb = lb.crop(crop)
                    ), \
               dict(
                im = ref_im.crop(crop),
                lb = ref_lb
                    ), \
               dict(
                im = ref_im2.crop(crop),
                lb = ref_lb2
                    )

class tripleOFHorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb, ref_im_lb, ref_im_lb2):
        if random.random() > self.p:
            return im_lb, ref_im_lb, ref_im_lb2
        else:
            im = im_lb['im']
            lb = im_lb['lb']

            ref_im = ref_im_lb['im']
            ref_lb = ref_im_lb['lb']

            ref_lb[...,0] = -ref_lb[...,0]
            ref_lb = np.fliplr(ref_lb)

            ref_im2 = ref_im_lb2['im']
            ref_lb2 = ref_im_lb2['lb']

            ref_lb2[...,0] = -ref_lb2[...,0]
            ref_lb2 = np.fliplr(ref_lb2)

            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = lb.transpose(Image.FLIP_LEFT_RIGHT),
                    ), \
                   dict(im = ref_im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = ref_lb,
                    ), \
                   dict(im = ref_im2.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = ref_lb2,
                    )

class tripleOFRandomScaleV2(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales
        # print('scales: ', scales)

    def __call__(self, im_lb, ref_im_lb, ref_im_lb2):
        im = im_lb['im']
        lb = im_lb['lb']

        ref_im = ref_im_lb['im']
        ref_lb = ref_im_lb['lb']

        ref_im2 = ref_im_lb2['im']
        ref_lb2 = ref_im_lb2['lb']

        W, H = im.size
        scale = random.choice(self.scales)
        # scale = np.random.uniform(min(self.scales), max(self.scales))
        w, h = int(W * scale), int(H * scale)

        ref_lb = np.concatenate([ref_lb, ref_lb[...,0:1]], axis=-1)
        ref_lb = cv2.resize(ref_lb,dsize=(w,h),interpolation=cv2.INTER_NEAREST)
        ref_lb = ref_lb[...,:2]
        ref_lb = ref_lb * scale

        ref_lb2 = np.concatenate([ref_lb2, ref_lb2[...,0:1]], axis=-1)
        ref_lb2 = cv2.resize(ref_lb2,dsize=(w,h),interpolation=cv2.INTER_NEAREST)
        ref_lb2 = ref_lb2[...,:2]
        ref_lb2 = ref_lb2 * scale
        # print(scale, self.scales)

        return dict(im = im.resize((w, h), Image.BILINEAR),
                    lb = lb.resize((w, h), Image.NEAREST),
                ), \
               dict(im = ref_im.resize((w, h), Image.BILINEAR),
                    lb = ref_lb,
                ), \
               dict(im = ref_im2.resize((w, h), Image.BILINEAR),
                    lb = ref_lb2,
                ),




class tripleColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb, ref_im_lb, ref_im_lb2):
        im = im_lb['im']
        lb = im_lb['lb']

        ref_im = ref_im_lb['im']
        ref_lb = ref_im_lb['lb']

        ref_im2 = ref_im_lb2['im']
        ref_lb2 = ref_im_lb2['lb']

        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)

        ref_im = ImageEnhance.Brightness(ref_im).enhance(r_brightness)
        ref_im = ImageEnhance.Contrast(ref_im).enhance(r_contrast)
        ref_im = ImageEnhance.Color(ref_im).enhance(r_saturation)

        ref_im2 = ImageEnhance.Brightness(ref_im2).enhance(r_brightness)
        ref_im2 = ImageEnhance.Contrast(ref_im2).enhance(r_contrast)
        ref_im2 = ImageEnhance.Color(ref_im2).enhance(r_saturation)

        return dict(im = im,
                    lb = lb,
                ), \
               dict(im = ref_im,
                    lb = ref_lb,
                ), \
               dict(im = ref_im2,
                    lb = ref_lb2,
                )

class tripleCompose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb, ref_im_lb, ref_im_lb2):
        for comp in self.do_list:
            im_lb, ref_im_lb, ref_im_lb2 = comp(im_lb, ref_im_lb, ref_im_lb2)
        return im_lb, ref_im_lb, ref_im_lb2



if __name__ == '__main__':
    flip = HorizontalFlip(p = 1)
    crop = RandomCrop((321, 321))
    rscales = RandomScale((0.75, 1.0, 1.5, 1.75, 2.0))
    img = Image.open('data/img.jpg')
    lb = Image.open('data/label.png')
