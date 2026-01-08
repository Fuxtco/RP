# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

from PIL import ImageFilter #图像滤波
import numpy as np
import torchvision.datasets as datasets #包含ImageFolder数据集类
import torchvision.transforms as transforms

from PIL import Image

logger = getLogger()


def convert_16bit_tif_to_8bit_rgb(pil_img):
    """
    Convert 16-bit or 48-bit TIFF images to 8-bit RGB.
    """
    arr = np.array(pil_img)

    # 如果是16bit（48bit tiff 的 dtype 也是 uint16）
    if arr.dtype == np.uint16:
        arr = (arr / 256).astype(np.uint8)   # 16bit → 8bit

    # 如果是单通道灰度图，把它变成 3 通道
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    return Image.fromarray(arr)


class MultiCropDataset(datasets.ImageFolder):
    #继承，可读取图片文件夹
    def __init__(
        self,
        data_path,
        size_crops, #输出尺寸
        nmb_crops, #每种尺寸的个数 len=2
        min_scale_crops, #随机裁剪最小比例
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
    ):
        super(MultiCropDataset, self).__init__(data_path) #调用父类构造函数__init__(data_path)
        assert len(size_crops) == len(nmb_crops) #条件不满足便报错，保证对齐
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset] #如果要缩减数据集
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()] #颜色增强模块
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)): #第几种crop类别
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            ) #输出尺寸和随机裁剪大小

            # extend将transform加入列表
            # compose将变换组合为流水线
            trans.extend([transforms.Compose([
                
                transforms.Lambda(convert_16bit_tif_to_8bit_rgb),

                randomresizedcrop, #随即裁剪
                transforms.RandomHorizontalFlip(p=0.5), #翻转
                transforms.Compose(color_transform), #颜色扰动
                transforms.ToTensor(), #张量化
                transforms.Normalize(mean=mean, std=std)]) #标准化
            ] * nmb_crops[i]) #创建nmb个相同的transform
        self.trans = trans

    #list-like类
    def __getitem__(self, index):
        # 得到多视图输出

        path, _ = self.samples[index] #继承父类sample函数，不需要label
        image = self.loader(path)
        # lambda：匿名函数，将image输入transform
        # map：对iterable中所有元素运用func（map(func, iterable)）
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    # 让实例像函数一样可调用
    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.

    # 创建颜色抖动，随即改变brightness,contrast,saturation,hue
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    # 概率包裹执行
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    # 随机执行灰度图
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    # 输出一个trans
    return color_distort
