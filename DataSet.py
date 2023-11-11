import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 自定义数据集类,目的是是使用DataLoader可以对数据进行批处理、打乱和并行加载
# 需要创建一个继承自 torch.utils.data.Dataset 的自定义数据集类。这个类应该至少包含两个方法：__len__ 返回数据集的大小，__getitem__ 返回给定索引的数据。

class Dataset(torch.utils.data.Dataset):
    """加载和处理训练集(含有blur和deblur),将图片转化为张量"""
    def __init__(self, img_list, crop_size=(256, 256)):
        """
        Args:
            img_list:  图像文件列表
            crop_size: 表示裁剪后的图像大小
        """
        super(type(self), self).__init__()
        self.img_list = img_list
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()

    def crop_resize_totensor(self, img, crop_location):
        """
        根据裁剪位置(crop_location)从原图中裁剪出三个尺寸不同的图像(256x256、128x128和64x64大小),并转化为张量
        Args:
            img:接收一张图像
            crop_location:裁剪位置作为参数
        Returns:
            将256x256、128x128和64x64大小的三张图像转换为张量形式
        """
        img256 = img.crop(crop_location)
        img128 = img256.resize((self.crop_size[0] // 2, self.crop_size[1] // 2), resample=Image.BILINEAR)
        img64 = img128.resize((self.crop_size[0] // 4, self.crop_size[1] // 4), resample=Image.BILINEAR)
        return self.to_tensor(img256), self.to_tensor(img128), self.to_tensor(img64)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        根据给定的训练索引idx获取对应的图像数据
        Args:
            idx:    给定的训练数据索引(可能是多个)
        Returns:
            batch:  键值对,将256x256、128x128和64x64大小的三张图像转换为批量的张量形式(张量大小在[-1,1]之间)
        """
        # 得到idx
        blurry_img_name = self.img_list[idx].split(' ')[-2]
        clear_img_name = self.img_list[idx].split(' ')[-1]
        blurry_img = Image.open(blurry_img_name)
        clear_img = Image.open(clear_img_name)
        # 断言判断模糊图像和清晰图像的size是否相同
        assert blurry_img.size == clear_img.size

        # 随机选择裁剪位置,裁剪出crop_size大小的图片
        # np.random.uniform是NumPy库中的一个随机数生成函数，用于从均匀分布中生成随机样本
        crop_left = int(np.floor(np.random.uniform(0, blurry_img.size[0] - self.crop_size[0] + 1)))
        crop_top = int(np.floor(np.random.uniform(0, blurry_img.size[1] - self.crop_size[1] + 1)))
        # 裁剪的位置(左侧,顶端,右端,底部)
        crop_location = (crop_left, crop_top, crop_left + self.crop_size[0], crop_top + self.crop_size[1])

        # 将裁剪好的图片下采样为256x256、128x128和64x64大小的三个张量
        img256, img128, img64 = self.crop_resize_totensor(blurry_img, crop_location)
        label256, label128, label64 = self.crop_resize_totensor(clear_img, crop_location)

        # 将3个size,模糊与清晰的六组图像数据放入字典batch中
        batch = {'img256': img256, 'img128': img128, 'img64': img64, 'label256': label256, 'label128': label128, 'label64': label64}
        for k in batch:
            batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]
        return batch


class TestDataset(torch.utils.data.Dataset):
    """加载测试集,将图片转化为张量"""
    def __init__(self, img_list):
        super(type(self), self).__init__()
        self.img_list = img_list
        self.to_tensor = transforms.ToTensor()

    def resize_totensor(self, img):
        img_size = img.size
        img256 = img
        img128 = img256.resize((img_size[0] // 2, img_size[1] // 2), resample=Image.BILINEAR)
        img64 = img128.resize((img_size[0] // 4, img_size[1] // 4), resample=Image.BILINEAR)
        return self.to_tensor(img256), self.to_tensor(img128), self.to_tensor(img64)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # filename processing
        blurry_img_name = self.img_list[idx].split(' ')[-2]
        clear_img_name = self.img_list[idx].split(' ')[-1]

        blurry_img = Image.open(blurry_img_name)
        clear_img = Image.open(clear_img_name)
        assert blurry_img.size == clear_img.size

        img256, img128, img64 = self.resize_totensor(blurry_img)
        label256 = self.to_tensor(clear_img)
        batch = {'img256': img256, 'img128': img128, 'img64': img64, 'label256': label256}
        for k in batch:
            batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]
        return batch