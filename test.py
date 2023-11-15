import torch
import numpy as np

import config
import os
from model.srnnet import SRNDeblurNet  # 导入基本网络
import torchvision.transforms as transforms
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


def get_test_list():
    input_dir = config.test['input_dir']
    input_name_list = []
    input_filepath_list = []
    for filename in os.listdir(input_dir):
        # 获取文件的完整路径
        filepath = os.path.join(input_dir, filename).replace("\\", "/")
        # 判断是否为文件
        if os.path.isfile(filepath):
            input_name_list.append(filename)
            input_filepath_list.append(filepath)
    return input_name_list, input_filepath_list


def to_tesnor_list(filepath):
    """读取filepath处的图片"""
    blurry_img = Image.open(filepath)
    size = blurry_img.size
    to_tesnor = transforms.ToTensor()

    # 原规模,0.5规模,0.25规模
    img1 = blurry_img
    img2 = img1.resize((size[0] // 2, size[1] // 2), resample=Image.BICUBIC)
    img3 = img2.resize((size[0] // 4, size[1] // 4), resample=Image.BICUBIC)
    # 转换为tensor,并增加第0维
    img1 = torch.unsqueeze(to_tesnor(img1), 0)
    img2 = torch.unsqueeze(to_tesnor(img2), 0)
    img3 = torch.unsqueeze(to_tesnor(img3), 0)
    # 组合成为batch字典输出
    batch = {'img256': img1, 'img128': img2, 'img64': img3}
    for k in batch:
        batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]
    return batch


if __name__ == "__main__":
    # 加载网络结构
    net = torch.nn.DataParallel(SRNDeblurNet(xavier_init_all=config.net['xavier_init_all'])).cuda()
    checkpoints = torch.load(config.test['model_params'])
    net.load_state_dict(checkpoints)

    # 加载模型参数
    input_name_list, input_filepath_list = get_test_list()
    to_tesnor_list(input_filepath_list[0])

    # 处理每张照片
    for index, input_filepath in enumerate(input_filepath_list):
        batch = to_tesnor_list(input_filepath)
        with torch.no_grad():
            db256, _, _ = net(batch['img256'], batch['img128'], batch['img64'])

            # 删去batchsize维度,限制到Imaes的Tensor(-1,1)
            db256 = torch.squeeze(db256, 0).clamp(-1, 1)
            db256 = (db256 + 1) / 2

            # 从tensor转换到image
            to_pil = transforms.ToPILImage()
            new_img = to_pil(db256)

            # 展示与保存
            # new_img.show()
            new_img.save(config.test['output_dir'] + "/" + config.test['output_prefix'] + input_name_list[index], "PNG")  # 保存为PNG格式
