import os
# 读取数据集的目录信息,生成数据的组织文件

# 训练集
prefix_dir = './DataSet'                   # 文件路径前缀
blur_dir = 'train/blur_lin'           # 模糊的图像数据集的路径
deblur_dir = 'train/deblurred_lin'    # 去模糊的图像数据集的路径
txt_name = './train_list'               # 保存的名称

# 评估集
# prefix_dir = './DataSet'                   # 文件路径前缀
# blur_dir = 'eval/blur'                  # 模糊的图像数据集的路径
# deblur_dir = 'eval/deblur'              # 去模糊的图像数据集的路径
# txt_name = './eval_list'                # 保存的名称


blur_img_list =[]
deblur_img_list = []
# 遍历模糊目录下的图像
for filename in os.listdir(blur_dir):
    # 获取文件的完整路径
    filepath = os.path.join(blur_dir, filename)
    # 判断是否为文件
    if os.path.isfile(filepath):
        blur_img_list.append(os.path.join(prefix_dir, filepath).replace("\\", "/"))
# 遍历去模糊目录下的图像
for filename in os.listdir(deblur_dir):
    # 获取文件的完整路径
    filepath = os.path.join(deblur_dir, filename)
    # 判断是否为文件
    if os.path.isfile(filepath):
        deblur_img_list.append(os.path.join(prefix_dir, filepath).replace("\\", "/"))

# 先检查数据数量上是否有问题
assert len(blur_img_list)==len(deblur_img_list), "用作训练集的模糊图像和去模糊图像应该一一匹配"

# 将数据组织成为一个list,使用文件保存
with open(txt_name, 'w') as file:
    for a,b in zip(blur_img_list,deblur_img_list):
        img_pairs = "{} {}\n".format(a, b)        # 一行数据表示一对模糊\去模糊的数据
        file.write(img_pairs)
file.close()
