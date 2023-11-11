# 用于控制是否在测试时使用
test_time = False

# 训练过程的一些配置参数
train = {}
train['train_img_list'] = './data/train_list'   # 训练图像列表文件的路径
train['val_img_list'] = './data/eval_list'      # 验证图像列表文件的路径。
train['batch_size'] = 4                         # 训练时的批量大小。
train['val_batch_size'] = 2                     # 验证时的批量大小。
train['num_epochs'] = 2000                      # 训练的总轮数。
train['log_epoch'] = 100                        # 每隔多少轮输出一次日志
train['optimizer'] = 'Adam'                     # 优化器的选择，这里是 'Adam'
train['learning_rate'] = 1e-4                   # 学习率的设置

# SGD优化器的参数设置
train['momentum'] = 0.9                         # 对于 SGD 优化器的动量参数
train['nesterov'] = True                        # 对于 SGD 优化器是否使用 Nesterov 动量


# 模型和日志保存,恢复的设置
train['save_dir'] = './checkpoints'                         # 保存模型和日志的子目录
train['model_params'] = './checkpoints/model_params.pth'    # 保存模型的参数
train['resume'] = './checkpoints/replicate'                 # 恢复训练的路径，即之前保存的模型参数的路径
train['resume_epoch'] = None                                # 恢复训练的轮数，None 表示恢复到最后一轮
train['resume_optimizer'] = './checkpoints/replicate'       # 恢复训练时的优化器状态的路径。

# net是关于神经网络的一些配置
net = {}
net['xavier_init_all'] = True                   # 是否使用 Xavier 初始化所有权重

# loss关于损失函数的一些配置
loss = {}
loss['weight_l2_reg'] = 0.0                     # 权重 L2 正则化的系数。