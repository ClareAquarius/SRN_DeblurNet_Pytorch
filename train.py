import torch
import numpy as np
import config                           # 导入参数的设置
from model.srnnet import SRNDeblurNet   # 导入基本网络
from DataSet import Dataset             # 引入数据集加载/处理方法
from tqdm import tqdm                   # 在命令行界面中显示进度条
from time import time
import sys

log10 = np.log(10)  # log10
MAX_DIFF = 2  # 图片可能的最大像素值(图片的张量值在[-1,1]之间),在峰值信噪比(PSNR)计算时需要


def compute_loss(db256, db128, db64, batch):
    """
    计算一个batch的损失函数
    Args:
        db256, db128, db64: 通过模型得到的256*256,128*128,64*64去模糊图像
        batch:              键值对,包含三个不同规模的批量训练图像的模糊和去模糊的张量
    Return:
        键值对:mse表示不同规模的均方误差,psnr表示在最大规模图片上的峰值信噪比(PSNR)

    """

    assert db256.shape[0] == batch['label256'].shape[0]

    loss = 0
    loss += mse(db256, batch['label256'])
    # 峰值信噪比(PSNR)
    psnr = 10 * torch.log(MAX_DIFF ** 2 / loss) / log10
    loss += mse(db128, batch['label128'])
    loss += mse(db64, batch['label64'])
    return {'mse': loss, 'psnr': psnr}


def backward(loss, optimizer):
    """
    Arg:
        loss:       键值对,损失函数计算值,包含mse(均方误差),psnr(最大规模图片上的峰值信噪比)
        optimizer:  优化器
    Return:

    """
    optimizer.zero_grad()  # 将优化器中的梯度清零
    loss['mse'].backward()  # 对 MSE 损失进行反向传播
    torch.nn.utils.clip_grad_norm_(net.module.convlstm.parameters(), 3)  # 对整个模型的梯度进行裁剪,这里的3是裁剪梯度的阈值
    optimizer.step()  # 更新模型的参数
    return


def set_learning_rate(optimizer, epoch):
    """
    使用了学习率衰减策略,根据epoch设置优化器optimizer的学习率,即周期越大,学习率越低
    Arg:
        optimizer: 优化器
        epoch:     当前训练的周期数
    """
    optimizer.param_groups[0]['lr'] = config.train['learning_rate'] * 0.3 ** (epoch // 500)


if __name__ == "__main__":
    # 1.读入数据集,并放入数据加载器中
    # 读入训练集和测试集数据,其中.read() 读取文件的全部内容为一个字符串,.strip() 用于去除字符串两端的空格和换行符等空白字符。
    train_img_list = open(config.train['train_img_list'], 'r').read().strip().split('\n')
    val_img_list = open(config.train['val_img_list'], 'r').read().strip().split('\n')
    # 将数据集放入DataLoader(数据加载器)中
    train_dataset = Dataset(train_img_list)
    val_dataset = Dataset(val_img_list)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train['batch_size'],
                                                   shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train['val_batch_size'],
                                                 shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    # 2.均方误差
    mse = torch.nn.MSELoss().cuda()

    # 3.网络
    net = torch.nn.DataParallel(SRNDeblurNet(xavier_init_all=config.net['xavier_init_all'])).cuda()
    # net = SRNDeblurNet(xavier_init_all = config.net['xavier_init_all']).cuda()

    # 4.优化器
    assert config.train['optimizer'] in ['Adam', 'SGD']
    if config.train['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config.train['learning_rate'],
                                     weight_decay=config.loss['weight_l2_reg'])
    if config.train['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.train['learning_rate'],
                                    weight_decay=config.loss['weight_l2_reg'], momentum=config.train['momentum'],
                                    nesterov=config.train['nesterov'])

    # 定义一些用于记录训练的参数
    train_loss_log_list = []        # 用于记录log记录时训练集的损失值
    val_loss_log_list = []          # 用于记录log记录时的验证集损失值
    first_val = True
    t = time()

    # 定义一些用于验证集上最优的参数和模型
    best_val_psnr = 0               # 记录目前为止在验证集上达到的最佳PSNR(峰值信噪比)值
    best_net = None                 # 验证过程中达到最佳 PSNR 时的模型
    best_optimizer = None           # 验证过程中达到最佳 PSNR 时的优化器

    for epoch in tqdm(range(config.train['num_epochs']), file=sys.stdout, desc=str(config.train['num_epochs'])+' epoches'):
        # 根据当前 epoch 设置学习率
        set_learning_rate(optimizer, epoch)

        # 训练
        for step, batch in enumerate(train_dataloader):
            # 这里的batch是键值对,但是值的第0维度是bathsize
            # 将batch数据移到GPU上,不需要计算梯度
            for k in batch:
                batch[k] = batch[k].cuda()
                batch[k].requires_grad = False
            # 得到网络预测结果
            db256, db128, db64 = net(batch['img256'], batch['img128'], batch['img64'])
            # 计算损失
            loss = compute_loss(db256, db128, db64, batch)

            # 反向传播和网络参数更新
            backward(loss, optimizer)

            # 将loss从gpu移动到cpu上
            for k in loss:
                loss[k] = float(loss[k].cpu().detach().numpy())
            # 记录训练的损失值
            train_loss_log_list.append({k: loss[k] for k in loss})


        # 验证(间隔log_epoch个周期验证一次)
        if first_val or epoch % config.train['log_epoch'] == config.train['log_epoch'] - 1:
            first_val = False
            # 验证时不需要记录梯度
            with torch.no_grad():
                for step, batch in enumerate(val_dataloader):
                    for k in batch:
                        batch[k] = batch[k].cuda()
                        batch[k].requires_grad = False
                    db256, db128, db64 = net(batch['img256'], batch['img128'], batch['img64'])
                    loss = compute_loss(db256, db128, db64, batch)
                    for k in loss:
                        loss[k] = float(loss[k].cpu().detach().numpy())
                    val_loss_log_list.append({k: loss[k] for k in loss})
                # 计算了训练损失(MSE和)的平均值
                train_loss_log_dict = {k: float(np.mean([dic[k] for dic in train_loss_log_list])) for k in
                                       train_loss_log_list[0]}
                val_loss_log_dict = {k: float(np.mean([dic[k] for dic in val_loss_log_list])) for k in
                                     val_loss_log_list[0]}


                # PSNR的值越大越好
                if best_val_psnr < val_loss_log_dict['psnr']:
                    best_val_psnr = val_loss_log_dict['psnr']   # 保存最优的PSNR值
                    best_net = net.state_dict()                 # 更新最优模型参数

                # 将训练集和测试集的损失列表清空
                train_loss_log_list.clear()
                val_loss_log_list.clear()

                tt = time()
                log_msg = ""
                log_msg += "epoch {} , {:.2f} imgs/s".format(epoch, (
                            config.train['log_epoch'] * len(train_dataloader) * config.train['batch_size'] + len(
                        val_dataloader) * config.train['val_batch_size']) / (tt - t))

                log_msg += " | train : "
                for idx, k_v in enumerate(train_loss_log_dict.items()):
                    k, v = k_v
                    if k == 'acc':
                        log_msg += "{} {:.3%} {}".format(k, v, ',')
                    else:
                        log_msg += "{} {:.5f} {}".format(k, v, ',')
                log_msg += "  | eval : "
                for idx, k_v in enumerate(val_loss_log_dict.items()):
                    k, v = k_v
                    if k == 'acc':
                        log_msg += "{} {:.3%} {}".format(k, v, ',')
                    else:
                        log_msg += "{} {:.5f} {}".format(k, v, ',' if idx < len(val_loss_log_list) - 1 else '')
                tqdm.write(log_msg, file=sys.stdout)
                sys.stdout.flush()
                t = time()


    print("最优PSNR为:", best_val_psnr)
    torch.save(best_net, config.train['model_params'])




    # for epoch in tqdm(range(last_epoch + 1, config.train['num_epochs']), file=sys.stdout):
    #     set_learning_rate(optimizer, epoch)
    #     tb.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch * len(train_dataloader), 'train')
    #     for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), file=sys.stdout,
    #                             desc='training'):
    #         t_list = []
    #         for k in batch:
    #             batch[k] = batch[k].cuda(async =  True)
    #             batch[k].requires_grad = False
    #         db256, db128, db64 = net(batch['img256'], batch['img128'], batch['img64'])
    #         loss = compute_loss(db256, db128, db64, batch)
    #
    #         backward(loss, optimizer)
    #
    #         for k in loss:
    #             loss[k] = float(loss[k].cpu().detach().numpy())
    #         train_loss_log_list.append({k: loss[k] for k in loss})
    #         for k, v in loss.items():
    #             tb.add_scalar(k, v, epoch * len(train_dataloader) + step, 'train')
    #
    #     # validate and log
    #     if first_val or epoch % config.train['log_epoch'] == config.train['log_epoch'] - 1:
    #         with torch.no_grad():
    #             first_val = False
    #             for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), file=sys.stdout,
    #                                     desc='validating'):
    #                 for k in batch:
    #                     batch[k] = batch[k].cuda(async =  True)
    #                     batch[k].requires_grad = False
    #                 db256, db128, db64 = net(batch['img256'], batch['img128'], batch['img64'])
    #                 loss = compute_loss(db256, db128, db64, batch)
    #                 for k in loss:
    #                     loss[k] = float(loss[k].cpu().detach().numpy())
    #                 val_loss_log_list.append({k: loss[k] for k in loss})
    #
    #             train_loss_log_dict = {k: float(np.mean([dic[k] for dic in train_loss_log_list])) for k in
    #                                    train_loss_log_list[0]}
    #             val_loss_log_dict = {k: float(np.mean([dic[k] for dic in val_loss_log_list])) for k in
    #                                  val_loss_log_list[0]}
    #             for k, v in val_loss_log_dict.items():
    #                 tb.add_scalar(k, v, (epoch + 1) * len(train_dataloader), 'eval')
    #             if best_val_psnr < val_loss_log_dict['psnr']:
    #                 best_val_psnr = val_loss_log_dict['psnr']
    #                 save_model(net, tb.path, epoch)
    #                 save_optimizer(optimizer, net, tb.path, epoch)
    #
    #             train_loss_log_list.clear()
    #             val_loss_log_list.clear()
    #
    #             tt = time()
    #             log_msg = ""
    #             log_msg += "epoch {} , {:.2f} imgs/s".format(epoch, (
    #                         config.train['log_epoch'] * len(train_dataloader) * config.train['batch_size'] + len(
    #                     val_dataloader) * config.train['val_batch_size']) / (tt - t))
    #
    #             log_msg += " | train : "
    #             for idx, k_v in enumerate(train_loss_log_dict.items()):
    #                 k, v = k_v
    #                 if k == 'acc':
    #                     log_msg += "{} {:.3%} {}".format(k, v, ',')
    #                 else:
    #                     log_msg += "{} {:.5f} {}".format(k, v, ',')
    #             log_msg += "  | eval : "
    #             for idx, k_v in enumerate(val_loss_log_dict.items()):
    #                 k, v = k_v
    #                 if k == 'acc':
    #                     log_msg += "{} {:.3%} {}".format(k, v, ',')
    #                 else:
    #                     log_msg += "{} {:.5f} {}".format(k, v, ',' if idx < len(val_loss_log_list) - 1 else '')
    #             tqdm.write(log_msg, file=sys.stdout)
    #             sys.stdout.flush()
    #             log_file.write(log_msg + '\n')
    #             log_file.flush()
    #             t = time()
    #             # print( torch.max( predicts , 1  )[1][:5] )
    #
    #         # train_loss_epoch_list = []
