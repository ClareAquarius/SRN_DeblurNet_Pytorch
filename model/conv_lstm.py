import torch.nn as nn
import torch


class CLSTM_cell(nn.Module):
    """定义单个Conv LSTM单元.
    Args:
      input_chans:  int 输入的通道数
      num_features: int 状态的通道数(表示c,w隐藏状态等的通道数)
      filter_size:  int 过滤器(卷积核)的高度和宽度

    """

    # input_chans是输入通道数,num_features是特征数
    def __init__(self, input_chans, num_features, filter_size):
        super(CLSTM_cell, self).__init__()
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        # 计算需要多大的padding才能保证输入输出图像不变
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)

    # foward需要传入input和hidden_state
    def forward(self, input, hidden_state):
        """
        Args:
            input:          shape:(B,C,H,W),其中C是input_chans
            hidden_state:   shape:(Batch, Chans, H, W),h,c两个具有是num_features通道,相同与图像相同形状的隐状态
        Returns:
            next_h, next_c: shape:(Batch, Chans, H, W),循环一次后下一个(h,c隐状态)
        """
        hidden, c = hidden_state
        # 将输入i和隐状态h 在通道维度上拼接,形状为(b,c,h,w)
        combined = torch.cat((input, hidden), 1)
        A = self.conv(combined)
        # 分别得到输入门(决定何时将数据读入单元),遗忘门(用来重置单元的内容),输出门(用来从单元中输出条目),候选记忆单元
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        # 计算下一个记忆元c
        next_c = f * c + i * g
        # 计算下一个隐状态h
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size, shape):
        # 用于初始化单个Conv LSTM单元的h和c隐状态
        return (torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda(),
                torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda())



# 下面这个包装多层的CLSTM模块,但srn没有采用
class CLSTM(nn.Module):
    """定义Conv LSTM层(含有num_layers个ConvLSTM cell基本单元)
        其中第一层输入通道为input_chans,剩下层输入通道为num_features(不算h的输入通道)
    Args:
      input_chans:  int 输入通道数
      num_features: int 状态的通道数(h,c等隐状态的通道数)
      filter_size:  int 过滤器(卷积核)的高度和宽度
      num_layers:   int ConvCLSTM的层数
    """

    def __init__(self, input_chans, num_features, filter_size, num_layers=1):
        super(CLSTM, self).__init__()

        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.num_layers = num_layers
        cell_list = []
        # 第一个conv CLSTM层有不同的输入通道数(输入通道数是input_chans)
        cell_list.append(CLSTM_cell(self.input_chans, self.filter_size, self.num_features).cuda())
        # 之后的conv CLSTM层(输入通道数是num_features)
        for _ in range(1, self.num_layers):
            cell_list.append(CLSTM_cell(self.num_features, self.filter_size, self.num_features).cuda())
        # 将ConvLSTM单元作为模型的子模块进行管理。
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input, hidden_state):
        """
        Args:
            hidden_state:   元组的列表,num_layers个隐状态(h,c)组成的列表, 第i元组应该为第i层的h和c隐状态
            input:          shape:(Batch, seq_len, Chans, H, W)
        Returns:
            next_hidden:    num_layers个隐状态(h,c)组成的列表,取得是每层最后一个时间步的隐藏状态(h,c)
            current_input:  shape:(Batch, seq_len, Chans, H, W),最后的conv CLSTM层中共seq_len个时间步输出的h隐状态
        """
        current_input = input.transpose(0, 1)  # transpose进行维度转置操作,将时间序列数据的维度放在第一个位置,得到(seq_len, B, C, H, W)
        next_hidden = []  # num_layers个隐状态(h,c)组成的列表
        seq_len = current_input.size(0)  # 得到时间步长

        # 先遍历num_layers个conv CLSTM层,然后对每一层conv CLSTM层执行seq_len步循环操作
        for id_layer in range(self.num_layers):
            hidden_c = hidden_state[id_layer]  # 得到第id_layer层的隐藏状态(h,c)
            output_inner = []

            # 对第id_layer个conv CLSTM层,执行seq_len个时间步循环
            for t in range(seq_len):
                hidden_c = self.cell_list[id_layer](current_input[t], hidden_c)
                output_inner.append(hidden_c[0])  # 只添加h隐藏状态

            next_hidden.append(hidden_c)  # 添加第i层conv CLSTM层的隐藏状态,取得是每层最后一个时间步的隐藏状态
            # 当 * 操作符用于函数调用时，它可以将一个可迭代对象（如列表、元组）解包成一个个独立的参数
            current_input = torch.cat(output_inner, 0).view(
                current_input.size(0), *output_inner[0].size())  # 最后一层conv CLSTM层共seq_len个时间步输出的h隐状态
        return next_hidden, current_input

    # 用于初始化Conv LSTM层的隐藏状态
    def init_hidden(self, batch_size, shape):
        init_states = []  # 这是元组的列表
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, shape))
        return init_states
