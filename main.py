import torch
from model.srnnet import SRNDeblurNet




x1 = torch.ones(size=(2, 3, 64, 64)).cuda()
x2 = torch.ones(size=(2, 3, 32, 32)).cuda()
x3 = torch.ones(size=(2, 3, 16, 16)).cuda()
net = SRNDeblurNet().cuda()
y = net(x1, x2, x3)
print(y)
