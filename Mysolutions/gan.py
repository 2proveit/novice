import numpy as np
import torch, numpy, torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])

train_ds = torchvision.datasets.MNIST('data',train=True, transform = transform, download = True)
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100,156),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,28*28),
            nn.Tanh()
        )

    def forward(self,x): # x represents noise input
        img = self.main(x)
        img = img.view(-1,28,28,1)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.mian = nn.Sequential(
            nn.Linear(28*28,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = x.view(28*28)
        x = self.main(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator().to(device)
dis = Discriminator().to(device)


d_optim = optim.Adam(dis.parameters(),lr =0.01)
g_optim = optim.Adam(gen.parameters(),lr = 0.01)
loss_fnction = nn.BCELoss()


def gen_img_plot(model, epoch, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig  = plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow((prediction[i]+1)/2)
        plt.axes('off')
    plt.show()


test_imput = torch.randn(16,100,device = device)

D_loss = []
G_loss = []
epoch =30

for epoch in range(epoch):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader) # len(dataloader) 返回批次数，len(dataset)返回样本数
    for step , (img,_) in enumerate(dataloader):
        img = img.to(device)
        size = img.size(0)
        random_noise = torch.randn(size,100,device=device)

        d_optim.zero_grad()
        real_output = dis(img)          #判别器输入真实图片的预测结果，希望结果为‘1’
        d_real_loss = loss_fnction(real_output, torch.ones_like(real_output))# 判别器在真实图像上的损失
        d_real_loss.backward()

        gen_img = gen(random_noise)# 得到一张生成的图片
        fake_output = dis(gen_img.detach())      # 判别器输入生成图片的预测结果，希望结果为‘0’
        d_fake_loss = loss_fnction(fake_output, torch.zeros_like(fake_output))
        d_fake_loss.backward()
        d_loss = d_real_loss + d_fake_loss
        d_optim.step()

        g_optim.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss_fnction(fake_output,torch.ones_like(fake_output)) # 生成器生成的图像在判别器上的损失
        g_loss.backward()
        g_optim.step()

    with torch.no_grad():
        d_epoch_loss += d_loss
        g_epoch_loss += g_loss
    with torch.no_grad():
        d_epoch_loss/=count
        g_epoch_loss/=count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print("epoch:{}".format(epoch))
        gen_img_plot(gen,test_input=test_imput)