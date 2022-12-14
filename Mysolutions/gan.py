import os.path

import numpy as np
import torch, numpy, torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

train_ds = torchvision.datasets.MNIST('data',train=True, transform = transform, download = True)
dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,28*28),
            nn.Tanh()
        )

    def forward(self,x): # x represents noise input
        img = self.main(x)
        img = img.view(-1,28,28,1)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28,512),
            #nn.Dropout(0.2),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(512,256),
            nn.ReLU(),
            #nn.Linear(256,128),
            #nn.ReLU(),
            #nn.Linear(128,64),
            #n.ReLU(),
            nn.Linear(256,1),
            #nn.ReLU(),
            nn.Sigmoid(),
        )

    def forward(self,x):
        x = x.view(-1,28*28)
        x = self.main(x)
        #print(x.shape)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator().to(device)
dis = Discriminator().to(device)


d_optim = optim.SGD(dis.parameters(), lr=0.01)
g_optim = optim.SGD(gen.parameters(), lr=0.01)
loss_fnction = nn.BCELoss()


def gen_img_plot(model, epoch, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig  = plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow((prediction[i]+1)/2)
        #plt.axes('off')
    plt.savefig(os.path.join("D:\Pytorch_project\Kaggle_competitions\Output",f'gan_mnist_epoch{epoch}_.png'))
    #plt.show()


test_imput = torch.randn(16,100,device = device)

D_loss = []
G_loss = []
epoch =30

for epoch in range(epoch):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader) # len(dataloader) ??????????????????len(dataset)???????????????
    for step , (img,_) in enumerate(dataloader):
        img = img.to(device)
        size = img.size(0)
        random_noise = torch.randn(size, 100, device=device)

        d_optim.zero_grad()
        real_output = dis(img)          #???????????????????????????????????????????????????????????????1???
        d_real_loss = loss_fnction(real_output, torch.ones_like(real_output))# ????????????????????????????????????
        d_real_loss.backward()

        gen_img = gen(random_noise)# ????????????noise??????????????? # gen fake img
        fake_output = dis(gen_img.detach())      # ???????????????????????????????????????????????????????????????0???
        d_fake_loss = loss_fnction(fake_output, torch.zeros_like(fake_output))
        d_fake_loss.backward()
        d_loss = d_real_loss + d_fake_loss
        d_optim.step()

        g_optim.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss_fnction(fake_output,torch.ones_like(fake_output)) # ????????????????????????????????????????????????
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
        print("dis loss:",d_epoch_loss.item())
        print("gen loss:",g_epoch_loss.item())
        gen_img_plot(gen,epoch,test_input=test_imput)
if __name__ == '__main__':
    pass