import torch, os, glob
import torch.nn as nn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter as sw
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# from tensorboard import SummeryWriter s
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Nets(nn.Module):
    def __init__(self):
        super(Nets, self).__init__()
        self.Lin = nn.Sequential(
            nn.ReLU(),
            nn.Linear(11, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        x = self.Lin(x)
        return x


all_data = pd.read_csv(r"D:\Pytorch_project\Kaggle_competitions\Mysolutions\data\feature_output.csv")
label = pd.read_csv(r"D:\Pytorch_project\Kaggle_competitions\Mysolutions\data\label_output.csv")
train_num = label.shape[0]
test_num = all_data.shape[0] - train_num
all_data_tensor = torch.tensor(all_data.values, dtype=torch.float)
label_tensor = torch.tensor(label.values, dtype=torch.float)


class Tabular(Dataset):
    def __init__(self, root, resize, mode, data, label):
        super(Tabular, self).__init__()
        self.root = root
        self.resize = resize
        if mode == 'train':
            self.data = data[:int(0.6 * data.shape[0])]
            self.label = label[:int(0.6 * label.shape[0])]
        elif mode == 'val':
            self.data = data[int(0.6 * data.shape[0]):]
            self.label = label[int(0.6 * label.shape[0]):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # idx= 0~len(self.data)
        # tf = transforms.Compose([
        #     lambda x:Image.open(x).convert('RGB') ,# 'strig'=>image data
        #     transforms.Resize((self.resize,self.resize)),
        #     transforms.ToTensor()
        # ])
        # img = tf.(img)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        t_data, t_label = torch.tensor(self.data[idx]).to(device), torch.tensor(self.label[idx]).to(device)
        return t_data, t_label


def main():
    net = Nets().to(device)
    writer = sw("logs")
    # for file in glob.glob("./logs"):
    #   os.remove(file)
    print("loading data to device:", device)
    train_data = Tabular(root=None, resize=None, mode='train', data=all_data_tensor[:train_num], label=label_tensor)
    val_data = Tabular(root=None, resize=None, mode="val", data=all_data_tensor[:train_num], label=label_tensor)
    loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, shuffle=True)
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    print("start to train".center(50, '-'))
    # train the model
    epoch_num = 50

    epoch_train_loss = []
    epoch_val_loss = []
    best_loss = 20000000
    for epoch in range(epoch_num):
        print("training epoch:", epoch)
        train_loss = []

        for x, y in loader:
            # step+=1
            ot = net(x)
            loss = mse(ot, y)
            # writer.add_scalar(tag="train loss", scalar_value=loss.item(),global_step=step)
            # print("train loss", loss.item())
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ave_t_loss = sum(train_loss) / len(train_loss)
        print("epoch ave loss:", ave_t_loss)
        epoch_train_loss.append(ave_t_loss)

        valloss = []
        for x, y in val_loader:
            # step+=1
            valot = net(x)
            val_loss = mse(y, valot)
            valloss.append(val_loss.item())
        ave_v_loss = sum(valloss) / len(valloss)
        epoch_val_loss.append(ave_v_loss)
        print("val ave loss:", ave_v_loss)
        # writer.add_scalar("val loss", val_loss.item())
        if best_loss > ave_v_loss:
            torch.save(net, '../Output/model.pt')
    writer.close()

    plt.plot(np.linspace(0, epoch_num - 1, epoch_num), epoch_train_loss, label='train loss')
    plt.plot(np.linspace(0, epoch_num - 1, epoch_num), epoch_val_loss, label='val loss')
    plt.legend()
    plt.show()
    print("validation ave loss:", sum(valloss) / len(valloss))
    print("validation finished!".center(50, '-'))


if __name__ == '__main__':
    main()