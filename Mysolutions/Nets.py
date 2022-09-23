import torch
import torch.nn as nn
import pandas as pd
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
            nn.Linear(11, 1)
        )

    def forward(self, x):
        x = self.Lin(x)
        return x


all_data = pd.read_csv(r"./data/feature_output.csv")
label = pd.read_csv(r"./data/feature_output.csv")
train_num = label.shape[0]
test_num = all_data.shape[0]-train_num
all_data_tensor = torch.tensor(all_data.values, dtype=torch.float)
label_tensor = torch.tensor(label.values, dtype=torch.float)


class Tabular(Dataset):
    def __init__(self, root, resize, mode, data, label):
        super(Tabular, self).__init__()
        self.root = root
        self.resize = resize
        if mode == 'train':
            self.data = data[:int(0.6*data.shape[0])]
            self.label = label[:int(0.6*label.shape[0])]
        elif mode == 'val':
            self.data = data[int(0.6*data.shape[0]):]
            self.label = label[int(0.6*label.shape[0])]

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
    print("loading data to device:", device)
    train_data = Tabular(root=None, resize=None, mode='train', data=all_data_tensor[:train_num], label=label_tensor)
    val_data = Tabular(root=None, resize=None, mode="val", data=all_data_tensor[:train_num], label=label_tensor)
    loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True,num_workers=8)
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    print("start to train\n".center(50, '-'))
    # train the model
    step =0
    for epoch in range(50):
        print("training epoch:", epoch)

        for x, y in loader:
            step+=1
            ot = net(x)
            loss = mse(ot, y)
            writer.add_scalar(tag="train loss", scalar_value=loss.item(),global_step=step)
            print("train loss", loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("training finish!\n".center(50, '-'))
    print("start to validation\n".center(50, '-'))
    step=0
    for x, y in val_loader:
        step+=1
        valot = net(x)
        val_loss = mse(y, valot)
        print("val loss:", val_loss.item(),step)
        writer.add_scalar("val loss", val_loss.item())
    writer.close()
    print("validation finished!\n".center(50, '-'))


if __name__ == '__main__':
    main()