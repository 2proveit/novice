import torch,os,math
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.Convd = nn.Sequential(
            nn.Conv2d(3,6,3,1,1), # [64,128,128]
            nn.ReLU(),
            nn.MaxPool2d(3,1,1), # [64,128,128]

            nn.Conv2d(6,8,3,1),# [32,126,126]
            nn.ReLU(),
            nn.MaxPool2d(3,2), # [32,62,62]

            nn.Conv2d(8,16,3,1), # [16,60,60]
            nn.ReLU(),
            nn.MaxPool2d(3,1),

            nn.Conv2d(16,32,3,2),
            nn.ReLU(),
            nn.MaxPool2d(3,1), # [32,26,26]

            nn.Conv2d(32,64,3,2),
            nn.ReLU(),
            nn.AvgPool2d(3,1), # [64,10,10]

            nn.Conv2d(64,64,3,2),# [64,4,4]
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(1,-1), # (start_dim end_dim)

            nn.Linear(64*4*4,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU(),
            nn.Linear(4,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x= self.Convd(x)
        x= self.fc(x)
        return x


class Insects(Dataset):
    def __init__(self,mode,transform=None):
        super(Insects, self).__init__()
        self.root = r"D:\Pytorch_project\Kaggle_competitions\Data\hymenoptera_data\hymenoptera_data\\"+mode
        self.mode = mode
        self.transform = transform
        self.label = ['ants','bees']
        self.images = os.listdir(os.path.join(self.root,self.label[0]))+os.listdir(os.path.join(self.root,self.label[1]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        lenth = len(os.listdir(os.path.join(self.root,self.label[0])))
        if item <= lenth-1:
            img = Image.open(os.path.join(self.root,self.label[0],self.images[item]))
            return self.transform(img),0
        else:
            img = Image.open(os.path.join(self.root,self.label[1],self.images[item]))
            return self.transform(img),1

def main():

    transformer = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    for file in os.listdir('logs'):
        os.remove(os.path.join("logs",file))
    configs = {
        "n_epoch":500,
        "lr":0.1,
        "device": "cuda" if torch.cuda.is_available() else 'cpu',
        "creterion":nn.MSELoss(),
        "model_save_path":"D:\\Pytorch_project\\Kaggle_competitions\\Output\\"
    }
    print("using device:", configs["device"])
    writer = SummaryWriter(log_dir="logs")
    train_data = Insects(mode='train',transform=transformer)
    train_dataloader = DataLoader(train_data,batch_size=4)

    val_data = Insects(mode='val', transform=transformer)
    val_dataloader = DataLoader(val_data,batch_size=4)

    net = MyNet().to(configs["device"])
    best_score = math.inf
    optimizer = torch.optim.Adam(net.parameters(),lr=configs["lr"])
    creterion = configs['creterion'].to(configs["device"])
    for epoch in range(configs['n_epoch']):
            #training
        print(f"epoch {epoch} start!".center(30,'-'))
        t_loss = []
        for X , y in tqdm(train_dataloader):
            X = X.to(configs["device"])
            y = y.to(configs["device"])
            y = y.to(torch.float)
            output = net(X)
            #print("output",output.shape)
            y=y.view(4,1)
            #print("y",y.shape)
            loss = creterion(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss.append(loss.item())
        ave_t_loss = sum(t_loss)/len(t_loss)
        writer.add_scalar("train",ave_t_loss,epoch)
        print("training loss {:.3f}".format(ave_t_loss))

        v_loss = []
        for X, val_y in tqdm(val_dataloader):
            X = X.to(configs["device"])
            val_y = val_y.to(configs["device"])
            val_y = val_y.to(torch.float)
            output = net(X)

            val_y=torch.unsqueeze(val_y,1)
            #print('val_y', val_y.shape)
            #print("val output", output.shape)
            loss = creterion(output, val_y)
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            v_loss.append(loss.item())
        ave_v_loss = sum(v_loss)/len(v_loss)
        writer.add_scalar("val",ave_v_loss,epoch)
        print("test loss {:.3f}".format(ave_v_loss))

        if ave_v_loss < best_score:
            best_score = ave_v_loss
            torch.save(net,configs["model_save_path"]+"ant_bee_best_score{:.3f}.pt".format(best_score))
    writer.close()
if __name__ == "__main__":
    main()
