import torch
import torch.nn as nn
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
for i in all_data:
    print(i)



def main():
    net = Nets().to(device)
    input = torch.randn(1, 11).to(device)
    mse = nn.MSELoss()
    ot = net(input)
    print("ot", ot)

    # train the model
    for epoch in range(5):
        for x,y in enumerate(all_data):
            ot = net(x)
            loss = mse(ot,y)
            print("train loss",loss)


if __name__ == '__main__':
    main()