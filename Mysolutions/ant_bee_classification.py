import torch,os
import glob ,tqdm
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms

transformer = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

class Insects(Dataset):
    def __init__(self,mode,root=r"D:\Pytorch_project\Kaggle_competitions\Data\hymenoptera_data\hymenoptera_data"):
        super(Insects, self).__init__()
        self.root = root
        self.mode = mode
        self.images = os.open(self.root,1)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item):
        return