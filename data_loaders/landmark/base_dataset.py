from torch.utils.data import Dataset
from PIL import Image
import os.path as osp
class TrainDataset(Dataset):
    def __int__(self,flist,transform):
        self.imlist = flist
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.imlist[index]
        img = self._reader(path)
        img = self.transform(img)
        return img,label
    def __len__(self):
        return len(self.imlist)

    def _reader(self,path):
        return Image.open(path)

class TestDataset(Dataset):
    def __int__(self,dataset_name,root,transform,query_flag=True):
        self.imlist = []
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.imlist[index]
        img = self._reader(path)
        img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imlist)

    def _reader(self,path):
        return Image.open(path)
