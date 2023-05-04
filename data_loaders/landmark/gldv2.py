from data_loaders.landmark.base_dataset import TrainDataset,TestDataset
import os.path as osp
import pickle
from typing import Sequence
def files_reader(path,root):
    flist = []
    with open(path) as f:
        for line in f.readlines():
            [imid,label]=line.split()
            flist.append([osp.join(root,imid),int(label)])
    return flist

class Gldv2TrainDataset1():
    def __init__(self,root,flist):
        self.root = root
        if type(flist) is str:
            self.imlist = files_reader(flist,root)
        elif isinstance(flist,Sequence):
            self.imlist = flist

class Gldv2TrainDataset(TrainDataset):
    def __init__(self,root,flist,transform):
        self.root = root
        if type(flist) is str:
            self.imlist = files_reader(flist,root)
        elif isinstance(flist,Sequence):
            self.imlist = flist
        else:
            raise BaseException('Gldv2TrainDataset: flist must be a txt file or list!')
        self.transform = transform

class Gldv2TestDataset(TestDataset):
    def __init__(self,dataset_name,root,transform,query_flag=True):
        if query_flag:
            self.file = osp.join(root,"gldv2_private_query_list.txt")
        else:
            self.file = osp.join(root, "gldv2_gallery_list.txt")
        self.imlist = files_reader(self.file,root)
        self.query_gts_list = osp.join(root, "gldv2_private_query_gt.txt")
        self.query_flag = query_flag
        self.query_gts = [[], [], []]
        self.transform = transform
        if query_flag:
            # [img_name: str, img_index: int, gts: int list]
            with open(self.query_gts_list, 'r') as f:
                for line in f.readlines():
                    img_name, img_index, tmp_gts = line.split(" ")
                    gts = [int(i) for i in tmp_gts.split(",")]
                    self.query_gts[0].append(img_name)
                    self.query_gts[1].append(int(img_index))
                    self.query_gts[2].append(gts)


class ROxfordParisTestDataset(TestDataset):
    def __init__(self,dataset_name,root,transform,query_flag=True):
        self.root = root
        dataset_name = dataset_name.lower()
        self.query_flag = query_flag
        gnd_fname = osp.join(self.root,'gnd_{}.pkl'.format(dataset_name))
        if dataset_name not in ['roxford5k','rparis6k','revisitop1m']:
            raise Exception('ROxfordParisTestDataset: only support roxford5k, rparis6k and revisitop1m')
        if dataset_name in ['roxford5k','rparis6k']:
            with open(gnd_fname,'rb') as f:
                cfg = pickle.load(f)
            cfg['gnd_fname'] = gnd_fname
            cfg['ext'] = '.jpg'
            cfg['qext'] = '.jpg'
        else:
            cfg = {}
            cfg['imlist_fname'] = osp.join(self.root, '{}.txt'.format(dataset_name))
            cfg['imlist'] = self.read_imlist(cfg['imlist_fname'])
            cfg['qimlist'] = []
            cfg['ext'] = ''
            cfg['qext'] = ''

        cfg['dir_data'] = osp.join(root, dataset_name)
        cfg['dir_images'] = osp.join(cfg['dir_data'], 'jpg')

        cfg['n'] = len(cfg['imlist'])
        cfg['nq'] = len(cfg['qimlist'])
        self.cfg = cfg
        self.transform = transform
        if query_flag:
            self.imlist = self.cfg['qimlist']
        else:
            self.imlist = self.cfg['imlist']
        self.query_gts = ''
        if query_flag:
            self.query_gts = cfg['gnd']

    def read_imlist(self,imlist_fn):
        with open(imlist_fn, 'r') as file:
            imlist = file.read().splitlines()
        return imlist

    def config_imname(self,i):
        return osp.join(self.cfg['dir_images'], self.imlist[i] + self.cfg['qext'] if self.query_flag else self.imlist[i] + self.cfg['ext'])

    def __getitem__(self, index):
        path = self.config_imname(index)
        img = self._reader(path)
        img = self.transform(img)
        return img,index


    def __len__(self):
        if self.query_flag:
            return len(self.cfg['qimlist'])
        else:
            return len(self.cfg['imlist'])



