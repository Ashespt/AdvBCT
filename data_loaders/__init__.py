from data_loaders.landmark.gldv2 import ROxfordParisTestDataset,Gldv2TrainDataset,Gldv2TestDataset, Gldv2TrainDataset1
import torch
from typing import Sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.data.random_erasing import RandomErasing
import torch.distributed as dist
from data_loaders.dataset.sampler_ddp import RandomIdentitySampler_DDP,train_collate_fn,val_collate_fn, RandomIdentitySampler1_DDP
factory_train_landmark_dataset={
    "gldv2":Gldv2TrainDataset
}

factory_test_landmark_dataset={
    "roxford5k":ROxfordParisTestDataset,
    "rparis6k":ROxfordParisTestDataset,
    "gldv2":Gldv2TestDataset
}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def build_transform(im_size):
    if not isinstance(im_size,Sequence):
        im_size = [im_size,im_size]
    base_trans = transforms.Compose([
        transforms.Resize(im_size),
        transforms.ToTensor(),
        normalize,
    ])
    return base_trans

def build_train_dataloader(data_name,file_dir=None,root='./',distributed=False,input_size=[224,224],batch_size=16, num_workers=4,
                       pin_memory=True,transform_train=None, drop_last=True, class_sampler=False):
    '''
    :param data_name: the name of dataset
    :param file_dir: GLDv2 path of train.txt, MS1M path of im.txt
    :param root: GLDv2 root of images, MS1M root path of train.idx, train.rec and etc.
    :param distributed:
    :param batch_size:
    :param num_workers:
    :param pin_memory:
    :param transform_train:
    :return:
    '''
    assert isinstance(data_name,str)
    assert data_name in [*factory_train_landmark_dataset]
    if transform_train is None:
        transform_train = build_transform(input_size)
    dataset = factory_train_landmark_dataset[data_name](root,file_dir,transform_train)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        if class_sampler:
            data = Gldv2TrainDataset1(root, file_dir)
            data_sampler = RandomIdentitySampler1_DDP(data.imlist, batch_size, 2)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler,batch_size, True)
    else:
        sampler = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None),
                        pin_memory=pin_memory, num_workers=num_workers, sampler=sampler, drop_last=drop_last)
    # if class_sampler:
    #     loader = DataLoader(dataset, pin_memory=pin_memory, num_workers=num_workers, batch_sampler=batch_sampler)
    return loader


def build_test_dataloader(data_name,root='./data',transform_test=None,query_flag=True,distributed=False,input_size=[224,224], batch_size=64,
                         num_workers=0, pin_memory=True):
    assert isinstance(data_name, str)
    assert data_name in [*factory_test_landmark_dataset]
    if transform_test is None:
        transform_test = build_transform(input_size)
    dataset = factory_test_landmark_dataset[data_name](data_name,root,transform_test,query_flag)
    gts = dataset.query_gts
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      pin_memory=pin_memory, num_workers=num_workers,
                      sampler=sampler, drop_last=False), gts


