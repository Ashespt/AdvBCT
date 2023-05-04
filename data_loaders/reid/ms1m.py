from torch.utils.data import Dataset
import os.path as osp
import mxnet as mx
import numbers
import torch
import numpy as np
from PIL import Image
def files_reader(path,root):
    flist = []
    with open(path) as f:
        for line in f.readlines():
            [imid,label]=line.split()
            flist.append([osp.join(root,imid),int(label)])
    return flist

class Ms1mTrainDataset(Dataset):
    def __init__(self,root,flist,transform):
        if not isinstance(root,str):
            raise BaseException('Ms1mTrainDataset: root must be the path of train.rec and train.idx')
        if isinstance(flist,str):
            lines = []
            with open(flist) as f:
                for line in f.readlines():
                    lines.append(int(line.split()[0]))
            flist = lines
        path_imgrec = osp.join(root, 'train.rec')
        path_imgidx = osp.join(root, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)

        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
        self.transform = transform
        self.valid_idx = flist

    def __getitem__(self, index):
        index_im = self.valid_idx[index]
        idx = self.imgidx[index_im]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        sample = Image.fromarray(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.valid_idx)


class IJBcTestDataset(Dataset):
    def __init__(self,dataset_name,root,transform,query_flag=True):
        gallery_s1_record = "ijbc_1N_gallery_G1.csv"
        gallery_s2_record = "ijbc_1N_gallery_G2.csv"
        gallery_s1_templates, gallery_s1_subject_ids = self.read_template_subject_id_list(
            osp(meta_dir, gallery_s1_record))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        sample = Image.fromarray(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


    def read_template_subject_id_list(self,path):
        ijb_meta = np.loadtxt(path, dtype=str, skiprows=1, delimiter=',')
        templates = ijb_meta[:, 0].astype(np.int)
        subject_ids = ijb_meta[:, 1].astype(np.int)
        return templates, subject_ids
