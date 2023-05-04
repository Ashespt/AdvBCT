import torch
import os
import argparse
import json
import numpy as np
from collections import defaultdict
from models import build_bct_models
from configs import get_config
from data_loaders import build_train_dataloader
import math

def main(config):

    _, model = build_bct_models("base_model", configs=config, debug=False)
    model.eval()
    model = model.cuda()
    data_loader = build_train_dataloader(config.TRAIN.DATASET, root=config.TRAIN.ROOT, file_dir=config.TRAIN.FILE_DIR,
                                         batch_size=config.TRAIN.BATCH_SIZE, distributed=False, drop_last=False)
    print('build data_loader', len(data_loader))
    labels_all = np.empty(len(data_loader.sampler), dtype=np.float32)
    features_all = np.empty((len(data_loader.sampler), config.MODEL.EMB_DIM), dtype=np.float32)
    print('feat shape: ', features_all.shape)
    pointer = 0
    if not os.path.exists(config.EVAL.SAVE_DIR):
        os.makedirs(config.EVAL.SAVE_DIR)
    for batch_idx, (images, labels) in enumerate(data_loader):
        with torch.no_grad():
            feat = model(images.cuda())
        batchsize = labels.size(0)
        features_all[pointer:pointer + batchsize] = feat.cpu().numpy()
        labels_all[pointer:pointer + batchsize] = labels.numpy()
        pointer += batchsize
        if batch_idx % 20 == 0:
            print(batch_idx, len(data_loader), batchsize, feat.cpu().numpy().shape)
    np.save(os.path.join(config.EVAL.SAVE_DIR, f'{config.TRAIN.DATASET}_{features_all.shape[0]}_feat.npy'), features_all)
    np.save(os.path.join(config.EVAL.SAVE_DIR, f'{config.TRAIN.DATASET}_{features_all.shape[0]}_label.npy'), labels_all)
    return features_all.shape[0]


def gen_class_meta(config,feat_num=0):
    feat = np.load(os.path.join(config.EVAL.SAVE_DIR, f'{config.TRAIN.DATASET}_{feat_num}_feat.npy'))
    label = np.load(os.path.join(config.EVAL.SAVE_DIR, f'{config.TRAIN.DATASET}_{feat_num}_label.npy'))
    print('feat, label', feat.shape, label.shape)
    feat_dict = defaultdict(list)
    for i in range(len(label)):
        feat[i] = feat[i] / np.linalg.norm(feat[i])
        feat_dict[int(label[i])].append(feat[i])

    data_dict = {}
    for i, k in enumerate(feat_dict.keys()):
        center = np.asarray(feat_dict[k]).mean(0)
        # center = center/np.linalg.norm(center)
        # print('ddd', np.asarray(feat_dict[k]).shape, center.shape)
        maxtmp = 0
        radius = []
        for f in feat_dict[k]:
            # f = f/np.linalg.norm(f)
            # diff = f - center/np.linalg.norm(center)
            diff = f - center
            tmp = np.linalg.norm(diff)
            maxtmp = max(tmp, maxtmp)
            radius.append(tmp.item())

        radius = sorted(radius)
        # 1.5IQR
        radius_new = []
        maxtmp = 0
        if len(radius) >= 4:
            nu = len(radius)
            q3, q1 = radius[int(3 * nu / 4)], radius[int(nu / 4)]
            IQR = q3 - q1
            for r in radius:
                if r < q1 - 1.5 * IQR or r > q3 + 1.5 * IQR:
                    continue
                maxtmp = max(r, maxtmp)
                radius_new.append(r)
        else:
            for r in radius:
                maxtmp = max(r, maxtmp)
                radius_new.append(r)
        if len(radius_new) == 0:
            radius_new = radius

        data_dict[k] = {'center': center.tolist(), 'radius': radius_new}
        print(i, len(feat_dict), k, len(radius), radius, maxtmp)
    # break
    with open(os.path.join(config.EVAL.SAVE_DIR, f'{config.TRAIN.DATASET}_{feat_num}_meta_radius_centernorm.json'), 'w') as fw:
        json.dump(data_dict, fw)

def gen_class_theta(config,feat_num=0):
    feat = np.load(os.path.join(config.EVAL.SAVE_DIR, f'{config.TRAIN.DATASET}_{feat_num}_feat.npy'))
    label = np.load(os.path.join(config.EVAL.SAVE_DIR, f'{config.TRAIN.DATASET}_{feat_num}_label.npy'))
    print('feat, label', feat.shape, label.shape)
    feat_dict = defaultdict(list)
    for i in range(len(label)):
        feat[i] = feat[i] / np.linalg.norm(feat[i])
        feat_dict[int(label[i])].append(feat[i])

    data_dict = {}
    for i, k in enumerate(feat_dict.keys()):
        center = np.asarray(feat_dict[k]).mean(0)
        # center = center/np.linalg.norm(center)
        # print('ddd', np.asarray(feat_dict[k]).shape, center.shape)
        radius = []
        for f in feat_dict[k]:
            # f = f / np.linalg.norm(f)
            diff = min(np.dot(f,center/np.linalg.norm(center)),1)
            theta = math.acos(diff)
            radius.append(theta)
        radius = sorted(radius)
        #1.5IQR
        radius_new = []
        maxtmp = 0
        if len(radius)>=4:
            nu = len(radius)
            q3,q1 = radius[int(3*nu/4)], radius[int(nu/4)]
            IQR = q3-q1
            for r in radius:
                if r < q1-1.5*IQR or r > q3+1.5*IQR:
                    continue
                maxtmp = max(r, maxtmp)
                radius_new.append(r)
        else:
            for r in radius:
                maxtmp = max(r, maxtmp)
                radius_new.append(r)
        if len(radius_new) == 0:
            radius_new = radius

        data_dict[k] = {'center': center.tolist(), 'radius': radius_new}
        print(i, len(feat_dict), k, len(radius_new), radius_new, maxtmp)
    # break
    with open(os.path.join(config.EVAL.SAVE_DIR, f'{config.TRAIN.DATASET}_{feat_num}_meta_theta_centernorm_after1.5iqr.json'), 'w') as fw:
        json.dump(data_dict, fw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BCT training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args, _ = parser.parse_known_args()
    config = get_config(args)
    config.defrost()
    print('config: ', config)
    feat_num = main(config)
    # feat_num = 470369
    gen_class_meta(config,feat_num)
    gen_class_theta(config,feat_num)

