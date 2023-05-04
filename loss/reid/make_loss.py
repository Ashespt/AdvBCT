# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes,feat_dim=256):    # modified by gu
    sampler = cfg.TRAIN.SAMPLER
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.TRAIN.METRIC_LOSS_TYPE:
        if cfg.TRAIN.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.TRAIN.TRIPLET_MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.TRAIN.TRIPLET_MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.TRAIN.METRIC_LOSS_TYPE))

    if cfg.TRAIN.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.TRAIN.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            if cfg.TRAIN.METRIC_LOSS_TYPE == 'triplet':
                if cfg.TRAIN.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    # return TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    return cfg.TRAIN.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.TRAIN.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    # return TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.TRAIN.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.TRAIN.SAMPLER))
    return loss_func, center_criterion


