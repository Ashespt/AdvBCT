from models.base_model import CreateBaseModel, CreateBaseReidModel,CreateModelWithTransformation
from models.modules import MLP
import os.path as osp


def build_vanilla_model(task='landmark', arch='resnet18', pretrained_path=None, out_dim=256,
                        pretrained=True, num_class=100000, use_cls=True):
    args = {"arch": arch,
            "pretrained": pretrained,
            "num_class": num_class,
            "use_cls": use_cls,
            "out_dim": out_dim,
            "pretrained_path": pretrained_path}
    if task == 'landmark':
        model = CreateBaseModel(args)
    else:
        cls_args = {
            'loss_type': 'triplet',
            'cosine_scale': 30,
            'cosine_margin': 0.5
        }
        model = CreateBaseReidModel(args, cls_args)
    return None, model


def build_tencent_bct(task='landmark', arch_old='resnet18', pretrained_old=False, pretrained_path_old=None,
                      num_class_old=100000, use_cls_old=True, out_dim=256,
                      arch_new='resnet18', pretrained_new=True, pretrained_path_new=None, num_class_new=100000,
                      use_cls_new=True,eboundary=False):
    args_old = {"arch": arch_old,
                "pretrained": pretrained_old,
                "pretrained_path": pretrained_path_old,
                "num_class": num_class_old,
                "use_cls": use_cls_old,
                "out_dim": out_dim},
    args_new = {"arch": arch_new,
                "pretrained": pretrained_new,
                "pretrained_path": pretrained_path_new,
                "num_class": num_class_new,
                "use_cls": use_cls_new,
                "out_dim": out_dim,
                "eboundary":eboundary}
    old_model = CreateBaseModel(args_old)
    for param in old_model.parameters():  # fix old parameters
        param.requires_grad = False
    if task == 'landmark':
        new_model = CreateBaseModel(args_new)
    else:
        cls_args = {
            'loss_type': 'triplet',
            'cosine_scale': 30,
            'cosine_margin': 0.5
        }
        new_model = CreateBaseReidModel(args_new, cls_args)
    return old_model, new_model


def build_bct(task='landmark', arch_old='resnet18', pretrained_old=False, pretrained_path_old=None,
              num_class_old=100000, use_cls_old=True, out_dim=256,
              arch_new='resnet18', pretrained_new=True, pretrained_path_new=None, num_class_new=100000,
              use_cls_new=True,eboundary=False):
    args_old = {"arch": arch_old,
                "pretrained": pretrained_old,
                "pretrained_path": pretrained_path_old,
                "num_class": num_class_old,
                "use_cls": use_cls_old,
                "out_dim": out_dim}
    args_new = {"arch": arch_new,
                "pretrained": pretrained_new,
                "pretrained_path": pretrained_path_new,
                "num_class": num_class_new,
                "use_cls": use_cls_new,
                "out_dim": out_dim,
                "eboundary":eboundary}
    if task == 'landmark':
        old_model = CreateBaseModel(args_old)
    else:
        cls_args = {
            'loss_type': 'triplet',
            'cosine_scale': 30,
            'cosine_margin': 0.5
        }
        old_model = CreateBaseReidModel(args_old, cls_args)

    for param in old_model.parameters():  # fix old parameters
        param.requires_grad = False
    if task == 'landmark':
        new_model = CreateBaseModel(args_new)
    else:
        cls_args = {
            'loss_type': 'triplet',
            'cosine_scale': 30,
            'cosine_margin': 0.5
        }
        new_model = CreateBaseReidModel(args_new, cls_args)
    return old_model, new_model


def build_bct_transformation(task='landmark', arch_old='resnet18', pretrained_old=False, pretrained_path_old=None,
              num_class_old=100000, use_cls_old=True, out_dim=256,
              arch_new='resnet18', pretrained_new=True, pretrained_path_new=None, num_class_new=100000,
              use_cls_new=True,K=0,eboundary=False):
    args_old = {"arch": arch_old,
                "pretrained": pretrained_old,
                "pretrained_path": pretrained_path_old,
                "num_class": num_class_old,
                "use_cls": use_cls_old,
                "out_dim": out_dim}
    args_new = {"arch": arch_new,
                "pretrained": pretrained_new,
                "pretrained_path": pretrained_path_new,
                "num_class": num_class_new,
                "use_cls": use_cls_new,
                "out_dim": out_dim,
                "eboundary":eboundary}
    if task == 'landmark':
        old_model = CreateBaseModel(args_old)
        if K !=0:
            old_model = CreateModelWithTransformation(old_model,K)
        else:
            for param in old_model.parameters():  # fix old parameters
                param.requires_grad = False
    else:
        cls_args = {
            'loss_type': 'triplet',
            'cosine_scale': 30,
            'cosine_margin': 0.5
        }
        old_model = CreateBaseReidModel(args_old, cls_args)

    if task == 'landmark':
        new_model = CreateBaseModel(args_new)
    else:
        cls_args = {
            'loss_type': 'triplet',
            'cosine_scale': 30,
            'cosine_margin': 0.5
        }
        new_model = CreateBaseReidModel(args_new, cls_args)
    return old_model, new_model

def build_adversarial_bct(task='landmark', arch_old='resnet18', pretrained_old=False, pretrained_path_old=None,
                          num_class_old=100000, use_cls_old=True, out_dim=256,
                          arch_new='resnet18', pretrained_new=True, pretrained_path_new=None, num_class_new=100000,
                          use_cls_new=True,eboundary=False):
    args_old = {"arch": arch_old,
                "pretrained": pretrained_old,
                "pretrained_path": pretrained_path_old,
                "num_class": num_class_old,
                "use_cls": use_cls_old,
                "out_dim": out_dim}
    args_new = {"arch": arch_new,
                "pretrained": pretrained_new,
                "pretrained_path": pretrained_path_new,
                "num_class": num_class_new,
                "use_cls": use_cls_new,
                "out_dim": out_dim,
                "adversarial": True,
                "eboundary":eboundary}
    if task == 'landmark':
        old_model = CreateBaseModel(args_old)
    else:
        cls_args = {
            'loss_type': 'triplet',
            'cosine_scale': 30,
            'cosine_margin': 0.5
        }
        old_model = CreateBaseReidModel(args_old, cls_args)
    for param in old_model.parameters():  # freeze old parameters
        param.requires_grad = False
    if task == 'landmark':
        new_model = CreateBaseModel(args_new)
    else:
        cls_args = {
            'loss_type': 'triplet',
            'cosine_scale': 30,
            'cosine_margin': 0.5
        }
        new_model = CreateBaseReidModel(args_new, cls_args)

    if old_model.out_dim != new_model.out_dim:
        raise Exception('The feature dimension of new model should be same as the old one.')

    return old_model, new_model
