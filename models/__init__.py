from models.build import build_bct,build_adversarial_bct,build_tencent_bct,build_vanilla_model,build_bct_transformation
from models.margin_softmax import large_margin_module
from models.loss import BackwardCompatibleLoss,UpgradeLoss,UpgradeCenterLoss,UpgradeCenterPartialLoss
factory_model={
    "base_model":build_vanilla_model,
    "bct":build_bct,
    "adv_bct":build_adversarial_bct
}

def build_bct_models(name,configs=None,debug=False):
    '''

    :param name:
    :param args:
    :return: model if not bct else new model and old model.
             for adv_bct, return new model, old model and discriminator
    '''
    if name == 'base_model':
        if debug:
            args = {
                'task': 'landmark',
                "arch": 'resnet18',
                "pretrained": False,
                "pretrained_path": None,
                "num_class": 100000,
                "use_cls": True,
                "out_dim": 256}
            return factory_model[name](**args)
        args = {
            'task': configs.TRAIN.DATASET_TYPE,
            "arch": configs.MODEL.ARCH,
            "pretrained": configs.MODEL.PRETRAINED,
            "pretrained_path": configs.MODEL.PRETRAINED_PATH,
            "num_class": configs.MODEL.NUM_CLASSES,
            "use_cls": configs.MODEL.USE_CLS,
            "out_dim": configs.MODEL.EMB_DIM}

    else:
        args = {
            'task': configs.TRAIN.DATASET_TYPE,
            "arch_old":configs.OLD_MODEL.ARCH,
            "pretrained_old":configs.OLD_MODEL.PRETRAINED,
            "pretrained_path_old":configs.OLD_MODEL.PRETRAINED_PATH,
            "num_class_old":configs.OLD_MODEL.NUM_CLASSES,
            "use_cls_old":configs.OLD_MODEL.USE_CLS,

            "out_dim":configs.MODEL.EMB_DIM,

            "arch_new":configs.NEW_MODEL.ARCH,
            "pretrained_new": configs.NEW_MODEL.PRETRAINED,
            "pretrained_path_new": configs.NEW_MODEL.PRETRAINED_PATH,
            "num_class_new":configs.NEW_MODEL.NUM_CLASSES,
            "use_cls_new": configs.NEW_MODEL.USE_CLS,
            "eboundary": configs.COMP_LOSS.ELASTIC_BOUNDARY
        }
    return factory_model[name](**args)
