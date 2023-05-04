import os
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn

from configs import get_config
from logger.logger import create_logger
from data_loaders import build_train_dataloader, build_test_dataloader, build_reid_dataloader
from models import build_bct_models
from trainer import LandmarkTrainer, LandmarkComTrainer,ReidTrainer, ReidComTrainer
from utils.util import cudalize, set_random_seed, resume_checkpoint
from models import BackwardCompatibleLoss,UpgradeLoss,UpgradeCenterLoss,UpgradeCenterPartialLoss
from solver.scheduler_factory import create_scheduler



def main(config):
    print('config: ', config)

    # fix random seeds for reproducibility
    if config.SEED is not None:
        set_random_seed(config.SEED)

    if config.DIST_URL == "env://" and config.WORLD_SIZE == -1:
        config.WORLD_SIZE = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    if config.MULTIPROCESSING_DISTRIBUTED:
        config.WORLD_SIZE = ngpus_per_node * config.WORLD_SIZE
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        main_worker(config.DEVICE, ngpus_per_node, config)


def main_worker(device, ngpus_per_node, config):
    config.LOCAL_RANK = device
    if config.DISTRIBUTED:
        if config.DIST_URL == "env://" and config.RANK == -1:
            config.RANK = int(os.environ["RANK"])
        if config.MULTIPROCESSING_DISTRIBUTED:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.RANK = config.RANK * ngpus_per_node + device
        dist.init_process_group(backend=config.DIST_BACKEND, init_method=config.DIST_URL,
                                world_size=config.WORLD_SIZE, rank=config.RANK)
        dist.barrier()
    cudnn.benchmark = True

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=config.LOCAL_RANK, name='train')

    # build model architecture
    if config.TRAIN.TYPE == 'base':
        _, model = build_bct_models("base_model", configs=config, debug=False)
    else:
        if config.TRAIN.TYPE == 'compatible':
            old_model, model = build_bct_models("bct", configs=config, debug=False)
        elif config.TRAIN.TYPE == 'adv_compatible':
            old_model, model = build_bct_models("adv_bct", configs=config, debug=False)
        else:
            logger.warn(f'train type {config.TRAIN.TYPE}')
            raise NotImplementedError
        # state_dict = torch.load(config.OLD_MODEL.MODEL_PATH, map_location='cpu')
        # old_model.load_state_dict(state_dict['model'])
        # logger.info(f"Load old model {config.OLD_MODEL.MODEL_PATH}")
    # if dist.get_rank() == 0:
    #     print('model', model)

    # build solver
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9,
                                weight_decay=5e-4)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)

    best_acc1 = 0.
    if config.DISTRIBUTED:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        torch.cuda.set_device(config.LOCAL_RANK)
        model.cuda(config.LOCAL_RANK)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK],
                                                          find_unused_parameters=True)

    if config.DISTRIBUTED and 'compatible' in config.TRAIN.TYPE:
        old_model = old_model.cuda()

    #build dataloader
    if config.TRAIN.DATASET_TYPE == 'landmark':
        train_loader = build_train_dataloader(config.TRAIN.DATASET, root=config.TRAIN.ROOT,
                                              file_dir=config.TRAIN.FILE_DIR, batch_size=config.TRAIN.BATCH_SIZE,
                                              input_size=config.TRAIN.INPUT_SIZE,
                                              distributed=config.DISTRIBUTED)
        test_query_loader, query_gts = build_test_dataloader(config.EVAL.DATASET, root=config.EVAL.ROOT,
                                                             distributed=False, batch_size=config.TRAIN.BATCH_SIZE,
                                                             input_size=config.TRAIN.INPUT_SIZE,
                                                             query_flag=True)
        test_gallery_loader, _ = build_test_dataloader(config.EVAL.DATASET, root=config.EVAL.ROOT, distributed=False,
                                                       input_size=config.TRAIN.INPUT_SIZE,
                                                       batch_size=config.TRAIN.BATCH_SIZE, query_flag=False)
        test_query_gts = query_gts
    else:
        train_loader, val_loader, num_query, camera_num, view_num = build_reid_dataloader(
            config.TRAIN.SAMPLER, data_name=config.TRAIN.DATASET, root=config.TRAIN.ROOT,
            file_dir=config.TRAIN.FILE_DIR,
            batch_size=config.TRAIN.BATCH_SIZE, distributed=config.DISTRIBUTED)


    # build loss
    criterion = {}
    criterion['base'] = nn.CrossEntropyLoss()
    if config.UPGRADE_LOSS.TYPE == 'center_limit':
        from pytorch_metric_learning.utils import distributed as pml_dist
        from pytorch_metric_learning import losses as pml_loss
        loss_fn = pml_loss.ContrastiveLoss(pos_margin=0, neg_margin=0.7)
        # criterion['upgrade'] = pml_dist.DistributedLossWrapper(loss=UpgradeLoss(loss_fn,embedding_size=256,
        #                     # memory_size=max(int(config.TRAIN.BATCH_SIZE*(dist.get_world_size()*2)),config.MODEL.NUM_CLASSES)),
        #                     memory_size=4096),
        #                     device_ids=device)
        criterion['upgrade'] = pml_dist.DistributedLossWrapper(loss=UpgradeCenterPartialLoss(loss_fn, embedding_size=256,
                                            num_class=config.MODEL.NUM_CLASSES,device_id=device,
                                            device_num=dist.get_world_size(),dataset_len=len(train_loader)))
    if config.COMP_LOSS.TYPE in ["bct", "bct_ract"]:
        criterion['back_comp'] = nn.CrossEntropyLoss().cuda()
    elif config.COMP_LOSS.TYPE in ["contra", "triplet", "l2", "hot_refresh", "triplet_ract","bct_limit","bct_limit_no_s2c"]:
        criterion['back_comp'] = BackwardCompatibleLoss(temp=config.COMP_LOSS.TEMPERATURE,
                                                        margin=config.COMP_LOSS.TRIPLET_MARGIN,
                                                        topk_neg=config.COMP_LOSS.TOPK_NEG,
                                                        loss_type=config.COMP_LOSS.TYPE,
                                                        loss_weight=config.COMP_LOSS.WEIGHT,
                                                        gather_all=config.GATHER_ALL)

    if config.TRAIN.DATASET_TYPE == 'landmark':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 10, 20], gamma=0.1, last_epoch=-1)
        test_loader_list = [test_query_loader, test_gallery_loader, test_query_gts]
        validation_loader_list = test_loader_list
        logger.info(f"start to train")
        if config.TRAIN.TYPE == 'base':
            # validation_loader_list = [eval_query_loader, eval_gallery_loader, eval_query_gts]
            trainer = LandmarkTrainer(model,
                                      train_loader=train_loader,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      grad_scaler=grad_scaler,
                                      config=config,
                                      logger=logger,
                                      validation_loader_list=validation_loader_list,
                                      test_loader_list=test_loader_list,
                                      lr_scheduler=lr_scheduler)

        elif config.TRAIN.TYPE in ['compatible','adv_compatible']:
            trainer = LandmarkComTrainer(model,
                                         old_model=old_model,
                                         train_loader=train_loader,
                                         criterion=criterion,
                                         optimizer=optimizer,
                                         grad_scaler=grad_scaler,
                                         config=config,
                                         logger=logger,
                                         validation_loader_list=validation_loader_list,
                                         test_loader_list=test_loader_list,
                                         lr_scheduler=lr_scheduler)

    elif config.TRAIN.DATASET_TYPE == 'reid':
        lr_scheduler = create_scheduler(config, optimizer)
        logger.info(f"start to train")
        if config.TRAIN.TYPE == 'base':
            trainer = ReidTrainer(model,
                                  train_loader=train_loader,
                                  val_loader=val_loader,
                                  grad_scaler=grad_scaler,
                                  config=config,
                                  logger=logger,
                                  lr_scheduler=lr_scheduler,
                                  num_classes=config.MODEL.NUM_CLASSES,
                                  num_query=num_query
                                  )
        elif config.TRAIN.TYPE == 'compatible':
            trainer = ReidComTrainer(model,
                                     old_model=old_model,
                                     train_loader=train_loader,
                                     val_loader=val_loader,
                                     criterion=criterion,
                                     grad_scaler=grad_scaler,
                                     config=config,
                                     logger=logger,
                                     lr_scheduler=lr_scheduler,
                                     num_classes=config.MODEL.NUM_CLASSES,
                                     num_query=num_query
                                     )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BCT training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args, _ = parser.parse_known_args()
    config = get_config(args)
    config.defrost()
    main(config)
