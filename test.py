import argparse
import os
from datetime import timedelta

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from data_loaders import build_test_dataloader
from models import build_bct_models
from logger.logger import create_logger
from utils.util import cudalize
from evaluate.eval import evaluate_func
from evaluate.eval_hot_refresh import evaluate_func as evaluate_func_refresh
from configs import get_config


def main(args):

    ngpus_per_node = torch.cuda.device_count()

    if args.DISTRIBUTED:
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.EVAL.DEVICE, ngpus_per_node, args)

def setup(rank,world_size,dist_backend='nccl',dist_url='tcp://localhost:10001'):

    # initialize the process group
    dist.init_process_group(backend=dist_backend, init_method=dist_url,
                            world_size=world_size, rank=rank)
    dist.barrier()
def main_worker(device, ngpus_per_node, args):
    rank = device
    world_size = ngpus_per_node * args.WORLD_SIZE
    if device == 1:
        print(args)
    if args.DISTRIBUTED:
        setup(rank,world_size,args.DIST_BACKEND,args.DIST_URL)
    cudnn.benchmark = True

    logger = create_logger(output_dir=args.EVAL.LOG_DIR, dist_rank=rank, name='eval')


    if args.EVAL.DATASET_TYPE == 'landmark':
        if args.EVAL.DATASET_NAME in ['gldv2','roxford5k', 'rparis6k']:
            query_loader, query_gts = build_test_dataloader(args.EVAL.DATASET, root=args.EVAL.ROOT,
                                                                 distributed=False, batch_size=args.TRAIN.BATCH_SIZE,
                                                                 query_flag=True)
            gallery_loader, _ = build_test_dataloader(args.EVAL.DATASET, root=args.EVAL.ROOT,
                                                           distributed=False, batch_size=16,
                                                           query_flag=False,num_workers=4)
            query_gts = query_gts
        else:
            raise NotImplementedError

    elif args.EVAL.DATASET_TYPE == 'face':
        query_loader = build_test_dataloader('ijb-c', './data/ijbc', query_flag=True,distributed=False)
        gallery_loader = build_test_dataloader('ijb-c', './data/ijbc', query_flag=False,distributed=False)
        query_gts = query_loader.query_gts
    else:
        raise NotImplementedError

    # build model architecture

    old_model,new_model = build_bct_models(args.EVAL.BCT_TYPE,args)[:2]
    new_model = cudalize(new_model, distributed=args.DISTRIBUTED,device=device,arch=args.NEW_MODEL.ARCH)
    # new_model.cuda()

    if old_model is None:
        logger.info(f"=> Self-model performance:")
        evaluate_func(new_model, query_loader, gallery_loader, query_gts, logger, args,
                      old_model=None, dataset_name=args.EVAL.DATASET_NAME)
    else:
        old_model.cuda(device)
        logger.info(f"=> Cross-model performance:")
        if args.EVAL.HOT_REFRESH:
            evaluate_func_refresh(new_model, query_loader, gallery_loader, query_gts, logger, args,
                          old_model=old_model, dataset_name=args.EVAL.DATASET_NAME,hot_refresh=True)
        else:
            evaluate_func(new_model, query_loader, gallery_loader, query_gts, logger, args,
                          old_model=old_model, dataset_name=args.EVAL.DATASET_NAME)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BCT training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args, _ = parser.parse_known_args()
    config = get_config(args)
    config.defrost()

    main(config)