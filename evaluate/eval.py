import torch
import numpy as np
import os
import os.path as osp
import torch.distributed as dist
# import mkl
# mkl.get_max_threads()
import faiss, time
from evaluate.roxford_rparis_metrics import calculate_mAP_roxford_rparis
from evaluate.metric import calculate_mAP_gldv2,calculate_mAP_ijb_c
from utils.util import AverageMeter


def concat_file(_save_dir, file_name, final_size):
    save_path = osp.join(_save_dir, f"{file_name}.npy")
    data = np.load(save_path)
    return data
    # concat_ret = np.empty(final_size, dtype=np.float32)
    # pointer = 0
    # for rank in range(dist.get_world_size()):
    #     file_path = osp.join(_save_dir, f"{file_name}_rank{rank}.npy")
    #     data = np.load(file_path)
    #     data_size = data.shape[0]
    #     print(rank, file_name, final_size, data_size, data.shape)

    #     concat_ret[pointer:pointer + data_size] = data
    #     pointer += data_size
    #     os.remove(file_path)
    # save_path = osp.join(_save_dir, f"{file_name}.npy")
    # np.save(save_path, concat_ret)
    # return concat_ret

# def concat_file(_save_dir, file_name, final_size):
#     concat_ret = np.empty(final_size, dtype=np.float32)
#     pointer = 0
#     # for rank in range(dist.get_world_size()):
#     file_path = osp.join(_save_dir, f"{file_name}_rank.npy")
#     data = np.load(file_path)
#     data_size = data.shape[0]
#     concat_ret[pointer:pointer + data_size] = data
#     pointer += data_size
#     os.remove(file_path)
#     save_path = osp.join(_save_dir, f"{file_name}.npy")
#     np.save(save_path, concat_ret)
#     return concat_ret


def calculate_rank(logger, query_feats, gallery_feats, topk):
    logger.info(f"query_feats shape: {query_feats.shape}")
    logger.info(f"gallery_feats shape: {gallery_feats.shape}")
    num_q, feat_dim = query_feats.shape

    logger.info("=> build faiss index")
    gallery_feats = gallery_feats / np.linalg.norm(gallery_feats, axis=1)[:, np.newaxis]
    query_feats = query_feats / np.linalg.norm(query_feats, axis=1)[:, np.newaxis]
    faiss_index = faiss.IndexFlatIP(feat_dim)
    # faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
    faiss_index.add(gallery_feats)
    logger.info("=> begin faiss search")
    _, ranked_gallery_indices = faiss_index.search(query_feats, topk)
    return ranked_gallery_indices


def evaluate_func(model, query_loader, gallery_loader, query_gts,logger,
                  config, old_model=None, dataset_name='gldv2'):
    model.eval()
    if old_model is None:   # self-model test
        old_model = model
        logger.info("=> same-model test")
    else:   # cross-model test
        logger.info("=> cross-model test")
        old_model.eval()
    # print('query_loader', query_loader,gallery_loader, dist.get_rank())
    if dist.get_rank() == 0:
        # extract query feat with new model
        extract_features(model, query_loader, 'q', logger, config)
        # # extract gallery feat with old/new model
        extract_features(old_model, gallery_loader, 'g', logger, config)  # use old_model to extract
        # if dist.get_rank() == 0:
        # dist.barrier()
        # torch.cuda.empty_cache()  # empty gpu cache if using faiss gpu index

    mAP = 0.0
    if torch.distributed.get_rank() == 0:
        logger.info("=> concat feat and label file")
        query_feats = concat_file(config.EVAL.SAVE_DIR, "feat_q",
                                  final_size=(len(query_loader.dataset), config.MODEL.EMB_DIM))
        query_labels = concat_file(config.EVAL.SAVE_DIR, "label_q",
                                   final_size=(len(query_loader.dataset),))
        query_labels = query_labels.astype(np.int32)

        gallery_feats = concat_file(config.EVAL.SAVE_DIR, "feat_g",
                                    final_size=(len(gallery_loader.dataset), config.MODEL.EMB_DIM))
        gallery_labels = concat_file(config.EVAL.SAVE_DIR, "label_g",
                                     final_size=(len(gallery_loader.dataset),))
        gallery_labels = gallery_labels.astype(np.int32)

        logger.info("=> calculate rank")
        if dataset_name == 'gldv2':
            ranked_gallery_indices = calculate_rank(logger, query_feats, gallery_feats, topk=100)
            logger.info("=> calculate mAP")
            mAP = calculate_mAP_gldv2(ranked_gallery_indices, query_gts[2], topk=100)
        elif dataset_name == 'ijb_c':
            ranked_gallery_indices = calculate_rank(logger, query_feats, gallery_feats, topk=100)
            logger.info("=> calculate mAP")
            mAP = calculate_mAP_ijb_c(ranked_gallery_indices, query_gts[2], topk=100)
        elif dataset_name == 'roxford5k' or dataset_name == 'rparis6k':
            ranked_gallery_indices = calculate_rank(logger, query_feats, gallery_feats, topk=gallery_feats.shape[0])
            logger.info("=> calculate mAP")
            mAP = calculate_mAP_roxford_rparis(logger, ranked_gallery_indices.transpose(), query_gts)
        else:
            raise ValueError
        # dist.barrier()
        # torch.cuda.empty_cache()
        print(f'mAP: {mAP}')
    return mAP


@torch.no_grad()
def extract_features(model, data_loader, tag, logger, config):
    batch_time = AverageMeter('Process Time', ':6.3f')
    data_time = AverageMeter('Test Data Time', ':6.3f')

    labels_all = np.empty(len(data_loader.sampler), dtype=np.float32)
    features_all = np.empty((len(data_loader.sampler), config.MODEL.EMB_DIM), dtype=np.float32)
    # print('ddd', data_loader.sampler, dist.get_rank())
    pointer = 0
    rank = dist.get_rank()
    end = time.time()
    for i, (images, labels) in enumerate(data_loader):
        data_time.update(time.time() - end)
        images, labels = images.to('cuda'), labels.to('cpu')
        # print(i, rank, len(data_loader.dataset), images.shape, images[0,0,0,:3])

        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            feat = model(images.cuda())
        batchsize = labels.size(0)
        features_all[pointer:pointer + batchsize] = feat.cpu().numpy()
        labels_all[pointer:pointer + batchsize] = labels.numpy()
        pointer += batchsize

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.EVAL.PRINT_FREQ == 0:
            logger.info('Extract {} Features on rank {}: [{}/{}/{}]\t'
                        'Time {:.3f} ({:.3f})\t'
                        'Data {:.3f} ({:.3f})\t'
                        .format(tag, rank, i, len(data_loader), batchsize,
                                batch_time.val, batch_time.avg,
                                data_time.val, data_time.avg))
    if not os.path.exists(config.EVAL.SAVE_DIR) and dist.get_rank()==0:
        os.makedirs(config.EVAL.SAVE_DIR)

    np.save(os.path.join(config.EVAL.SAVE_DIR, f'feat_{tag}.npy'), features_all)
    np.save(os.path.join(config.EVAL.SAVE_DIR, f'label_{tag}.npy'), labels_all)

    # print('ss', os.path.join(config.EVAL.SAVE_DIR, f'feat_{tag}_rank{rank}.npy'), features_all.shape)

    # np.save(os.path.join(config.EVAL.SAVE_DIR, f'feat_{tag}_rank{rank}.npy'), features_all)
    # np.save(os.path.join(config.EVAL.SAVE_DIR, f'label_{tag}_rank{rank}.npy'), labels_all)

@torch.no_grad()
def extract_features_ddp(model, data_loader, tag, logger, config):
    batch_time = AverageMeter('Process Time', ':6.3f')
    data_time = AverageMeter('Test Data Time', ':6.3f')

    rank = dist.get_rank()
    end = time.time()
    feats_result = None
    labels_result = None
    ngpus= dist.get_world_size()
    print(f'device {rank} start to extract features')
    for i, (images, index,labels) in enumerate(data_loader):
        index = index.cuda(non_blocking=True)
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True).float()
        # print(i, rank, len(data_loader.dataset), images.shape, images[0,0,0,:3])

        # with torch.cuda.amp.autocast(enabled=config.USE_AMP):
        feat = model(images).cuda(non_blocking=True)

        if rank == 0 and feats_result is None:
            feats_result = torch.zeros(len(data_loader.dataset), feat.shape[-1])
            feats_result = feats_result.cuda(non_blocking=True)
            labels_result = torch.zeros(len(data_loader.dataset))
            labels_result = labels_result.cuda(non_blocking=True)

        batchsize = labels.size(0)

        batch_time.update(time.time() - end)
        end = time.time()
        if i % config.EVAL.PRINT_FREQ == 0 and rank == 0:
            logger.info('Extract {} Features on rank {}: [{}/{}/{}]\t'
                        'Time {:.3f} ({:.3f})\t'
                        'Data {:.3f} ({:.3f})\t'
                        .format(tag, rank, i, len(data_loader), batchsize,
                                batch_time.val, batch_time.avg,
                                data_time.val, data_time.avg))
        y_all = torch.empty(ngpus, index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = dist.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)
        feat_all = torch.empty(
            ngpus,
            feat.size(0),
            feat.size(1),
            dtype=feat.dtype,
            device=feat.device,
        )
        label_all = torch.empty(
            ngpus,
            labels.size(0),
            dtype=labels.dtype,
            device=labels.device,
        )
        feat_l = list(feat_all.unbind(0))
        label_l = list(label_all.unbind(0))
        output_all_reduce_feat = dist.all_gather(feat_l, feat, async_op=True)
        output_all_reduce_label = dist.all_gather(label_l, labels, async_op=True)
        output_all_reduce_feat.wait()
        output_all_reduce_label.wait()
        if rank == 0:
            feats_result.index_copy_(0, index_all, torch.cat(feat_l))
            labels_result.index_copy_(0, index_all, torch.cat(label_l))

    if not os.path.exists(config.EVAL.SAVE_DIR) and rank==0:
        os.makedirs(config.EVAL.SAVE_DIR)

        np.save(os.path.join(config.EVAL.SAVE_DIR, f'feat_{tag}.npy'), feats_result.cpu().numpy())
        np.save(os.path.join(config.EVAL.SAVE_DIR, f'label_{tag}.npy'), labels_result.cpu().numpy())

