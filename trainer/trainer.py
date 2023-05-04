import os
from pathlib import Path
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from evaluate import evaluate_func
from models.margin_softmax import large_margin_module
from utils.util import AverageMeter, tensor_to_float
from torch.utils.tensorboard import SummaryWriter
import json
import numpy as np
class ResultWriter():
    def __init__(self,result_path):
        self.result_path = result_path
        with open(result_path,'w') as f:
            f.write('epoch mAP\n')

    def write(self,epoch,mAP):
        with open(self.result_path,'a') as f:
            f.write(f"{epoch} {mAP}\n")


# LandmarkTrainer
class LandmarkTrainer:
    """
    Trainer class
    """

    def __init__(self, model, train_loader,
                 criterion, optimizer, grad_scaler, config, logger,
                 validation_loader_list=[None, None, None],
                 test_loader_list=[None, None, None],
                 lr_scheduler=None):
        self.config = config
        self.logger = logger
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler

        # cfg_trainer = config['trainer']
        self.epochs = config.TRAIN.EPOCHS #cfg_trainer['epochs']
        self.save_period = config.TRAIN.SAVE_FREQ #cfg_trainer['save_period']
        self.start_epoch = config.TRAIN.START_EPOCH

        self.checkpoint_dir = config.OUTPUT
        self.writer = SummaryWriter(config.OUTPUT)
        self.result_writer = ResultWriter(os.path.join(config.OUTPUT, 'eval_result.txt'))

        self.best_acc1 = 0.0 #self.config.config["best_acc1"]
        self.len_epoch = len(self.train_loader)
        self.device = config.DEVICE

        self.query_loader_public, self.gallery_loader_public, self.query_gts_public = validation_loader_list
        self.query_loader_private, self.gallery_loader_private, self.query_gts_private = test_loader_list

        self.lr_scheduler = lr_scheduler

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            if self.config.DISTRIBUTED:
                self.train_loader.sampler.set_epoch(epoch)
            epoch_time = time.time()
            self._train_epoch(epoch)
            self.lr_scheduler.step()
            
            _model = self.model
            _old_model = None
            
            epoch_time = time.time() - epoch_time
            if not self.config.MULTIPROCESSING_DISTRIBUTED or \
                    (self.config.MULTIPROCESSING_DISTRIBUTED and torch.distributed.get_rank() == 0):
                self.logger.info(f"Epoch {epoch + 1} training takes {epoch_time / 60.0:.2f} minutes")

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if (epoch + 1) % self.config.TRAIN.VAL_FREQ == 0:
                acc1 = evaluate_func(model=_model,
                                     query_loader=self.query_loader_public,
                                     gallery_loader=self.gallery_loader_public,
                                     query_gts=self.query_gts_public,
                                     logger=self.logger,
                                     config=self.config,
                                     old_model=_old_model,
                                     dataset_name=self.config.EVAL.DATASET)
                self.best_acc1 = max(acc1, self.best_acc1)
                self.logger.info(f"best acc: {self.best_acc1:.4f}")
                self.writer.add_scalar("mAP",self.best_acc1,epoch+1)
                self.result_writer.write(epoch+1, self.best_acc1)

            if (epoch + 1) % self.save_period == 0 and dist.get_rank() == 0:
                self.logger.info('==> Saving checkpoint')
                self._save_checkpoint(epoch, _model)



    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        """
        batch_time = AverageMeter('BatchTime', ':6.3f')
        data_time = AverageMeter('DataTime', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses],
            prefix=f"Epoch:[{epoch + 1}/{self.epochs}]  ", logger=self.logger,
        )

        self.model.train()
        end_time = time.time()
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)

            total_steps = epoch * self.len_epoch + batch_idx

            # compute output
            with torch.cuda.amp.autocast(enabled=self.config.USE_AMP):
                images, labels = images.to(self.device), labels.to(self.device)
                feat, cls_score = self.model(images.cuda())
                loss = self.criterion['base'](cls_score, labels)
                # print('image', dist.get_rank(), epoch, batch_idx, len(self.train_loader), images.shape, labels.shape, loss)
            self.writer.add_scalar("Loss", losses.avg, total_steps)
            lr = self.optimizer.param_groups[0]['lr']

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            losses.update(loss.item(), images.size(0))
            # grad_scaler can handle the case that use_amp=False indicated in the official pytorch doc
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if batch_idx % self.config.TRAIN.PRINT_FREQ == 0:
                progress.display(batch_idx, suffix=f"\tlr:{self.optimizer.param_groups[0]['lr']:.6f}")
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], total_steps)

            # if batch_idx == 2:
            #     break
           
    def _save_checkpoint(self, epoch, _model):
        """
        Saving checkpoints

        :param epoch: current epoch number
        """
        arch = type(self.model).__name__
        checkpoint = {
            'epoch': epoch + 1,
            'arch': arch,
            'model': _model.module.state_dict(),
            'best_acc1': self.best_acc1,
            'optimizer': self.optimizer.state_dict(),
            'grad_scaler': self.grad_scaler.state_dict(),
            'config': self.config
        }
        save_dir = Path(os.path.join(self.checkpoint_dir, "ckpt"))
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f'checkpoint_epoch{epoch + 1}.pth.tar'
        torch.save(checkpoint, ckpt_path)
        self.logger.info("Saving checkpoint: {} ...".format(ckpt_path))

# LandmarkComTrainer
class LandmarkComTrainer:
    """
    Trainer class
    """

    def __init__(self, model, old_model, train_loader,
                 criterion, optimizer, grad_scaler, config, logger,
                 validation_loader_list=[None, None, None],
                 test_loader_list=[None, None, None],
                 lr_scheduler=None,K=0):
        self.config = config
        self.logger = logger
        self.old_model = old_model
        self.old_model.eval()
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler

        self.epochs = config.TRAIN.EPOCHS
        self.save_period = config.TRAIN.SAVE_FREQ
        self.start_epoch = config.TRAIN.START_EPOCH

        self.checkpoint_dir = config.OUTPUT
        self.writer = SummaryWriter(config.OUTPUT)

        self.best_acc_cross = 0.0
        self.best_acc_self = 0.0
        self.len_epoch = len(self.train_loader)
        self.device = config.DEVICE

        self.query_loader_public, self.gallery_loader_public, self.query_gts_public = validation_loader_list
        self.query_loader_private, self.gallery_loader_private, self.query_gts_private = test_loader_list

        self.lr_scheduler = lr_scheduler
        # for lce/ours
        self.old_meta = self.load_old_meta()
        # for adv training
        self.model_criterion = torch.nn.NLLLoss()

        self.K = K
        self.logger.info(f'K value is {self.K}')
        # for uniBCT
        if self.config.COMP_LOSS.TYPE == 'uni_bct':
            self.old_centers = self.load_old_centers()

    def load_old_centers(self):
        self.logger.info(f'start to load metadata from {self.config.EVAL.OLD_SAVE_FILE}')
        path = self.config.EVAL.OLD_SAVE_FILE
        if not os.path.exists(path):
            self.logger.warn(f'{self.config.EVAL.OLD_SAVE_FILE} doesnt exist')
            return None
        with open(path, 'r') as f:
            old_meta = json.load(f)
        centers = []
        for i in range(self.config.OLD_MODEL.NUM_CLASSES):
            centers.append(old_meta[str(i)]['center'])
        centers = torch.tensor(centers).float()
        self.logger.info(f'centers shape: {centers.shape}')
        return centers

    def load_old_meta(self):
        self.logger.info(f'start to load metadata from {self.config.EVAL.OLD_SAVE_FILE}')
        path = self.config.EVAL.OLD_SAVE_FILE
        if not os.path.exists(path):
            self.logger.warn(f'{self.config.EVAL.OLD_SAVE_FILE} doesnt exist, which is required by some models')
            return None
        else:
            with open(path, 'r') as f:
                old_meta = json.load(f)
            return old_meta

    def train(self):
        _model = self.model
        _old_model = self.old_model
        self.best_acc_self = evaluate_func(model=_old_model,
                                           query_loader=self.query_loader_public,
                                           gallery_loader=self.gallery_loader_public,
                                           query_gts=self.query_gts_public,
                                           logger=self.logger,
                                           config=self.config,
                                           dataset_name=self.config.EVAL.DATASET)
        self.logger.info(f"best acc of old-model : {self.best_acc_self:.4f}")
        self.best_acc_cross = evaluate_func(model=_model,
                                            query_loader=self.query_loader_public,
                                            gallery_loader=self.gallery_loader_public,
                                            query_gts=self.query_gts_public,
                                            logger=self.logger,
                                            config=self.config,
                                            old_model=_old_model,
                                            dataset_name=self.config.EVAL.DATASET)
        self.logger.info(f"best acc of cross-model : {self.best_acc_cross:.4f}")
        self.best_acc_self = evaluate_func(model=_model,
                                           query_loader=self.query_loader_public,
                                           gallery_loader=self.gallery_loader_public,
                                           query_gts=self.query_gts_public,
                                           logger=self.logger,
                                           config=self.config,
                                           dataset_name=self.config.EVAL.DATASET)
        self.logger.info(f"best acc of self-model : {self.best_acc_self:.4f}")
        for epoch in range(self.start_epoch, self.epochs):
            epoch_time = time.time()

            if self.config.DISTRIBUTED:
                self.train_loader.sampler.set_epoch(epoch)

            self._back_comp_train_epoch(epoch,self.epochs)

            self.lr_scheduler.step()

            _model = self.model
            _old_model = self.old_model

            epoch_time = time.time() - epoch_time
            if not self.config.MULTIPROCESSING_DISTRIBUTED or \
                    (self.config.MULTIPROCESSING_DISTRIBUTED and torch.distributed.get_rank() == 0):
                self.logger.info(f"Epoch {epoch + 1} training takes {epoch_time / 60.0:.2f} minutes")

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if (epoch + 1) % self.config.TRAIN.VAL_FREQ == 0:
                acc1 = evaluate_func(model=_model,
                                     query_loader=self.query_loader_public,
                                     gallery_loader=self.gallery_loader_public,
                                     query_gts=self.query_gts_public,
                                     logger=self.logger,
                                     config=self.config,
                                     old_model=_old_model,
                                     dataset_name=self.config.EVAL.DATASET)
                self.best_acc_cross = max(acc1, self.best_acc_cross)
                self.logger.info(f"best acc of cross-model : {self.best_acc_cross:.4f}")

                acc1 = evaluate_func(model=_model,
                                     query_loader=self.query_loader_public,
                                     gallery_loader=self.gallery_loader_public,
                                     query_gts=self.query_gts_public,
                                     logger=self.logger,
                                     config=self.config,
                                     dataset_name=self.config.EVAL.DATASET)
                self.best_acc_self = max(acc1, self.best_acc_self)
                self.logger.info(f"best acc of self-model : {self.best_acc_self:.4f}")

            if (epoch + 1) % self.save_period == 0:
                self.logger.info('==> Saving checkpoint')
                self._save_checkpoint(epoch, _model)


    def _back_comp_train_epoch(self, epoch,n_epoch=0):
        batch_time = AverageMeter('BatchTime', ':6.3f')
        data_time = AverageMeter('DataTime', ':6.3f')
        losses_cls = AverageMeter('Cls Loss', ':.4f')

        losses_all = AverageMeter('Total Loss', ':.4f')
        meter_list = [batch_time, data_time, losses_all, losses_cls]
        if self.config.TRAIN.TYPE == 'adv_compatible':
            losses_model_cls = AverageMeter('Model Cls Loss', ':.4f')
            meter_list.append(losses_model_cls)

        if self.config.UPGRADE_LOSS.TYPE == 'center_limit':
            losses_limit_upgrade = AverageMeter('Upgrade limited Loss', ':.4f')
            meter_list.append(losses_limit_upgrade)

        if self.config.COMP_LOSS.TYPE in ['bct_limit','bct_limit_no_s2c']:
            losses_limit_comp = AverageMeter('Comp bct_limit Loss', ':.4f')
            meter_list.append(losses_limit_comp)

        if self.config.COMP_LOSS.TYPE == 'uni_bct':
            losses_uni_comp = AverageMeter('Comp unibct Loss', ':.4f')
            meter_list.append(losses_uni_comp)

        if self.config.COMP_LOSS.TYPE == 'hot_refresh':
            losses_hotrefresh_comp = AverageMeter('Comp hot_refresh Loss', ':.4f')
            meter_list.append(losses_hotrefresh_comp)

        if self.config.COMP_LOSS.TYPE == 'bct':
            losses_bct_comp = AverageMeter('Comp bct Loss', ':.4f')
            meter_list.append(losses_bct_comp)

        if self.config.COMP_LOSS.TYPE == 'lce':
            losses_boundary = AverageMeter('Boundary Loss', ':.4f')
            losses_alignment = AverageMeter('Alignment Loss', ':.4f')
            meter_list.append(losses_boundary)
            meter_list.append(losses_alignment)

        progress = ProgressMeter(
            len(self.train_loader),
            meter_list,
            prefix=f"Epoch:[{epoch + 1}/{self.epochs}]  ", logger=self.logger,
        )

        self.model.train()
        end_time = time.time()
        len_dataloader = len(self.train_loader)

        if self.config.UPGRADE_LOSS.TYPE == 'others':
            for name, param in self.model.named_parameters():
                if 'projection_cls' in name:
                    param.requires_grad = False
        for batch_idx, (images, labels) in enumerate(self.train_loader):

            #for adv training
            p = float(batch_idx + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            data_time.update(time.time() - end_time)
            total_steps = epoch * self.len_epoch + batch_idx

            # compute output
            with torch.cuda.amp.autocast(enabled=self.config.USE_AMP):
                images, labels = images.to(self.device), labels.to(self.device)
                if self.config.COMP_LOSS.TYPE == 'lce':
                    # add a transformation head to the old model.
                    # Follow the protocol of LCE, K=0
                    center_olds = torch.zeros([len(images), 256]).to(self.device)
                    theta_olds = torch.zeros([len(images)]).to(self.device)
                    with torch.no_grad():
                        for j in range(len(images)):
                            if str(labels[j].item()) in self.old_meta:
                                c_o = torch.tensor(self.old_meta[str(labels[j].item())]['center']).to(self.device)
                                center_olds[j] = c_o
                                if len(self.old_meta[str(labels[j].item())]['radius']) > 1:
                                    theta_olds[j] = (self.old_meta[str(labels[j].item())]['radius'][-1])
                                else:
                                    theta_olds[j] = 0
                    if self.K != 0:
                        feat_old, center_olds = self.old_model(images.cuda(), center_olds)
                    else:
                        feat_old = self.old_model(images.cuda()).detach()
                    center_olds = F.normalize(center_olds)
                else:
                    with torch.no_grad():
                        feat_old = self.old_model(images.cuda()).detach()

                if self.config.COMP_LOSS.ELASTIC_BOUNDARY:
                    radius_eb = torch.zeros(self.config.NEW_MODEL.NUM_CLASSES)
                    for j in range(len(labels)):
                        if str(labels[j].item()) in self.old_meta:
                            radius_max = self.old_meta[str(labels[j].item())]['radius'][-1]
                            radius_eb[labels[j].item()] = abs(radius_max - self.config.TRAIN.THRESHOLD)
                        else:
                            radius_eb[labels[j].item()] = 0
                if self.config.TRAIN.TYPE == 'adv_compatible':
                    # classification label of model embeddings
                    model_label_new = torch.zeros(len(labels))
                    model_label_new = model_label_new.long()
                    model_label_old = torch.ones(len(labels))
                    model_label_old = model_label_old.long()
                    model_label_old = model_label_old.cuda()
                    model_label_new = model_label_new.cuda()
                    if self.config.COMP_LOSS.ELASTIC_BOUNDARY:
                        feat, cls_score1, model_score_new,model_score_old,radius_eb = self.model(images.cuda(),feat_old,alpha,radius_eb)
                        # print(radius_eb[radius_eb!=0])
                    else:
                        feat, cls_score1, model_score_new, model_score_old = self.model(images.cuda(),feat_old, alpha)
                    model_loss = self.model_criterion(model_score_new, model_label_new) + self.model_criterion(
                        model_score_old, model_label_old)
                    model_loss_value = tensor_to_float(model_loss)
                    losses_model_cls.update(model_loss_value, len(labels))
                    self.writer.add_scalar("Model Cls Loss", losses_model_cls.avg, total_steps)
                else:
                    if self.config.COMP_LOSS.ELASTIC_BOUNDARY:
                        feat, cls_score1, radius_eb = self.model(images.cuda(),radius=radius_eb)
                    else:
                        feat, cls_score1 = self.model(images.cuda())

                # upgrade loss
                if self.config.LOSS.TYPE == "softmax":
                    cls_score = self.model.module.projection_cls(feat)
                else:
                    cls_score = F.linear(F.normalize(feat), F.normalize(self.model.module.projection_cls.weight))
                    cls_score = large_margin_module(self.config.LOSS.TYPE, cls_score, labels,
                                                    s=self.config.LOSS.SCALE,
                                                    m=self.config.LOSS.MARGIN)
                loss = self.criterion['base'](cls_score, labels)


                # original BCT loss
                # from paper "Towards backward-compatible representation learning", CVPR 2020
                if self.config.COMP_LOSS.TYPE in ['bct','bct_ract']:
                    labels_bct = []
                    counter = 0
                    for j in range(len(images)):
                        if labels[j].item() < self.config.OLD_MODEL.NUM_CLASSES:
                            if counter == 0:
                                feats_bct = feat[j][None,]
                            else:
                                feats_bct = torch.cat((feats_bct,feat[j][None,:]),0)
                            labels_bct.append(labels[j].item())
                            counter += 1

                    n2o_cls_score = self.old_model.projection_cls(feats_bct)
                    self.writer.add_scalar("Comp bct loss", losses_bct_comp.avg, total_steps)

                if self.config.COMP_LOSS.TYPE == 'bct':
                    loss_back_comp = self.criterion['back_comp'](n2o_cls_score, torch.tensor(labels_bct).long().cuda())
                    loss_back_comp_value = tensor_to_float(loss_back_comp)
                    losses_bct_comp.update(loss_back_comp_value, len(labels))
                    self.writer.add_scalar("Comp bct loss", losses_bct_comp.avg, total_steps)
                elif self.config.COMP_LOSS.TYPE == 'bct_ract':
                    masks = F.one_hot(labels, num_classes=cls_score.size(1))
                    masked_cls_score = cls_score - masks * 1e9
                    concat_cls_score = torch.cat((n2o_cls_score, masked_cls_score), 1)
                    loss_back_comp = self.criterion['back_comp'](concat_cls_score, labels)
                    loss_back_comp_value = tensor_to_float(loss_back_comp)
                    losses_bct_comp.update(loss_back_comp_value, len(labels))
                # our compatible loss
                elif self.config.COMP_LOSS.TYPE == 'bct_limit':
                    limit_loss = 0.  # new feat is in old range
                    feat = F.normalize(feat)
                    cnt = 0
                    for j in range(len(feat)):
                        if str(labels[j].item()) in self.old_meta:
                            # diff = feat[j] - F.normalize(torch.tensor(self.old_meta[str(labels[j].item())]['center'])[None,:]).to(feat.device)
                            diff = feat[j] - torch.tensor(self.old_meta[str(labels[j].item())]['center'])[None, :].to(
                                feat.device)
                            if self.config.COMP_LOSS.ELASTIC_BOUNDARY:
                                if self.old_meta[str(labels[j].item())]['radius'][-1] < self.config.TRAIN.THRESHOLD:
                                    radius = self.old_meta[str(labels[j].item())]['radius'][-1] + radius_eb[
                                        labels[j].item()]
                                    if batch_idx % self.config.TRAIN.PRINT_FREQ == 0 and j == 0:
                                        self.logger.info(
                                            f'original {self.old_meta[str(labels[j].item())]["radius"][-1]},+ {radius_eb[labels[j].item()].item()}, radius {radius.item()}')
                                else:
                                    radius = self.old_meta[str(labels[j].item())]['radius'][-1] - radius_eb[
                                        labels[j].item()]
                                    if batch_idx % self.config.TRAIN.PRINT_FREQ == 0 and j == 0:
                                        self.logger.info(
                                            f'original {self.old_meta[str(labels[j].item())]["radius"][-1]},- {radius_eb[labels[j].item()].item()}, radius {radius.item()}')

                            else:
                                idx = len(self.old_meta[str(labels[j].item())]['radius']) // 2  # use median
                                radius = self.old_meta[str(labels[j].item())]['radius'][0]
                            if len(self.old_meta[str(labels[j].item())]['radius']) <= 1:
                                # radius = 0.5
                                continue
                            tmp = max(torch.norm(diff, p=2) - radius, 0)
                            if tmp > 0.:
                                limit_loss += tmp
                                cnt += 1
                    if cnt:
                        limit_loss /= cnt
                        # print(f'{limit_loss}')
                    loss_back_comp = limit_loss
                    limit_loss_value = tensor_to_float(limit_loss)
                    losses_limit_comp.update(limit_loss_value, len(labels))
                    self.writer.add_scalar("Comp ours", losses_limit_comp.avg, total_steps)
                elif self.config.COMP_LOSS.TYPE == 'bct_limit_no_s2c':
                    limit_loss = 0.  # new feat is in old range
                    cnt = 0
                    # compatible loss
                    f_new = F.normalize(feat)
                    f_old = F.normalize(feat_old)
                    for j in range(len(feat)):
                        if str(labels[j].item()) in self.old_meta:
                            center_old = F.normalize(torch.tensor(self.old_meta[str(labels[j].item())]['center'])[None,:]).to(feat.device)
                            diff_new = torch.norm(f_new[j] - center_old, p=2)
                            diff_old = torch.norm(f_old[j] - center_old, p=2)
                            tmp = max(diff_new - diff_old, 0)
                            if tmp > 0.:
                                limit_loss += tmp
                                cnt += 1
                    if cnt:
                        limit_loss /= cnt
                    loss_back_comp = self.criterion['back_comp'](feat, feat_old, labels) + limit_loss
                    limit_loss_value = tensor_to_float(limit_loss)
                    losses_limit_comp.update(limit_loss_value, len(labels))
                    self.writer.add_scalar("Comp ours", losses_limit_comp.avg, total_steps)
                elif self.config.COMP_LOSS.TYPE == 'lce':
                    # boundary_loss
                    boundary_loss = 0
                    feat = F.normalize(feat)
                    cnt = 0
                    for j in range(len(center_olds)):
                        if theta_olds[j] == 0:
                            continue
                        boundary_loss += max(torch.acos(torch.dot(center_olds[j], feat[j])) - theta_olds[j], 0)
                        cnt += 1
                    boundary_loss /= cnt
                    # alignment loss
                    alignment_loss = 0
                    cnt = 0
                    for i, j in enumerate(labels):
                        if theta_olds[i] == 0:
                            continue
                        d_center = (1 - torch.dot(center_olds[i], F.normalize(self.model.module.projection_cls.weight[j, :][None, :])[0]))
                        alignment_loss += d_center
                        cnt += 1
                    alignment_loss = alignment_loss / cnt
                    loss_back_comp = boundary_loss + alignment_loss
                    alignment_loss_value = tensor_to_float(alignment_loss)
                    boundary_loss_value = tensor_to_float(boundary_loss)
                    losses_alignment.update(alignment_loss_value, images.size(0))
                    losses_boundary.update(boundary_loss_value, images.size(0))
                    self.writer.add_scalar("Comp lce alignment loss", losses_alignment.avg, total_steps)
                    self.writer.add_scalar("Comp lce boundaryloss", losses_boundary.avg, total_steps)
                elif self.config.COMP_LOSS.TYPE == 'uni_bct':
                    labels_bct = []
                    counter = 0
                    for j in range(len(images)):
                        if labels[j].item() < self.config.OLD_MODEL.NUM_CLASSES:
                            if counter == 0:
                                feats_bct = feat[j][None,]
                            else:
                                feats_bct = torch.cat((feats_bct, feat[j][None, :]), 0)
                            labels_bct.append(labels[j].item())
                            counter += 1
                    labels_bct = torch.tensor(labels_bct).cuda()
                    cls_score = F.linear(F.normalize(feats_bct), F.normalize(self.old_centers).to(feat.device))
                    cls_score = large_margin_module('arcface', cls_score, labels_bct,
                                                    s=self.config.LOSS.SCALE,
                                                    m=self.config.LOSS.MARGIN)
                    loss_back_comp = self.criterion['base'](cls_score, labels_bct)
                    losses_uni_comp.update(loss_back_comp.item(), images.size(0))
                elif self.config.COMP_LOSS.TYPE == 'hot_refresh':
                    loss_back_comp = self.criterion['back_comp'](feat, feat_old, labels)
                    loss_back_comp_value = tensor_to_float(loss_back_comp)
                    losses_hotrefresh_comp.update(loss_back_comp_value, len(labels))
                    self.writer.add_scalar("Comp hot_refresh loss", losses_hotrefresh_comp.avg, total_steps)
                else:
                    raise NotImplementedError("Unknown backward compatible loss type")


            losses_cls.update(loss.item(), images.size(0))
            self.writer.add_scalar("Cls loss", losses_cls.avg, total_steps)

            if self.config.TRAIN.TYPE == 'adv_compatible':
                loss = loss + loss_back_comp + model_loss*(n_epoch-epoch+1)/n_epoch
            else:
                loss = loss + loss_back_comp

            losses_all.update(loss.item(), images.size(0))
            self.writer.add_scalar("Total loss", losses_all.avg, total_steps)
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if batch_idx % self.config.TRAIN.PRINT_FREQ == 0:
                progress.display(batch_idx, suffix=f"\tlr:{self.optimizer.param_groups[0]['lr']:.6f}")
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], total_steps)


    def _save_checkpoint(self, epoch, _model):
        """
        Saving checkpoints

        :param epoch: current epoch number
        """
        arch = type(self.model).__name__
        checkpoint = {
            'epoch': epoch + 1,
            'arch': arch,
            'model': _model.module.state_dict(),
            'best_acc_cross': self.best_acc_cross,
            'best_acc_self': self.best_acc_self,
            'optimizer': self.optimizer.state_dict(),
            'grad_scaler': self.grad_scaler.state_dict(),
            'config': self.config
        }
        save_dir = Path(os.path.join(self.checkpoint_dir, "ckpt"))
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f'checkpoint_epoch{epoch + 1}.pth.tar'
        torch.save(checkpoint, ckpt_path)
        self.logger.info("Saving checkpoint: {} ...".format(ckpt_path))

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch, suffix=''):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries) + suffix)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return 'Iter:[' + fmt + '/' + fmt.format(num_batches) + ']'
