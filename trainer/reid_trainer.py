import os
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
# from evaluate.evaluate import evaluate_func
from evaluate import evaluate_func
from models.margin_softmax import large_margin_module
from utils.util import AverageMeter, tensor_to_float
from torch.utils.tensorboard import SummaryWriter
from loss.reid import make_loss
from evaluate.metric import R1_mAP_eval


def make_optimizer(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.TRAIN.LR_SCHEDULER.BASE_LR
        weight_decay = cfg.TRAIN.LR_SCHEDULER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.TRAIN.LR_SCHEDULER.BASE_LR * cfg.TRAIN.LR_SCHEDULER.BIAS_LR_FACTOR
            weight_decay = cfg.TRAIN.LR_SCHEDULER.WEIGHT_DECAY_BIAS
        if cfg.TRAIN.LR_SCHEDULER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.TRAIN.LR_SCHEDULER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = getattr(torch.optim, 'SGD')(params, momentum=cfg.TRAIN.LR_SCHEDULER.MOMENTUM)

    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.TRAIN.LR_SCHEDULER.CENTER_LR)

    return optimizer, optimizer_center


class ResultWriter():
    def __init__(self, result_path):
        self.result_path = result_path
        with open(result_path, 'w') as f:
            f.write('epoch mAP\n')

    def write(self, epoch, mAP):
        with open(self.result_path, 'a') as f:
            f.write(f"{epoch} {mAP:.4f}\n")


# TODO BASETrainer

# ReidTrainer
class ReidTrainer:
    """
    Trainer class
    """

    def __init__(self, model, train_loader, val_loader,
                 grad_scaler, config, logger, lr_scheduler, num_classes, num_query):
        self.loss_fn, self.center_criterion = make_loss(config, num_classes=num_classes)
        self.optimizer, self.optimizer_center = make_optimizer(config, model, self.center_criterion)
        self.config = config
        self.logger = logger
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grad_scaler = grad_scaler

        # cfg_trainer = config['trainer']
        self.epochs = config.TRAIN.EPOCHS  # cfg_trainer['epochs']
        self.save_period = config.TRAIN.SAVE_FREQ  # cfg_trainer['save_period']
        self.start_epoch = config.TRAIN.START_EPOCH

        self.checkpoint_dir = config.OUTPUT
        self.writer = SummaryWriter(config.OUTPUT)
        self.result_writer = ResultWriter(os.path.join(config.OUTPUT, 'eval_result.txt'))

        self.mAP = 0.0  # self.config.config["best_acc1"]
        self.len_epoch = len(self.train_loader)
        self.device = config.DEVICE

        self.lr_scheduler = lr_scheduler

        self.evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=True)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            epoch_time = time.time()
            self.evaluator.reset()
            self._train_epoch(epoch)
            self.lr_scheduler.step(epoch)

            _model = self.model
            _old_model = None

            epoch_time = time.time() - epoch_time
            if not self.config.MULTIPROCESSING_DISTRIBUTED or \
                    (self.config.MULTIPROCESSING_DISTRIBUTED and torch.distributed.get_rank() == 0):
                self.logger.info(f"Epoch {epoch + 1} training takes {epoch_time / 60.0:.2f} minutes")

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if (epoch + 1) % self.config.TRAIN.VAL_FREQ == 0:
                if dist.get_rank() == 0:
                    self.model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(self.val_loader):
                        with torch.no_grad():
                            img = img.to(self.device)
                            camids = camids.to(self.device)
                            target_view = target_view.to(self.device)
                            feat = self.model(img)
                            self.evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = self.evaluator.compute()
                    self.mAP = max(mAP, self.mAP)
                    self.logger.info(f"mAP: {self.mAP:.4f}")
                    self.writer.add_scalar("mAP", self.mAP, epoch + 1)
                    self.result_writer.write(epoch + 1, self.mAP)
                    for r in [1, 5, 10]:
                        self.logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()

            if (epoch + 1) % self.save_period == 0:
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
        for n_iter, (img, vid, target_cam, target_view) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)

            total_steps = epoch * self.len_epoch + n_iter

            self.optimizer.zero_grad()
            self.optimizer_center.zero_grad()
            img = img.to(self.device)
            target = vid.to(self.device)
            target_cam = target_cam.to(self.device)
            target_view = target_view.to(self.device)
            # compute output
            with torch.cuda.amp.autocast(enabled=self.config.USE_AMP):
                score, feat = self.model(img)
                loss = self.loss_fn(score, feat, target)
                # print('image', dist.get_rank(), epoch, batch_idx, len(self.train_loader), images.shape, labels.shape, loss)
            self.writer.add_scalar("Loss", losses.avg, total_steps)
            lr = self.optimizer.param_groups[0]['lr']

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            losses.update(loss.item(), img.size(0))
            # grad_scaler can handle the case that use_amp=False indicated in the official pytorch doc
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if 'center' in self.config.TRAIN.METRIC_LOSS_TYPE:
                for param in self.center_criterion.parameters():
                    param.grad.data *= (1. / self.config.TRAIN.CENTER_LOSS_WEIGHT)
                self.grad_scaler.step(self.optimizer_center)
                self.grad_scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            if n_iter % self.config.TRAIN.PRINT_FREQ == 0:
                progress.display(n_iter, suffix=f"\tlr:{self.optimizer.param_groups[0]['lr']:.6f}")
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], total_steps)
            self.writer.add_scalar("acc", acc, total_steps)

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
            'best_acc1': self.mAP,
            'optimizer': self.optimizer.state_dict(),
            'grad_scaler': self.grad_scaler.state_dict(),
            'config': self.config
        }
        save_dir = Path(os.path.join(self.checkpoint_dir, "ckpt"))
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f'checkpoint_epoch{epoch + 1}.pth.tar'
        torch.save(checkpoint, ckpt_path)
        self.logger.info("Saving checkpoint: {} ...".format(ckpt_path))


# ReidComTrainer
class ReidComTrainer:
    """
    Trainer class
    """

    def __init__(self, model, old_model, train_loader, val_loader, criterion,
                 grad_scaler, config, logger, lr_scheduler, num_classes, num_query):
        self.loss_fn, self.center_criterion = make_loss(config, num_classes=num_classes)
        self.optimizer, self.optimizer_center = make_optimizer(config, model, self.center_criterion)
        self.config = config
        self.logger = logger
        self.model = model
        self.old_model = old_model
        self.old_model.eval()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grad_scaler = grad_scaler
        self.criterion = criterion
        self.criterion['base'] = self.loss_fn

        # cfg_trainer = config['trainer']
        self.epochs = config.TRAIN.EPOCHS  # cfg_trainer['epochs']
        self.save_period = config.TRAIN.SAVE_FREQ  # cfg_trainer['save_period']
        self.start_epoch = config.TRAIN.START_EPOCH

        self.checkpoint_dir = config.OUTPUT
        self.writer = SummaryWriter(config.OUTPUT)
        self.result_writer = ResultWriter(os.path.join(config.OUTPUT, 'eval_result.txt'))

        self.mAP = 0.0  # self.config.config["best_acc1"]
        self.mAP_comp = 0.0  # self.config.config["best_acc1"]
        self.len_epoch = len(self.train_loader)
        self.device = config.DEVICE

        self.lr_scheduler = lr_scheduler

        self.evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=True)
        self.lr_scheduler = lr_scheduler

    def train(self):
        _model = self.model
        _old_model = self.old_model
        self.evaluator.reset()
        for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(self.val_loader):
            with torch.no_grad():
                img = img.to(self.device)
                feat = self.old_model(img)
                self.evaluator.update((feat, vid, camid))
                self.evaluator.update_gallery()

        for epoch in range(self.start_epoch, self.epochs):
            self.evaluator.reset()
            epoch_time = time.time()

            self._back_comp_train_epoch(epoch)

            self.lr_scheduler.step(epoch)

            _model = self.model
            _old_model = self.old_model

            epoch_time = time.time() - epoch_time
            if not self.config.MULTIPROCESSING_DISTRIBUTED or \
                    (self.config.MULTIPROCESSING_DISTRIBUTED and torch.distributed.get_rank() == 0):
                self.logger.info(f"Epoch {epoch + 1} training takes {epoch_time / 60.0:.2f} minutes")

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if (epoch + 1) % self.config.TRAIN.VAL_FREQ == 0:
                if dist.get_rank() == 0:
                    self.model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(self.val_loader):
                        with torch.no_grad():
                            img = img.to(self.device)
                            camids = camids.to(self.device)
                            target_view = target_view.to(self.device)
                            feat = self.model(img)
                            self.evaluator.update((feat, vid, camid))
                    cmc, mAP, cmc_comp, mAP_comp = self.evaluator.compute(comp=True)
                    self.mAP = max(mAP, self.mAP)
                    self.mAP_comp = max(mAP_comp, self.mAP_comp)
                    self.logger.info(f"mAP: {self.mAP:.4f}")
                    self.logger.info(f"mAP_comp: {self.mAP_comp:.4f}")
                    self.writer.add_scalar("mAP", self.mAP, epoch + 1)
                    self.writer.add_scalar("mAP_comp", self.mAP_comp, epoch + 1)
                    self.result_writer.write(epoch + 1, self.mAP)
                    for r in [1, 5, 10]:
                        self.logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    for r in [1, 5, 10]:
                        self.logger.info("comp CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_comp[r - 1]))
                    torch.cuda.empty_cache()

            if (epoch + 1) % self.save_period == 0:
                self.logger.info('==> Saving checkpoint')
                self._save_checkpoint(epoch, _model)

    def _back_comp_train_epoch(self, epoch):
        batch_time = AverageMeter('BatchTime', ':6.3f')
        data_time = AverageMeter('DataTime', ':6.3f')
        losses_cls = AverageMeter('Cls Loss', ':.4f')
        losses_back_comp = AverageMeter('Backward Comp Loss', ':.4f')
        losses_all = AverageMeter('Total Loss', ':.4f')
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses_all, losses_cls, losses_back_comp],
            prefix=f"Epoch:[{epoch + 1}/{self.epochs}]  ", logger=self.logger,
        )
        self.model.train()
        end_time = time.time()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)

            total_steps = epoch * self.len_epoch + n_iter

            # compute output
            img = img.to(self.device)
            target = vid.to(self.device)
            target_cam = target_cam.to(self.device)
            target_view = target_view.to(self.device)
            # compute output
            with torch.cuda.amp.autocast(enabled=self.config.USE_AMP):
                score, feat = self.model(img)
                with torch.no_grad():
                    feat_old = self.old_model(img).detach()

                loss = self.criterion['base'](score, feat, target)

                n2o_cls_score = self.old_model.projection_cls(feat)

                # Point2center backward compatible loss (original BCT loss),
                # from paper "Towards backward-compatible representation learning", CVPR 2020
                if self.config.COMP_LOSS.TYPE == 'bct':
                    loss_back_comp = self.criterion['back_comp'](n2o_cls_score, target)
                elif self.config.COMP_LOSS.TYPE == 'bct_ract':
                    masks = F.one_hot(target, num_classes=score.size(1))
                    masked_cls_score = score - masks * 1e9
                    concat_cls_score = torch.cat((n2o_cls_score, masked_cls_score), 1)
                    loss_back_comp = self.criterion['back_comp'](concat_cls_score, score)
                else:
                    if self.config.COMP_LOSS.TYPE == 'lwf':
                        old_cls_score = self.old_model.projection_cls(feat_old)
                        old_cls_score = F.softmax(old_cls_score / self.comfig.COMP_LOSS.DISTILLATION_TEMP, dim=1)
                        loss_back_comp = -torch.sum(
                            F.log_softmax(n2o_cls_score / self.comfig.COMP_LOSS.TEMPERATURE) * old_cls_score) \
                                         / img.size(0)
                    elif self.config.COMP_LOSS.TYPE == 'fd':
                        criterion_mse = nn.MSELoss(reduce=False).cuda(self.device)
                        loss_back_comp = torch.mean(criterion_mse(feat, feat_old), dim=1)
                        predicted_target = score.argmax(dim=1)
                        bool_target_is_match = (predicted_target == target)
                        focal_weight = self.args.comp_loss["focal_beta"] * bool_target_is_match + self.args.comp_loss[
                            "focal_alpha"]
                        loss_back_comp = torch.mul(loss_back_comp, focal_weight).mean()
                    elif self.config.COMP_LOSS.TYPE in ['contra', 'triplet', 'l2', 'hot_refresh', 'triplet_ract']:
                        loss_back_comp = self.criterion['back_comp'](feat, feat_old, target)
                    else:
                        raise NotImplementedError("Unknown backward compatible loss type")

            losses_cls.update(loss.item(), img.size(0))
            self.writer.add_scalar("Cls loss", losses_cls.avg, total_steps)

            loss_back_comp_value = tensor_to_float(loss_back_comp)
            losses_back_comp.update(loss_back_comp_value, len(target))
            loss = loss + loss_back_comp
            self.writer.add_scalar("Comp loss", losses_back_comp.avg, total_steps)

            losses_all.update(loss.item(), img.size(0))
            self.writer.add_scalar("Total loss", losses_all.avg, total_steps)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if n_iter % self.config.TRAIN.PRINT_FREQ == 0:
                progress.display(n_iter, suffix=f"\tlr:{self.optimizer.param_groups[0]['lr']:.6f}")
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
            'mAP': self.mAP,
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
