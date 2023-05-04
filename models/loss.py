import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.losses import CrossBatchMemory
from pytorch_metric_learning.utils.module_with_records import ModuleWithRecords
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
__all__ = ['BackwardCompatibleLoss','UpgradeLoss']

def get_matches_and_diffs(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels

    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    return matches, diffs

def get_all_pairs_indices(labels, ref_labels=None):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    matches, diffs = get_matches_and_diffs(labels, ref_labels)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx

def mat_based_loss(mat, indices_tuple):
    a1, p, a2, n = indices_tuple
    pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
    pos_mask[a1, p] = 1
    neg_mask[a2, n] = 1
    return pos_mask, neg_mask

def gather_tensor(raw_tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensor_large = [torch.zeros_like(raw_tensor)
                    for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_large, raw_tensor.contiguous())
    tensor_large = torch.cat(tensor_large, dim=0)
    return tensor_large


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    dist_m = dist_m.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist_m


def calculate_loss(feat_new, feat_old, feat_new_large, feat_old_large, targets_large,
                   masks, loss_type, temp, criterion,
                   loss_weight=1.0, topk_neg=-1):
    B, D = feat_new.shape
    indices_tuple = get_all_pairs_indices(targets_large)
    labels_idx = torch.arange(B) + torch.distributed.get_rank() * B
    if feat_new_large is None:
        feat_new_large = feat_new
        feat_old_large = feat_old

    if loss_type == 'contra':
        logits_n2o_pos = torch.bmm(feat_new.view(B, 1, D), feat_old.view(B, D, 1))  # B*1
        logits_n2o_pos = torch.squeeze(logits_n2o_pos, 1)
        logits_n2o_neg = torch.mm(feat_new, feat_old_large.permute(1, 0))  # B*B
        logits_n2o_neg = logits_n2o_neg - masks * 1e9
        if topk_neg > 0:
            logits_n2o_neg = torch.topk(logits_n2o_neg, topk_neg, dim=1)[0]
        logits_all = torch.cat((logits_n2o_pos, logits_n2o_neg), 1)  # B*(1+k)
        logits_all /= temp

        labels_idx = torch.zeros(B).long().cuda()
        loss = criterion(logits_all, labels_idx) * loss_weight
    elif loss_type in ['hot_refresh', 'bct_limit' ,'bct_limit_no_s2c']:
        logits_n2o_pos = torch.bmm(feat_new.view(B, 1, D), feat_old.view(B, D, 1))  # B*1
        logits_n2o_pos = torch.squeeze(logits_n2o_pos, 1)
        logits_n2o_neg = torch.mm(feat_new, feat_old_large.permute(1, 0))  # B*B
        logits_n2o_neg = logits_n2o_neg - masks * 1e9
        logits_n2n_neg = torch.mm(feat_new, feat_new_large.permute(1, 0))  # B*B
        logits_n2n_neg = logits_n2n_neg - masks * 1e9
        if topk_neg > 0:
            logits_n2o_neg = torch.topk(logits_n2o_neg, topk_neg, dim=1)[0]
            logits_n2n_neg = torch.topk(logits_n2n_neg, topk_neg, dim=1)[0]
        logits_all = torch.cat((logits_n2o_pos, logits_n2o_neg, logits_n2n_neg), 1)  # B*(1+2B)
        logits_all /= temp
        labels_idx = torch.zeros(B).long().cuda()
        loss = criterion(logits_all, labels_idx) * loss_weight
    elif loss_type == 'hard_info':
        new_new = torch.mm(feat_new_large, feat_new_large.t())
        new_old = torch.mm(feat_new_large, feat_old_large.t())
        mask_pos, mask_neg = mat_based_loss(new_new, indices_tuple)
        batch_size = new_new.shape[0]
        new_new[new_new==0] = torch.finfo(new_new.dtype).min
        pos_new_new = new_new*mask_pos
        neg_new_new = new_new*mask_neg

        new_old[new_old==0] = torch.finfo(new_old.dtype).min
        mask_pos = mask_pos + torch.eye(batch_size).to(mask_pos.device)
        mask_neg = mask_neg + torch.eye(batch_size).to(mask_neg.device)
        pos_new_old = new_old*mask_pos
        neg_new_old = new_old*mask_neg
        logits_all = []
        for i in range(batch_size):
            pos_item, neg_item = [], []
            pos_item.append(pos_new_new[i][pos_new_new[i] != 0])
            pos_item.append(pos_new_old[i][pos_new_old[i] != 0])
            neg_item.append(neg_new_new[i][neg_new_new[i] != 0])
            neg_item.append(neg_new_old[i][neg_new_old[i] != 0])
            pos_item = torch.cat(pos_item, dim=0)
            neg_item = torch.cat(neg_item, dim=0)
            pos = torch.topk(pos_item, largest=False, k=1)[0]
            neg = torch.topk(neg_item, largest=True, k=1791)[0]
            item = torch.cat([pos, neg], dim=0)
            logits_all.append(item.unsqueeze(0))
        logits_all = torch.cat(logits_all, 0)
        logits_all /= temp
        labels_idx = torch.zeros(batch_size).long().cuda()
        loss = criterion(logits_all, labels_idx) * loss_weight
    elif loss_type == 'triplet':
        logits_n2o = euclidean_dist(feat_new, feat_old_large)
        logits_n2o_pos = torch.gather(logits_n2o, 1, labels_idx.view(-1, 1).cuda())

        # find the hardest negative
        if topk_neg > 0:
            logits_n2o_neg = torch.topk(logits_n2o + masks * 1e9, topk_neg, dim=1, largest=False)[0]

        logits_n2o_pos = logits_n2o_pos.expand_as(logits_n2o_neg).contiguous().view(-1)
        logits_n2o_neg = logits_n2o_neg.view(-1)
        hard_labels_idx = torch.ones_like(logits_n2o_pos)
        loss = criterion(logits_n2o_neg, logits_n2o_pos, hard_labels_idx) * loss_weight

    elif loss_type == 'triplet_ract':
        logits_n2o = euclidean_dist(feat_new, feat_old_large)
        logits_n2o_pos = torch.gather(logits_n2o, 1, labels_idx.view(-1, 1).cuda())

        logits_n2n = euclidean_dist(feat_new, feat_new_large)
        # find the hardest negative
        if topk_neg > 0:
            logits_n2o_neg = torch.topk(logits_n2o + masks * 1e9, topk_neg, dim=1, largest=False)[0]
            logits_n2n_neg = torch.topk(logits_n2n + masks * 1e9, topk_neg, dim=1, largest=False)[0]

        logits_n2o_pos = logits_n2o_pos.expand_as(logits_n2o_neg).contiguous().view(-1)
        logits_n2o_neg = logits_n2o_neg.view(-1)
        logits_n2n_neg = logits_n2n_neg.view(-1)
        hard_labels_idx = torch.ones_like(logits_n2o_pos)
        loss = criterion(logits_n2o_neg, logits_n2o_pos, hard_labels_idx)
        loss += criterion(logits_n2n_neg, logits_n2o_pos, hard_labels_idx)
        loss *= loss_weight

    elif loss_type == 'l2':
        loss = criterion(feat_new, feat_old) * loss_weight

    else:
        loss = 0.

    return loss


class BackwardCompatibleLoss(nn.Module):
    def __init__(self, temp=0.01, margin=0.8, topk_neg=-1,
                 loss_type='contra', loss_weight=1.0, gather_all=True):
        super(BackwardCompatibleLoss, self).__init__()
        self.temperature = temp
        self.loss_weight = loss_weight
        self.topk_neg = topk_neg

        #   loss_type options:
        #   - contra
        #   - triplet
        #   - l2
        #   - hot_refresh (paper: "")
        #   - triplet_ract
        if loss_type in ['contra', 'hot_refresh', 'bct_limit','bct_limit_no_s2c']:
            self.criterion = nn.CrossEntropyLoss().cuda()
        elif loss_type in ['triplet', 'triplet_ract']:
            assert topk_neg > 0, \
                "Please select top-k negatives for triplet loss"
            # not use nn.TripletMarginLoss()
            self.criterion = nn.MarginRankingLoss(margin=margin).cuda()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss().cuda()
        else:
            raise NotImplementedError("Unknown loss type: {}".format(loss_type))
        self.loss_type = loss_type
        self.gather_all = gather_all

    def forward(self, feat, feat_old, targets):
        # features l2-norm
        feat = F.normalize(feat, dim=1, p=2)
        feat_old = F.normalize(feat_old, dim=1, p=2).detach()
        batch_size = feat.size(0)
        # gather tensors from all GPUs
        if self.gather_all:
            feat_large = gather_tensor(feat)
            feat_old_large = gather_tensor(feat_old)
            targets_large = gather_tensor(targets)
            batch_size_large = feat_large.size(0)
            current_gpu = dist.get_rank()
            masks = targets_large.expand(batch_size_large, batch_size_large) \
                .eq(targets_large.expand(batch_size_large, batch_size_large).t())
            masks = masks[current_gpu * batch_size: (current_gpu + 1) * batch_size, :]  # size: (B, B*n_gpus)
        else:
            feat_large, feat_old_large, targets_large = None, None, None
            masks = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())

        # compute loss
        loss_comp = calculate_loss(feat, feat_old, feat_large, feat_old_large, targets_large, masks, self.loss_type, self.temperature,
                                   self.criterion, self.loss_weight, self.topk_neg)
        return loss_comp


class UpgradeLoss(ModuleWithRecords):
    def __init__(self, loss, embedding_size=256, memory_size=500,miner=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.embedding_memory = torch.zeros(self.memory_size, self.embedding_size)
        self.label_memory = torch.zeros(self.memory_size).long()
        self.has_been_filled = False
        self.queue_idx = 0
        self.add_to_recordable_attributes(
            list_of_names=["embedding_size", "memory_size", "queue_idx"], is_stat=False
        )
        self.loss = loss

    def forward(self, embeddings, labels, indices_tuple=None, enqueue_idx=None):
        if enqueue_idx is not None:
            assert len(enqueue_idx) <= len(self.embedding_memory)
            assert len(enqueue_idx) < len(embeddings)
        else:
            assert len(embeddings) <= len(self.embedding_memory)
        self.reset_stats()
        device = embeddings.device
        labels = c_f.to_device(labels, device=device)
        self.embedding_memory = c_f.to_device(
            self.embedding_memory, device=device, dtype=embeddings.dtype
        )
        self.label_memory = c_f.to_device(
            self.label_memory, device=device, dtype=labels.dtype
        )
        do_remove_self_comparisons = False
        if enqueue_idx is not None:
            mask = torch.zeros(len(embeddings), device=device, dtype=torch.bool)
            mask[enqueue_idx] = True
            emb_for_queue = embeddings[mask]
            labels_for_queue = labels[mask]
            embeddings = embeddings[~mask]
            labels = labels[~mask]
            # do_remove_self_comparisons = False
        else:
            emb_for_queue = embeddings
            labels_for_queue = labels
            # do_remove_self_comparisons = True

        queue_batch_size = len(emb_for_queue)
        self.add_to_memory(emb_for_queue, labels_for_queue, queue_batch_size)

        if not self.has_been_filled:
            E_mem = self.embedding_memory[: self.queue_idx]
            L_mem = self.label_memory[: self.queue_idx]
        else:
            E_mem = self.embedding_memory
            L_mem = self.label_memory

        combined_embeddings = torch.cat([embeddings, E_mem], dim=0)
        combined_labels = torch.cat([labels, L_mem], dim=0)

        # combined_embeddings_normed = F.normalize(combined_embeddings, dim=1, p=2)
        combined_embeddings_normed = combined_embeddings

        labels_set = set(combined_labels.detach().cpu().tolist())
        center_embeddings = torch.zeros([len(labels_set), self.embedding_size])
        center_embeddings = c_f.to_device(
            center_embeddings, device=device, dtype=embeddings.dtype
        )
        center_labels = torch.zeros(len(labels_set))
        center_labels = c_f.to_device(
            center_labels, device=device, dtype=labels.dtype
        )

        for i,l in enumerate(labels_set):
            center_i_features = combined_embeddings_normed[combined_labels==l]
            center = torch.mean(center_i_features,dim=0)
            # center = F.normalize(center[None,:],p=2)
            center_embeddings[i] = center
            center_labels[i] = i

        indices_tuple = self.create_indices_tuple(
            center_embeddings.shape[0],
            center_embeddings,
            center_labels,
            indices_tuple,
            do_remove_self_comparisons
        )
        loss = self.loss(center_embeddings, center_labels, indices_tuple)
        return loss

    def create_indices_tuple(
        self,
        batch_size,
        embeddings,
        labels,
        input_indices_tuple,
        do_remove_self_comparisons,
    ):
        indices_tuple = lmu.get_all_pairs_indices(labels, labels)

        if do_remove_self_comparisons:
            indices_tuple = self.remove_self_comparisons(indices_tuple)

        # indices_tuple = c_f.shift_indices_tuple(indices_tuple, batch_size)

        if input_indices_tuple is not None:
            if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
                input_indices_tuple = lmu.convert_to_pairs(input_indices_tuple, labels)
            elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
                input_indices_tuple = lmu.convert_to_triplets(
                    input_indices_tuple, labels
                )
            indices_tuple = tuple(
                [
                    torch.cat([x, c_f.to_device(y, x)], dim=0)
                    for x, y in zip(indices_tuple, input_indices_tuple)
                ]
            )

        return indices_tuple

    def add_to_memory(self, embeddings, labels, batch_size):
        self.curr_batch_idx = (
            torch.arange(
                self.queue_idx, self.queue_idx + batch_size, device=labels.device
            )
            % self.memory_size
        )
        self.embedding_memory[self.curr_batch_idx] = embeddings.detach()
        self.label_memory[self.curr_batch_idx] = labels.detach()
        prev_queue_idx = self.queue_idx
        self.queue_idx = (self.queue_idx + batch_size) % self.memory_size
        if (not self.has_been_filled) and (self.queue_idx <= prev_queue_idx):
            self.has_been_filled = True

    def remove_self_comparisons(self, indices_tuple):
        # remove self-comparisons
        assert len(indices_tuple) in [3, 4]
        s, e = self.curr_batch_idx[0], self.curr_batch_idx[-1]
        if len(indices_tuple) == 3:
            a, p, n = indices_tuple
            keep_mask = self.not_self_comparisons(a, p, s, e)
            a = a[keep_mask]
            p = p[keep_mask]
            n = n[keep_mask]
            assert len(a) == len(p) == len(n)
            return a, p, n
        elif len(indices_tuple) == 4:
            a1, p, a2, n = indices_tuple
            keep_mask = self.not_self_comparisons(a1, p, s, e)
            a1 = a1[keep_mask]
            p = p[keep_mask]
            assert len(a1) == len(p)
            assert len(a2) == len(n)
            return a1, p, a2, n

    # a: anchors
    # p: positives
    # s: curr batch start idx in queue
    # e: curr batch end idx in queue
    def not_self_comparisons(self, a, p, s, e):
        curr_batch = torch.any(p.unsqueeze(1) == self.curr_batch_idx, dim=1)
        a_c = a[curr_batch]
        p_c = p[curr_batch]
        p_c -= s
        if e <= s:
            p_c[p_c <= e - s] += self.memory_size
        without_self_comparisons = curr_batch.clone()
        without_self_comparisons[torch.where(curr_batch)[0][a_c == p_c]] = False
        return without_self_comparisons | ~curr_batch


class UpgradeCenterLoss(ModuleWithRecords):
    def __init__(self, loss, embedding_size=256, num_class=500,device_id=0,device_num=8,dataset_len=0):
        super().__init__()
        self.num_class = num_class
        self.embedding_size = embedding_size
        cur_center_num = num_class // device_num
        self.cur_center_num = cur_center_num
        self.device_id = device_id
        self.device_num = device_num
        self.dataset_len = dataset_len
        self.counter = 0
        if device_id < device_num-1:
            self.class_center = torch.zeros([cur_center_num,embedding_size]).float().to(device_id)
            self.class_past_num = torch.zeros([cur_center_num]).long().to(device_id)
            self.class_curr_id = range(device_id*cur_center_num,cur_center_num*(device_id+1))
        else:
            self.class_center = torch.zeros([cur_center_num+num_class-cur_center_num*device_num, embedding_size]).float().to(device_id)
            self.class_past_num = torch.zeros([cur_center_num+num_class-cur_center_num*device_num]).long().to(device_id)
            self.class_curr_id = range(device_id * cur_center_num, num_class)
        self.add_to_recordable_attributes(
            list_of_names=["embedding_size", "num_class","cur_center_num"], is_stat=False
        )
        self.margin = 1
        self.loss = loss

    def shift_index(self,l):
        return l-self.cur_center_num*self.device_id

    def reset_center(self):
        if self.device_id < self.device_num-1:
            self.class_center = torch.zeros([self.cur_center_num,self.embedding_size]).float().to(self.device_id)
            self.class_past_num = torch.zeros([self.cur_center_num]).long().to(self.device_id)
            self.class_curr_id = range(self.device_id*self.cur_center_num,self.cur_center_num*(self.device_id+1))
        else:
            self.class_center = torch.zeros([self.cur_center_num+self.num_class-self.cur_center_num*self.device_num, self.embedding_size]).float().to(self.device_id)
            self.class_past_num = torch.zeros([self.cur_center_num+self.num_class-self.cur_center_num*self.device_num]).long().to(self.device_id)
            self.class_curr_id = range(self.device_id * self.cur_center_num, self.num_class)
        self.counter = 0

    def forward(self, embeddings, labels,indices_tuple=None,reset=False):
        device = embeddings.device
        self.counter += 1
        self.class_center = c_f.to_device(
            self.class_center, device=device, dtype=embeddings.dtype
        )
        for l in set(labels.detach().cpu().tolist()):
            if l not in self.class_curr_id:
                continue
            feat_l = embeddings[labels==l].detach()
            feat_l = F.normalize(feat_l)
            l = self.shift_index(l)
            curr_num = feat_l.shape[0]+self.class_past_num[l]
            self.class_center[l] = self.class_center[l]*(self.class_past_num[l]/curr_num) + (torch.sum(feat_l,dim=0)/curr_num)
            self.class_center[l] = F.normalize(self.class_center[l][None,:])
            self.class_past_num[l] = curr_num
        valid_label_idx = self.class_past_num != 0
        center_labels = c_f.to_device(
            torch.arange(0,sum(valid_label_idx)), device=device, dtype=labels.dtype
        )

        dist = torch.mm(self.class_center[valid_label_idx],self.class_center[valid_label_idx].t())
        dist_up = torch.triu(dist,diagonal=1)
        dist_up = 1-dist_up[dist_up != 0]
        dist_up = self.margin-dist_up
        dist_up = dist_up[dist_up > 0]
        loss = dist_up.mean()
        # loss = self.loss(self.class_center[valid_label_idx], center_labels, indices_tuple)
        if self.counter == self.dataset_len:
            print('reset upgrade class center')
            self.reset_center()
        return loss

    def create_indices_tuple(
        self,
        batch_size,
        embeddings,
        labels,
        input_indices_tuple
    ):
        indices_tuple = lmu.get_all_pairs_indices(labels, labels)

        # if do_remove_self_comparisons:
        #     indices_tuple = self.remove_self_comparisons(indices_tuple)

        # indices_tuple = c_f.shift_indices_tuple(indices_tuple, batch_size)

        if input_indices_tuple is not None:
            if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
                input_indices_tuple = lmu.convert_to_pairs(input_indices_tuple, labels)
            elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
                input_indices_tuple = lmu.convert_to_triplets(
                    input_indices_tuple, labels
                )
            indices_tuple = tuple(
                [
                    torch.cat([x, c_f.to_device(y, x)], dim=0)
                    for x, y in zip(indices_tuple, input_indices_tuple)
                ]
            )

        return indices_tuple

class UpgradeCenterPartialLoss(CrossBatchMemory):
    def __init__(self, loss, embedding_size=256, num_class=500,device_id=0,device_num=8,dataset_len=0):
        super().__init__(None,embedding_size=embedding_size)
        self.num_class = num_class
        self.embedding_size = embedding_size
        self.device_id = device_id
        self.device_num = device_num
        self.dataset_len = dataset_len
        self.counter = 0
        self.class_center = torch.zeros([num_class,embedding_size]).float()
        self.class_past_num = torch.zeros([num_class]).long()
        self.add_to_recordable_attributes(
            list_of_names=["embedding_size", "num_class","cur_center_num"], is_stat=False
        )
        self.margin = 1
        self.loss = loss


    def reset_center(self):
        self.class_center = torch.zeros([self.num_class, self.embedding_size]).float()
        self.class_past_num = torch.zeros([self.num_class]).long()
        self.counter = 0

    def forward(self, embeddings, labels,indices_tuple=None, enqueue_idx=None):
        device = embeddings.device
        self.counter += 1

        curr_embedding = torch.zeros([len(set(labels.detach().cpu().tolist())),self.embedding_size]).float().to(device)
        curr_class = torch.zeros([len(set(labels.detach().cpu().tolist()))]).long().to(device)

        for i,l in enumerate(set(labels.detach().cpu().tolist())):
            feat_l = embeddings[labels==l].detach()
            feat_l = F.normalize(feat_l)
            # new instance num of class l
            new_num = feat_l.shape[0]+self.class_past_num[l]
            # fetch the center of class l
            center = self.class_center[l].to(device)
            # fetch history class num
            past_num = self.class_past_num[l]
            # update the center of class l
            curr_embedding[i] = center*(past_num/new_num) + (torch.sum(feat_l,dim=0)/new_num) # (center*past_num+sum(feat_l))/new_num
            curr_embedding[i] = F.normalize(curr_embedding[i][None,:])
            self.class_past_num[l] = new_num
            curr_class[i] = i

        dist = torch.mm(curr_embedding,curr_embedding.t())
        dist_up = torch.triu(dist,diagonal=1)
        dist_up = 1-dist_up[dist_up != 0]
        dist_up = self.margin-dist_up
        dist_up = dist_up[dist_up > 0]
        loss = dist_up.mean()
        # loss = self.loss(self.class_center[valid_label_idx], center_labels, indices_tuple)
        if self.counter == self.dataset_len:
            print('reset upgrade class center')
            self.reset_center()
        return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
