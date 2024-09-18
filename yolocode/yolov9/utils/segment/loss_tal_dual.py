import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import sigmoid_focal_loss

from utils.general import xywh2xyxy, xyxy2xywh
from utils.metrics import bbox_iou
from utils.segment.tal.anchor_generator import dist2bbox, make_anchors, bbox2dist
from utils.segment.tal.assigner import TaskAlignedAssigner
from utils.torch_utils import de_parallel
from utils.segment.general import crop_mask


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight
            ).sum()
        return loss


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # iou loss
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).view(-1, 4)
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).view(-1, 4)
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)

        iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
        loss_iou = 1.0 - iou

        loss_iou *= bbox_weight
        loss_iou = loss_iou.sum() / target_scores_sum

        # dfl loss
        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).view(-1, 4, self.reg_max + 1)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).view(-1, 4)
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, iou

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = (
            F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction="none").view(
                target_left.shape
            )
            * weight_left
        )
        loss_right = (
            F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction="none").view(
                target_left.shape
            )
            * weight_right
        )
        return (loss_left + loss_right).mean(-1, keepdim=True)


class ComputeLoss:
    # Compute losses
    def __init__(self, model, use_dfl=True, overlap=True):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls = FocalLoss(BCEcls, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.no = m.no
        self.nm = m.nm
        self.overlap = overlap
        self.reg_max = m.reg_max
        self.device = device

        self.assigner = TaskAlignedAssigner(
            topk=int(os.getenv('YOLOM', 10)),
            num_classes=self.nc,
            alpha=float(os.getenv('YOLOA', 0.5)),
            beta=float(os.getenv('YOLOB', 6.0)),
        )
        self.assigner2 = TaskAlignedAssigner(
            topk=int(os.getenv('YOLOM', 10)),
            num_classes=self.nc,
            alpha=float(os.getenv('YOLOA', 0.5)),
            beta=float(os.getenv('YOLOB', 6.0)),
        )
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.bbox_loss2 = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
        self.use_dfl = use_dfl

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, p, targets, masks, img=None, epoch=0):
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl

        feats_, pred_masks_, proto_ = p if len(p) == 3 else p[1]

        feats, pred_masks, proto = feats_[0], pred_masks_[0], proto_[0]
        feats2, pred_masks2, proto2 = feats_[1], pred_masks_[1], proto_[1]

        batch_size, _, mask_h, mask_w = proto.shape

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        pred_distri2, pred_scores2 = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats2], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores2 = pred_scores2.permute(0, 2, 1).contiguous()
        pred_distri2 = pred_distri2.permute(0, 2, 1).contiguous()
        pred_masks2 = pred_masks2.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = targets[:, 0].view(-1, 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR.') from e

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        pred_bboxes2 = self.bbox_decode(anchor_points, pred_distri2)  # xyxy, (b, h*w, 4)

        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_labels2, target_bboxes2, target_scores2, fg_mask2, target_gt_idx2 = self.assigner2(
            pred_scores2.detach().sigmoid(),
            (pred_bboxes2.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum()

        target_scores_sum2 = target_scores2.sum()

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        loss[2] *= 0.25
        loss[2] += self.BCEcls(pred_scores2, target_scores2.to(dtype)).sum() / target_scores_sum2  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[3], _ = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

            # masks loss
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            for i in range(batch_size):
                if fg_mask[i].sum():
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    loss[1] += self.single_mask_loss(
                        gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy, marea
                    )  # seg loss

            loss[0] *= 0.25
            loss[3] *= 0.25
            loss[1] *= 0.25

        # bbox loss
        if fg_mask2.sum():
            loss0_, loss3_, _ = self.bbox_loss2(
                pred_distri2,
                pred_bboxes2,
                anchor_points,
                target_bboxes2 / stride_tensor,
                target_scores2,
                target_scores_sum2,
                fg_mask2,
            )

            # masks loss
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            for i in range(batch_size):
                if fg_mask2[i].sum():
                    mask_idx = target_gt_idx2[i][fg_mask2[i]]
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    xyxyn = target_bboxes2[i][fg_mask2[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    loss[1] += self.single_mask_loss(
                        gt_mask, pred_masks2[i][fg_mask2[i]], proto2[i], mxyxy, marea
                    )  # seg loss

            loss[0] += loss0_
            loss[3] += loss3_

        loss[0] *= 7.5  # box gain
        loss[1] *= 2.5 / batch_size
        loss[2] *= 0.5  # cls gain
        loss[3] *= 1.5  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        # Mask loss for one image
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n, 32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        # loss = sigmoid_focal_loss(pred_mask, gt_mask, alpha = .25, gamma = 2., reduction = 'none')

        # p_m = torch.flatten(pred_mask.softmax(dim = 1))
        # g_m = torch.flatten(gt_mask)
        # i_m = torch.sum(torch.mul(p_m, g_m))
        # u_m = torch.sum(torch.add(p_m, g_m))
        # dice_coef = (2. * i_m + 1.) / (u_m + 1.)
        # dice_loss = (1. - dice_coef)
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()


class ComputeLossLH:
    # Compute losses
    def __init__(self, model, use_dfl=True, overlap=True):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls = FocalLoss(BCEcls, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.no = m.no
        self.nm = m.nm
        self.overlap = overlap
        self.reg_max = m.reg_max
        self.device = device

        self.assigner = TaskAlignedAssigner(
            topk=int(os.getenv('YOLOM', 10)),
            num_classes=self.nc,
            alpha=float(os.getenv('YOLOA', 0.5)),
            beta=float(os.getenv('YOLOB', 6.0)),
        )
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
        self.use_dfl = use_dfl

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, p, targets, masks, img=None, epoch=0):
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl

        feats_, pred_masks_, proto_ = p if len(p) == 3 else p[1]

        feats, pred_masks, proto = feats_[0], pred_masks_[0], proto_[0]
        feats2, pred_masks2, proto2 = feats_[1], pred_masks_[1], proto_[1]

        batch_size, _, mask_h, mask_w = proto.shape

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        pred_distri2, pred_scores2 = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats2], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores2 = pred_scores2.permute(0, 2, 1).contiguous()
        pred_distri2 = pred_distri2.permute(0, 2, 1).contiguous()
        pred_masks2 = pred_masks2.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = targets[:, 0].view(-1, 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR.') from e

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        pred_bboxes2 = self.bbox_decode(anchor_points, pred_distri2)  # xyxy, (b, h*w, 4)

        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores2.detach().sigmoid(),
            (pred_bboxes2.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum()

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        loss[2] *= 0.25
        loss[2] += self.BCEcls(pred_scores2, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[3], _ = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

            # masks loss
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            for i in range(batch_size):
                if fg_mask[i].sum():
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    loss[1] += self.single_mask_loss(
                        gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy, marea
                    )  # seg loss

            loss[0] *= 0.25
            loss[3] *= 0.25
            loss[1] *= 0.25

        # bbox loss
        if fg_mask.sum():
            loss0_, loss3_, _ = self.bbox_loss(
                pred_distri2,
                pred_bboxes2,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

            # masks loss
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            for i in range(batch_size):
                if fg_mask[i].sum():
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    loss[1] += self.single_mask_loss(
                        gt_mask, pred_masks2[i][fg_mask[i]], proto2[i], mxyxy, marea
                    )  # seg loss

            loss[0] += loss0_
            loss[3] += loss3_

        loss[0] *= 7.5  # box gain
        loss[1] *= 2.5 / batch_size
        loss[2] *= 0.5  # cls gain
        loss[3] *= 1.5  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        # Mask loss for one image
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n, 32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        # loss = sigmoid_focal_loss(pred_mask, gt_mask, alpha = .25, gamma = 2., reduction = 'none')

        # p_m = torch.flatten(pred_mask.softmax(dim = 1))
        # g_m = torch.flatten(gt_mask)
        # i_m = torch.sum(torch.mul(p_m, g_m))
        # u_m = torch.sum(torch.add(p_m, g_m))
        # dice_coef = (2. * i_m + 1.) / (u_m + 1.)
        # dice_loss = (1. - dice_coef)
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()


class ComputeLossLH0:
    # Compute losses
    def __init__(self, model, use_dfl=True, overlap=True):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls = FocalLoss(BCEcls, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.no = m.no
        self.nm = m.nm
        self.overlap = overlap
        self.reg_max = m.reg_max
        self.device = device

        self.assigner = TaskAlignedAssigner(
            topk=int(os.getenv('YOLOM', 10)),
            num_classes=self.nc,
            alpha=float(os.getenv('YOLOA', 0.5)),
            beta=float(os.getenv('YOLOB', 6.0)),
        )
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
        self.use_dfl = use_dfl

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, p, targets, masks, img=None, epoch=0):
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl

        feats_, pred_masks_, proto_ = p if len(p) == 3 else p[1]

        feats, pred_masks, proto = feats_[0], pred_masks_[0], proto_[0]
        feats2, pred_masks2, proto2 = feats_[1], pred_masks_[1], proto_[1]

        batch_size, _, mask_h, mask_w = proto.shape

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        pred_distri2, pred_scores2 = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats2], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores2 = pred_scores2.permute(0, 2, 1).contiguous()
        pred_distri2 = pred_distri2.permute(0, 2, 1).contiguous()
        pred_masks2 = pred_masks2.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = targets[:, 0].view(-1, 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR.') from e

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        pred_bboxes2 = self.bbox_decode(anchor_points, pred_distri2)  # xyxy, (b, h*w, 4)

        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores2.detach().sigmoid(),
            (pred_bboxes2.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum()

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        loss[2] *= 0.25
        loss[2] += self.BCEcls(pred_scores2, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[3], _ = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

            # masks loss
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            for i in range(batch_size):
                if fg_mask[i].sum():
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    loss[1] += self.single_mask_loss(
                        gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy, marea
                    )  # seg loss

            loss[0] *= 0.25
            loss[3] *= 0.25
            loss[1] *= 0.25

        # bbox loss
        if fg_mask.sum():
            loss0_, loss3_, _ = self.bbox_loss(
                pred_distri2,
                pred_bboxes2,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

            # masks loss
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            for i in range(batch_size):
                if fg_mask[i].sum():
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    loss[1] += 0.0 * self.single_mask_loss(
                        gt_mask, pred_masks2[i][fg_mask[i]], proto2[i], mxyxy, marea
                    )  # seg loss

            loss[0] += loss0_
            loss[3] += loss3_

        loss[0] *= 7.5  # box gain
        loss[1] *= 2.5 / batch_size
        loss[2] *= 0.5  # cls gain
        loss[3] *= 1.5  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        # Mask loss for one image
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n, 32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        # loss = sigmoid_focal_loss(pred_mask, gt_mask, alpha = .25, gamma = 2., reduction = 'none')

        # p_m = torch.flatten(pred_mask.softmax(dim = 1))
        # g_m = torch.flatten(gt_mask)
        # i_m = torch.sum(torch.mul(p_m, g_m))
        # u_m = torch.sum(torch.add(p_m, g_m))
        # dice_coef = (2. * i_m + 1.) / (u_m + 1.)
        # dice_loss = (1. - dice_coef)
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()
