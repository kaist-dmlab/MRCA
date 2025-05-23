# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import numpy as np
import json
import math
import torch
from torch import nn
from torch.autograd.function import Function
from typing import Dict, List, Optional, Tuple, Union
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.layers import batched_nms
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads, select_foreground_proposals
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads, _ScaleGradient
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from .detic_fast_rcnn import DeticFastRCNNOutputLayers
from ..debug import debug_second_stage

from torch.cuda.amp import autocast

@ROI_HEADS_REGISTRY.register()
class DeticCascadeROIHeads(CascadeROIHeads):
    @configurable
    def __init__(
        self,
        *,
        mult_proposal_score: bool = False,
        with_image_labels: bool = False,
        add_image_box: bool = False,
        image_box_size: float = 1.0,
        ws_num_props: int = 512,
        add_feature_to_prop: bool = False,
        mask_weight: float = 1.0,
        one_class_per_proposal: bool = False,
        seg_in_feature: str = 'p3',
        bsgal_mask_loss = True,
        seperate_sup = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mult_proposal_score = mult_proposal_score
        self.with_image_labels = with_image_labels
        self.add_image_box = add_image_box
        self.image_box_size = image_box_size
        self.ws_num_props = ws_num_props
        self.add_feature_to_prop = add_feature_to_prop
        self.mask_weight = mask_weight
        self.one_class_per_proposal = one_class_per_proposal
        self.bsgal_mask_loss=bsgal_mask_loss
        # TODO
        self.seg_in_feature = seg_in_feature
        self.refine_mask = getattr(self.mask_head, 'refine_mask', False)
        self.save_feature = False
        self.seperate_sup = seperate_sup

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'mult_proposal_score': cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE,
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'add_image_box': cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX,
            'image_box_size': cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE,
            'ws_num_props': cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS,
            'add_feature_to_prop': cfg.MODEL.ROI_BOX_HEAD.ADD_FEATURE_TO_PROP,
            'mask_weight': cfg.MODEL.ROI_HEADS.MASK_WEIGHT,
            'one_class_per_proposal': cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL,
            'seg_in_feature' : cfg.MODEL.ROI_HEADS.SEG_IN_FEATURE,
            'bsgal_mask_loss': cfg.MODEL.USE_XPASTE_MASK_LOSS,
            'seperate_sup': cfg.INPUT.SEPERATE_SUP,
        })
        return ret


    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictors']
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []
        for box_head, bbox_reg_weights in zip(ret['box_heads'], \
            cascade_bbox_reg_weights):
            box_predictors.append(
                DeticFastRCNNOutputLayers(
                    cfg, box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights)
                ))
        ret['box_predictors'] = box_predictors
        return ret


    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances], sem_seg_gt: torch.Tensor =None):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
        if not self.bsgal_mask_loss:
            instances_=[]
            for instance_per_img in instances:
                instance_per_img=instance_per_img[instance_per_img.instance_source==0]
                instances_.append(instance_per_img)
            instances=instances_
        features_origin = features
        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        if self.refine_mask :
            return self.mask_head(features, features_origin[self.seg_in_feature], instances, sem_seg_gt)
        return self.mask_head(features, instances)


    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets):
        """
        Match proposals with groundtruth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.
        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances
        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            instance_source=None
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                if targets_per_image.has('instance_source'):
                    instance_source=targets_per_image.instance_source[matched_idxs]
                    # instance_source[proposal_labels == 0] = 0 # TODO 
                    instance_source[proposal_labels == 0] = -1  # modified on 12.1 to identify background
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
                if targets_per_image.has('instance_source'):
                    #instance_source=torch.zeros_like(matched_idxs) # TODO
                    instance_source=torch.zeros_like(matched_idxs)-1 # modified on 12.1 to identify background
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes
            if instance_source is not None:
                assert len(gt_classes)==len(instance_source),'len(gt_classes)=%s, len(instance_source)=%s'%(len(gt_classes),len(instance_source))
                proposals_per_image.instance_source=instance_source
            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_fg_samples".format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples".format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals

    def _forward_box(self, features, proposals, targets=None, 
        ann_type='box', classifier_info=(None,None,None), only_gt_proposals=False):
        """
        Add mult proposal scores at testing
        Add ann_type
        """
        if (not self.training) and self.mult_proposal_score:
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [p.get('scores') for p in proposals]
            else:
                proposal_scores = [p.get('objectness_logits') for p in proposals]
        
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(
                    prev_pred_boxes, image_sizes,
                    logits=[p.objectness_logits for p in proposals])
                if self.training and ann_type in ['box']:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            predictions = self._run_stage(features, proposals, k,classifier_info=classifier_info)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(
                (predictions[0], predictions[1]), proposals) #  511 X 4
            head_outputs.append((self.box_predictor[k], predictions, proposals))
        
        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                if storage._current_prefix != '':
                    print ('storage._current_prefix',storage._current_prefix)
                with storage.name_scope("stage{}".format(stage)):
                    if ann_type != 'box': 
                        stage_losses = {}
                        if ann_type in ['image', 'caption', 'captiontag']:
                            image_labels = [x._pos_category_ids for x in targets]
                            weak_losses = predictor.image_label_losses(
                                predictions, proposals, image_labels,
                                classifier_info=classifier_info,
                                ann_type=ann_type)
                            stage_losses.update(weak_losses)
                    else: # supervised
                        if not self.seperate_sup:
                            if not only_gt_proposals:
                                stage_losses = predictor.losses(
                                    (predictions[0], predictions[1]), proposals,
                                    classifier_info=classifier_info)
                            else:
                                stage_losses = predictor.no_grad_losses(
                                    (predictions[0], predictions[1]), proposals,
                                    classifier_info=classifier_info)
                        else: 
                            stage_losses = predictor.losses(
                                (predictions[0], predictions[1],predictions[2]), proposals,
                                classifier_info=classifier_info)
                        if self.with_image_labels:
                            stage_losses['image_loss'] = \
                                predictions[0].new_zeros([1])[0]
                losses.update({k + "_stage{}".format(stage): v \
                    for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            if self.mult_proposal_score:
                scores = [(s * ps[:, None]) ** 0.5 \
                    for s, ps in zip(scores, proposal_scores)]
            if self.one_class_per_proposal:
                scores = [s * (s == s[:, :-1].max(dim=1)[0][:, None]).float() for s in scores]
            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(
                (predictions[0], predictions[1]), proposals)
            pred_instances, inds = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            if hasattr(self, 'save_bbox_features'):
                self.save_bbox_features = self.save_bbox_features[inds[0]]
            return pred_instances


    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], only_gt_proposals=False
    ) -> List[Instances]:
        if not only_gt_proposals:
            if self.proposal_append_gt:
                proposals = add_ground_truth_to_proposals(targets, proposals)
            proposals_with_gt = []
            num_fg_samples = []
            num_bg_samples = []
            for proposals_per_image, targets_per_image in zip(proposals, targets):
                has_gt = len(targets_per_image) > 0
                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                )
                matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix) # 2000 + num_gt,matched_ids 指向对应的gt, labels指是否匹配
                sampled_idxs, gt_classes = self._sample_proposals(
                    matched_idxs, matched_labels, targets_per_image.gt_classes  # 512
                )

                # Set target attributes of the sampled proposals:
                proposals_per_image = proposals_per_image[sampled_idxs]
                proposals_per_image.gt_classes = gt_classes

                if has_gt:
                    sampled_targets = matched_idxs[sampled_idxs]  # 
                    sampled_labels = matched_labels[sampled_idxs]
                    for (trg_name, trg_value) in targets_per_image.get_fields().items():
                        if (trg_name.startswith("gt_") or trg_name=='instance_source') and not proposals_per_image.has(trg_name):
                            proposals_per_image.set(trg_name, trg_value[sampled_targets]) # TODO
                            # update instance_source
                            if trg_name=='instance_source': 
                                proposals_per_image.instance_source[sampled_labels == 0] = -1
                num_bg_samples.append((gt_classes == self.num_classes).sum().item())
                num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
                proposals_with_gt.append(proposals_per_image)
            storage = get_event_storage()
            storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
            storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        else:
            # only gt proposals
            proposals_with_gt = []
            if self.proposal_append_gt:
                proposals = add_ground_truth_to_proposals(targets, proposals)
            for proposals_per_image, targets_per_image in zip(proposals, targets):
                if len(targets_per_image) > 0:
                    proposals_per_image = proposals_per_image[-len(targets_per_image):]
                    try:
                        proposals_per_image.gt_classes = targets_per_image.gt_classes
                    except:
                        print('targets_per_image.gt_classes',targets_per_image.gt_classes)
                    for (trg_name, trg_value) in targets_per_image.get_fields().items():
                        if (trg_name.startswith("gt_") or trg_name=='instance_source') and not proposals_per_image.has(trg_name):
                            proposals_per_image.set(trg_name, trg_value)
                else: # 1个gt都没有
                    match_quality_matrix = pairwise_iou(
                        targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                    )
                    matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix) # 2000 + num_gt
                    sampled_idxs, gt_classes = self._sample_proposals(
                        matched_idxs, matched_labels, targets_per_image.gt_classes
                    )
                    proposals_per_image = proposals_per_image[sampled_idxs]
                    proposals_per_image.gt_classes = gt_classes

                proposals_with_gt.append(proposals_per_image)


        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None,
        ann_type='box', classifier_info=(None,None,None), sem_seg_gt=None,
        only_gt_proposals=False, **kwargs):
        '''
        enable debug and image labels
        classifier_info is shared across the batch
        '''
        if self.training: # TODO add only gt_proposals
            if ann_type in ['box', 'prop', 'proptag']:
                proposals = self.label_and_sample_proposals( 
                    proposals, targets,only_gt_proposals)
            else:
                proposals = self.get_top_proposals(proposals)
            
            losses = self._forward_box(features, proposals, targets, \
                ann_type=ann_type, classifier_info=classifier_info, only_gt_proposals=only_gt_proposals)
            if ann_type == 'box' and targets[0].has('gt_masks'):
                mask_losses = self._forward_mask(features, proposals, sem_seg_gt)
                losses.update({k: v * self.mask_weight \
                    for k, v in mask_losses.items()})
                losses.update(self._forward_keypoint(features, proposals))
            else:
                losses.update(self._get_empty_mask_loss(
                    features, proposals,
                    device=proposals[0].objectness_logits.device))
            return proposals, losses
        else:
            pred_instances = self._forward_box(
                features, proposals, classifier_info=classifier_info)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


    def get_top_proposals(self, proposals):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.clip(proposals[i].image_size)
        proposals = [p[:self.ws_num_props] for p in proposals]
        for i, p in enumerate(proposals):
            p.proposal_boxes.tensor = p.proposal_boxes.tensor.detach()
            if self.add_image_box:
                proposals[i] = self._add_image_box(p)
        return proposals


    def _add_image_box(self, p):
        image_box = Instances(p.image_size)
        n = 1
        h, w = p.image_size
        f = self.image_box_size
        image_box.proposal_boxes = Boxes(
            p.proposal_boxes.tensor.new_tensor(
                [w * (1. - f) / 2., 
                    h * (1. - f) / 2.,
                    w * (1. - (1. - f) / 2.), 
                    h * (1. - (1. - f) / 2.)]
                ).view(n, 4))
        image_box.objectness_logits = p.objectness_logits.new_ones(n)
        return Instances.cat([p, image_box])


    def _get_empty_mask_loss(self, features, proposals, device):
        if self.mask_on:
            return {'loss_mask': torch.zeros(
                (1, ), device=device, dtype=torch.float32)[0]}
        else:
            return {}


    def _create_proposals_from_boxes(self, boxes, image_sizes, logits):
        """
        Add objectness_logits
        """
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size, logit in zip(
            boxes, image_sizes, logits):
            boxes_per_image.clip(image_size)
            if self.training:
                inds = boxes_per_image.nonempty()
                boxes_per_image = boxes_per_image[inds]
                logit = logit[inds]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            prop.objectness_logits = logit
            proposals.append(prop)
        return proposals


    def _run_stage(self, features, proposals, stage, \
        classifier_info=(None,None,None)):
        """
        Support classifier_info and add_feature_to_prop
        """
        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        if stage == self.num_cascade_stages - 1 and self.save_feature:
            self.save_bbox_features = box_features
        box_features = self.box_head[stage](box_features)
        if self.add_feature_to_prop:
            feats_per_image = box_features.split(
                [len(p) for p in proposals], dim=0)
            for feat, p in zip(feats_per_image, proposals):
                p.feat = feat
        return self.box_predictor[stage](
            box_features, 
            classifier_info=classifier_info)


@ROI_HEADS_REGISTRY.register()
class DeticCascadeROIHeadsLogits(CascadeROIHeads):
    @configurable
    def __init__(
        self,
        *,
        mult_proposal_score: bool = False,
        with_image_labels: bool = False,
        add_image_box: bool = False,
        image_box_size: float = 1.0,
        ws_num_props: int = 512,
        add_feature_to_prop: bool = False,
        mask_weight: float = 1.0,
        one_class_per_proposal: bool = False,
        seg_in_feature: str = 'p3',
        bsgal_mask_loss = True,
        seperate_sup = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mult_proposal_score = mult_proposal_score
        self.with_image_labels = with_image_labels
        self.add_image_box = add_image_box
        self.image_box_size = image_box_size
        self.ws_num_props = ws_num_props
        self.add_feature_to_prop = add_feature_to_prop
        self.mask_weight = mask_weight
        self.one_class_per_proposal = one_class_per_proposal
        self.bsgal_mask_loss=bsgal_mask_loss
        # TODO
        self.seg_in_feature = seg_in_feature
        self.refine_mask = getattr(self.mask_head, 'refine_mask', False)
        self.save_feature = False
        self.seperate_sup = seperate_sup

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'mult_proposal_score': cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE,
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'add_image_box': cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX,
            'image_box_size': cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE,
            'ws_num_props': cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS,
            'add_feature_to_prop': cfg.MODEL.ROI_BOX_HEAD.ADD_FEATURE_TO_PROP,
            'mask_weight': cfg.MODEL.ROI_HEADS.MASK_WEIGHT,
            'one_class_per_proposal': cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL,
            'seg_in_feature' : cfg.MODEL.ROI_HEADS.SEG_IN_FEATURE,
            'bsgal_mask_loss': cfg.MODEL.USE_XPASTE_MASK_LOSS,
            'seperate_sup': cfg.INPUT.SEPERATE_SUP,
        })
        return ret


    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictors']
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []
        for box_head, bbox_reg_weights in zip(ret['box_heads'], \
            cascade_bbox_reg_weights):
            box_predictors.append(
                DeticFastRCNNOutputLayers(
                    cfg, box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights)
                ))
        ret['box_predictors'] = box_predictors
        return ret


    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances], sem_seg_gt: torch.Tensor =None):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
        if not self.bsgal_mask_loss:
            instances_=[]
            for instance_per_img in instances:
                instance_per_img=instance_per_img[instance_per_img.instance_source==0]
                instances_.append(instance_per_img)
            instances=instances_
        features_origin = features
        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        if self.refine_mask :
            return self.mask_head(features, features_origin[self.seg_in_feature], instances, sem_seg_gt)
        return self.mask_head(features, instances)


    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets):
        """
        Match proposals with groundtruth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.
        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances
        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            instance_source=None
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                if targets_per_image.has('instance_source'):
                    instance_source=targets_per_image.instance_source[matched_idxs]
                    # instance_source[proposal_labels == 0] = 0 # TODO 
                    instance_source[proposal_labels == 0] = -1  # modified on 12.1 to identify background
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
                if targets_per_image.has('instance_source'):
                    #instance_source=torch.zeros_like(matched_idxs) # TODO
                    instance_source=torch.zeros_like(matched_idxs)-1 # modified on 12.1 to identify background
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes
            if instance_source is not None:
                assert len(gt_classes)==len(instance_source),'len(gt_classes)=%s, len(instance_source)=%s'%(len(gt_classes),len(instance_source))
                proposals_per_image.instance_source=instance_source
            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_fg_samples".format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples".format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals

    def _forward_box(self, features, proposals, targets=None, 
        ann_type='box', classifier_info=(None,None,None), only_gt_proposals=False):
        """
        Add mult proposal scores at testing
        Add ann_type
        """
        if (not self.training) and self.mult_proposal_score:
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [p.get('scores') for p in proposals]
            else:
                proposal_scores = [p.get('objectness_logits') for p in proposals]
        
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        # for k in range(self.num_cascade_stages):
        #     if k > 0:
        #         proposals = self._create_proposals_from_boxes(
        #             prev_pred_boxes, image_sizes,
        #             logits=[p.objectness_logits for p in proposals])
        #         if self.training and ann_type in ['box']:
        #             proposals = self._match_and_label_boxes(proposals, k, targets)
        #     predictions = self._run_stage(features, proposals, k,classifier_info=classifier_info)
        #     prev_pred_boxes = self.box_predictor[k].predict_boxes(
        #         (predictions[0], predictions[1]), proposals) #  511 X 4
        #     head_outputs.append((self.box_predictor[k], predictions, proposals))

        # for k in range(self.num_cascade_stages):
        #     if k > 0:
        #         proposals = self._create_proposals_from_boxes(
        #             prev_pred_boxes, image_sizes,
        #             logits=[p.objectness_logits for p in proposals])
        #         if self.training and ann_type in ['box']:
        #             proposals = self._match_and_label_boxes(proposals, k, targets)
        k  = self.num_cascade_stages-1
        predictions = self._run_stage(features, proposals, k,classifier_info=classifier_info)
        # print("predictions")
        # print(predictions)
        return predictions[0]

        prev_pred_boxes = self.box_predictor[k].predict_boxes(
            (predictions[0], predictions[1]), proposals) #  511 X 4
        head_outputs.append((self.box_predictor[k], predictions, proposals))
        
        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                if storage._current_prefix != '':
                    print ('storage._current_prefix',storage._current_prefix)
                with storage.name_scope("stage{}".format(stage)):
                    if ann_type != 'box': 
                        stage_losses = {}
                        if ann_type in ['image', 'caption', 'captiontag']:
                            image_labels = [x._pos_category_ids for x in targets]
                            weak_losses = predictor.image_label_losses(
                                predictions, proposals, image_labels,
                                classifier_info=classifier_info,
                                ann_type=ann_type)
                            stage_losses.update(weak_losses)
                    else: # supervised
                        if not self.seperate_sup:
                            if not only_gt_proposals:
                                stage_losses = predictor.losses(
                                    (predictions[0], predictions[1]), proposals,
                                    classifier_info=classifier_info)
                            else:
                                stage_losses = predictor.no_grad_losses(
                                    (predictions[0], predictions[1]), proposals,
                                    classifier_info=classifier_info)
                        else: 
                            stage_losses = predictor.losses(
                                (predictions[0], predictions[1],predictions[2]), proposals,
                                classifier_info=classifier_info)
                        if self.with_image_labels:
                            stage_losses['image_loss'] = \
                                predictions[0].new_zeros([1])[0]
                losses.update({k + "_stage{}".format(stage): v \
                    for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            print("scores?")
            print(scores)
            print(len(scores[0][0]))
            if self.mult_proposal_score:
                scores = [(s * ps[:, None]) ** 0.5 \
                    for s, ps in zip(scores, proposal_scores)]
            if self.one_class_per_proposal:
                scores = [s * (s == s[:, :-1].max(dim=1)[0][:, None]).float() for s in scores]

            print("scores2?")
            print(scores)
            print(len(scores[0][0]))

            predictor, predictions, proposals = head_outputs[-1]
            print("predictions?")
            print(predictions)
            print(len(predictions[0][0]))
            boxes = predictor.predict_boxes(
                (predictions[0], predictions[1]), proposals)
            print("boxes?")
            print(boxes)
            
            pred_instances, inds = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )

            print("pred instances?")
            print(pred_instances)
            print(len(pred_instances[0][0]))


            if hasattr(self, 'save_bbox_features'):
                self.save_bbox_features = self.save_bbox_features[inds[0]]
            return pred_instances


    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], only_gt_proposals=False
    ) -> List[Instances]:
        if not only_gt_proposals:
            if self.proposal_append_gt:
                proposals = add_ground_truth_to_proposals(targets, proposals)
            proposals_with_gt = []
            num_fg_samples = []
            num_bg_samples = []
            for proposals_per_image, targets_per_image in zip(proposals, targets):
                has_gt = len(targets_per_image) > 0
                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                )
                matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix) # 2000 + num_gt,matched_ids 指向对应的gt, labels指是否匹配
                sampled_idxs, gt_classes = self._sample_proposals(
                    matched_idxs, matched_labels, targets_per_image.gt_classes  # 512
                )

                # Set target attributes of the sampled proposals:
                proposals_per_image = proposals_per_image[sampled_idxs]
                proposals_per_image.gt_classes = gt_classes

                if has_gt:
                    sampled_targets = matched_idxs[sampled_idxs]  # 
                    sampled_labels = matched_labels[sampled_idxs]
                    for (trg_name, trg_value) in targets_per_image.get_fields().items():
                        if (trg_name.startswith("gt_") or trg_name=='instance_source') and not proposals_per_image.has(trg_name):
                            proposals_per_image.set(trg_name, trg_value[sampled_targets]) # TODO
                            # update instance_source
                            if trg_name=='instance_source': 
                                proposals_per_image.instance_source[sampled_labels == 0] = -1
                num_bg_samples.append((gt_classes == self.num_classes).sum().item())
                num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
                proposals_with_gt.append(proposals_per_image)
            storage = get_event_storage()
            storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
            storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
            # storage.history(name="roi_head/num_fg_samples").values()
            # record matched_idxs, not a scalar
            # storage.put_image("roi_head/matched_idxs", matched_idxs)
            # storage.history(name="roi_head/matched_idxs").values()
            # storage._vis_data.pop()[1]
        else:
            # only gt proposals
            proposals_with_gt = []
            if self.proposal_append_gt:
                proposals = add_ground_truth_to_proposals(targets, proposals)
            for proposals_per_image, targets_per_image in zip(proposals, targets):
                if len(targets_per_image) > 0:
                    proposals_per_image = proposals_per_image[-len(targets_per_image):]
                    try:
                        proposals_per_image.gt_classes = targets_per_image.gt_classes
                    except:
                        print('targets_per_image.gt_classes',targets_per_image.gt_classes)
                    for (trg_name, trg_value) in targets_per_image.get_fields().items():
                        if (trg_name.startswith("gt_") or trg_name=='instance_source') and not proposals_per_image.has(trg_name):
                            proposals_per_image.set(trg_name, trg_value)
                else: # 1个gt都没有
                    match_quality_matrix = pairwise_iou(
                        targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                    )
                    matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix) # 2000 + num_gt
                    sampled_idxs, gt_classes = self._sample_proposals(
                        matched_idxs, matched_labels, targets_per_image.gt_classes
                    )
                    proposals_per_image = proposals_per_image[sampled_idxs]
                    proposals_per_image.gt_classes = gt_classes

                proposals_with_gt.append(proposals_per_image)


        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None,
        ann_type='box', classifier_info=(None,None,None), sem_seg_gt=None,
        only_gt_proposals=False, **kwargs):
        '''
        enable debug and image labels
        classifier_info is shared across the batch
        '''
        if self.training: # TODO add only gt_proposals
            if ann_type in ['box', 'prop', 'proptag']:
                proposals = self.label_and_sample_proposals( 
                    proposals, targets,only_gt_proposals)
            else:
                proposals = self.get_top_proposals(proposals)
            
            losses = self._forward_box(features, proposals, targets, \
                ann_type=ann_type, classifier_info=classifier_info, only_gt_proposals=only_gt_proposals)
            if ann_type == 'box' and targets[0].has('gt_masks'):
                mask_losses = self._forward_mask(features, proposals, sem_seg_gt)
                losses.update({k: v * self.mask_weight \
                    for k, v in mask_losses.items()})
                losses.update(self._forward_keypoint(features, proposals))
            else:
                losses.update(self._get_empty_mask_loss(
                    features, proposals,
                    device=proposals[0].objectness_logits.device))
            return proposals, losses
        else:
            pred_instances = self._forward_box(
                features, proposals, classifier_info=classifier_info)
            # print("output of forward box")
            # print(pred_instances)
            # pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


    def get_top_proposals(self, proposals):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.clip(proposals[i].image_size)
        proposals = [p[:self.ws_num_props] for p in proposals]
        for i, p in enumerate(proposals):
            p.proposal_boxes.tensor = p.proposal_boxes.tensor.detach()
            if self.add_image_box:
                proposals[i] = self._add_image_box(p)
        return proposals


    def _add_image_box(self, p):
        image_box = Instances(p.image_size)
        n = 1
        h, w = p.image_size
        f = self.image_box_size
        image_box.proposal_boxes = Boxes(
            p.proposal_boxes.tensor.new_tensor(
                [w * (1. - f) / 2., 
                    h * (1. - f) / 2.,
                    w * (1. - (1. - f) / 2.), 
                    h * (1. - (1. - f) / 2.)]
                ).view(n, 4))
        image_box.objectness_logits = p.objectness_logits.new_ones(n)
        return Instances.cat([p, image_box])


    def _get_empty_mask_loss(self, features, proposals, device):
        if self.mask_on:
            return {'loss_mask': torch.zeros(
                (1, ), device=device, dtype=torch.float32)[0]}
        else:
            return {}


    def _create_proposals_from_boxes(self, boxes, image_sizes, logits):
        """
        Add objectness_logits
        """
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size, logit in zip(
            boxes, image_sizes, logits):
            boxes_per_image.clip(image_size)
            if self.training:
                inds = boxes_per_image.nonempty()
                boxes_per_image = boxes_per_image[inds]
                logit = logit[inds]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            prop.objectness_logits = logit
            proposals.append(prop)
        return proposals


    def _run_stage(self, features, proposals, stage, \
        classifier_info=(None,None,None)):
        """
        Support classifier_info and add_feature_to_prop
        """
        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        if stage == self.num_cascade_stages - 1 and self.save_feature:
            self.save_bbox_features = box_features
        box_features = self.box_head[stage](box_features)
        if self.add_feature_to_prop:
            feats_per_image = box_features.split(
                [len(p) for p in proposals], dim=0)
            for feat, p in zip(feats_per_image, proposals):
                p.feat = feat
        return self.box_predictor[stage](
            box_features, 
            classifier_info=classifier_info)




@ROI_HEADS_REGISTRY.register()
class DeticCascadeROIHeadsEmbeds(CascadeROIHeads):
    @configurable
    def __init__(
        self,
        *,
        mult_proposal_score: bool = False,
        with_image_labels: bool = False,
        add_image_box: bool = False,
        image_box_size: float = 1.0,
        ws_num_props: int = 512,
        add_feature_to_prop: bool = False,
        mask_weight: float = 1.0,
        one_class_per_proposal: bool = False,
        seg_in_feature: str = 'p3',
        bsgal_mask_loss = True,
        seperate_sup = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mult_proposal_score = mult_proposal_score
        self.with_image_labels = with_image_labels
        self.add_image_box = add_image_box
        self.image_box_size = image_box_size
        self.ws_num_props = ws_num_props
        self.add_feature_to_prop = add_feature_to_prop
        self.mask_weight = mask_weight
        self.one_class_per_proposal = one_class_per_proposal
        self.bsgal_mask_loss=bsgal_mask_loss
        # TODO
        self.seg_in_feature = seg_in_feature
        self.refine_mask = getattr(self.mask_head, 'refine_mask', False)
        self.save_feature = False
        self.seperate_sup = seperate_sup

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'mult_proposal_score': cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE,
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'add_image_box': cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX,
            'image_box_size': cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE,
            'ws_num_props': cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS,
            'add_feature_to_prop': cfg.MODEL.ROI_BOX_HEAD.ADD_FEATURE_TO_PROP,
            'mask_weight': cfg.MODEL.ROI_HEADS.MASK_WEIGHT,
            'one_class_per_proposal': cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL,
            'seg_in_feature' : cfg.MODEL.ROI_HEADS.SEG_IN_FEATURE,
            'bsgal_mask_loss': cfg.MODEL.USE_XPASTE_MASK_LOSS,
            'seperate_sup': cfg.INPUT.SEPERATE_SUP,
        })
        return ret


    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictors']
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []
        for box_head, bbox_reg_weights in zip(ret['box_heads'], \
            cascade_bbox_reg_weights):
            box_predictors.append(
                DeticFastRCNNOutputLayers(
                    cfg, box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights)
                ))
        ret['box_predictors'] = box_predictors
        return ret


    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances], sem_seg_gt: torch.Tensor =None):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
        if not self.bsgal_mask_loss:
            instances_=[]
            for instance_per_img in instances:
                instance_per_img=instance_per_img[instance_per_img.instance_source==0]
                instances_.append(instance_per_img)
            instances=instances_
        features_origin = features
        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        if self.refine_mask :
            return self.mask_head(features, features_origin[self.seg_in_feature], instances, sem_seg_gt)
        return self.mask_head(features, instances)


    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets):
        """
        Match proposals with groundtruth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.
        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances
        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            instance_source=None
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                if targets_per_image.has('instance_source'):
                    instance_source=targets_per_image.instance_source[matched_idxs]
                    # instance_source[proposal_labels == 0] = 0 # TODO 
                    instance_source[proposal_labels == 0] = -1  # modified on 12.1 to identify background
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
                if targets_per_image.has('instance_source'):
                    #instance_source=torch.zeros_like(matched_idxs) # TODO
                    instance_source=torch.zeros_like(matched_idxs)-1 # modified on 12.1 to identify background
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes
            if instance_source is not None:
                assert len(gt_classes)==len(instance_source),'len(gt_classes)=%s, len(instance_source)=%s'%(len(gt_classes),len(instance_source))
                proposals_per_image.instance_source=instance_source
            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_fg_samples".format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples".format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals

    def _forward_box(self, features, proposals, targets=None, 
        ann_type='box', classifier_info=(None,None,None), only_gt_proposals=False):
        """
        Add mult proposal scores at testing
        Add ann_type
        """
        if (not self.training) and self.mult_proposal_score:
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [p.get('scores') for p in proposals]
            else:
                proposal_scores = [p.get('objectness_logits') for p in proposals]
        
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        k  = self.num_cascade_stages-1
        predictions = self._run_stage(features, proposals, k,classifier_info=classifier_info)
        # print("predictions")
        print(predictions.shape)
        return predictions

    

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], only_gt_proposals=False
    ) -> List[Instances]:
        if not only_gt_proposals:
            if self.proposal_append_gt:
                proposals = add_ground_truth_to_proposals(targets, proposals)
            proposals_with_gt = []
            num_fg_samples = []
            num_bg_samples = []
            for proposals_per_image, targets_per_image in zip(proposals, targets):
                has_gt = len(targets_per_image) > 0
                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                )
                matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix) # 2000 + num_gt,matched_ids 指向对应的gt, labels指是否匹配
                sampled_idxs, gt_classes = self._sample_proposals(
                    matched_idxs, matched_labels, targets_per_image.gt_classes  # 512
                )

                # Set target attributes of the sampled proposals:
                proposals_per_image = proposals_per_image[sampled_idxs]
                proposals_per_image.gt_classes = gt_classes

                if has_gt:
                    sampled_targets = matched_idxs[sampled_idxs]  # 
                    sampled_labels = matched_labels[sampled_idxs]
                    for (trg_name, trg_value) in targets_per_image.get_fields().items():
                        if (trg_name.startswith("gt_") or trg_name=='instance_source') and not proposals_per_image.has(trg_name):
                            proposals_per_image.set(trg_name, trg_value[sampled_targets]) # TODO
                            # update instance_source
                            if trg_name=='instance_source': 
                                proposals_per_image.instance_source[sampled_labels == 0] = -1
                num_bg_samples.append((gt_classes == self.num_classes).sum().item())
                num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
                proposals_with_gt.append(proposals_per_image)
            storage = get_event_storage()
            storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
            storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        else:
            # only gt proposals
            proposals_with_gt = []
            if self.proposal_append_gt:
                proposals = add_ground_truth_to_proposals(targets, proposals)
            for proposals_per_image, targets_per_image in zip(proposals, targets):
                if len(targets_per_image) > 0:
                    proposals_per_image = proposals_per_image[-len(targets_per_image):]
                    try:
                        proposals_per_image.gt_classes = targets_per_image.gt_classes
                    except:
                        print('targets_per_image.gt_classes',targets_per_image.gt_classes)
                    for (trg_name, trg_value) in targets_per_image.get_fields().items():
                        if (trg_name.startswith("gt_") or trg_name=='instance_source') and not proposals_per_image.has(trg_name):
                            proposals_per_image.set(trg_name, trg_value)
                else: # 1个gt都没有
                    match_quality_matrix = pairwise_iou(
                        targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                    )
                    matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix) # 2000 + num_gt
                    sampled_idxs, gt_classes = self._sample_proposals(
                        matched_idxs, matched_labels, targets_per_image.gt_classes
                    )
                    proposals_per_image = proposals_per_image[sampled_idxs]
                    proposals_per_image.gt_classes = gt_classes

                proposals_with_gt.append(proposals_per_image)


        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None,
        ann_type='box', classifier_info=(None,None,None), sem_seg_gt=None,
        only_gt_proposals=False, **kwargs):
        '''
        enable debug and image labels
        classifier_info is shared across the batch
        '''
        if self.training: # TODO add only gt_proposals
            if ann_type in ['box', 'prop', 'proptag']:
                proposals = self.label_and_sample_proposals( 
                    proposals, targets,only_gt_proposals)
            else:
                proposals = self.get_top_proposals(proposals)
            
            losses = self._forward_box(features, proposals, targets, \
                ann_type=ann_type, classifier_info=classifier_info, only_gt_proposals=only_gt_proposals)
            if ann_type == 'box' and targets[0].has('gt_masks'):
                mask_losses = self._forward_mask(features, proposals, sem_seg_gt)
                losses.update({k: v * self.mask_weight \
                    for k, v in mask_losses.items()})
                losses.update(self._forward_keypoint(features, proposals))
            else:
                losses.update(self._get_empty_mask_loss(
                    features, proposals,
                    device=proposals[0].objectness_logits.device))
            return proposals, losses
        else:
            pred_instances = self._forward_box(
                features, proposals, classifier_info=classifier_info)

            return pred_instances, {}


    def get_top_proposals(self, proposals):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.clip(proposals[i].image_size)
        proposals = [p[:self.ws_num_props] for p in proposals]
        for i, p in enumerate(proposals):
            p.proposal_boxes.tensor = p.proposal_boxes.tensor.detach()
            if self.add_image_box:
                proposals[i] = self._add_image_box(p)
        return proposals


    def _add_image_box(self, p):
        image_box = Instances(p.image_size)
        n = 1
        h, w = p.image_size
        f = self.image_box_size
        image_box.proposal_boxes = Boxes(
            p.proposal_boxes.tensor.new_tensor(
                [w * (1. - f) / 2., 
                    h * (1. - f) / 2.,
                    w * (1. - (1. - f) / 2.), 
                    h * (1. - (1. - f) / 2.)]
                ).view(n, 4))
        image_box.objectness_logits = p.objectness_logits.new_ones(n)
        return Instances.cat([p, image_box])


    def _get_empty_mask_loss(self, features, proposals, device):
        if self.mask_on:
            return {'loss_mask': torch.zeros(
                (1, ), device=device, dtype=torch.float32)[0]}
        else:
            return {}


    def _create_proposals_from_boxes(self, boxes, image_sizes, logits):
        """
        Add objectness_logits
        """
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size, logit in zip(
            boxes, image_sizes, logits):
            boxes_per_image.clip(image_size)
            if self.training:
                inds = boxes_per_image.nonempty()
                boxes_per_image = boxes_per_image[inds]
                logit = logit[inds]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            prop.objectness_logits = logit
            proposals.append(prop)
        return proposals


    def _run_stage(self, features, proposals, stage, \
        classifier_info=(None,None,None)):
        """
        Support classifier_info and add_feature_to_prop
        """
        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)

        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        if stage == self.num_cascade_stages - 1 and self.save_feature:
            self.save_bbox_features = box_features
        box_features = self.box_head[stage](box_features)

        return box_features

