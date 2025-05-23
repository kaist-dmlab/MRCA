# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
import cv2
from typing import List, Optional, Union
import torch
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        inp_dict = {}
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        self.inp_dict = inp_dict
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def __call__(self, dataset_dict, augmentations=None):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        mask_=None
        if 'mask_cat' in dataset_dict:
            file_name=dataset_dict["file_name"].split('|')
            try:
                if len(file_name)==1:
                    image = utils.read_image(file_name[0])
                    mask_=image[:,:,-1]
                    image=image[:,:,:3]
                else:
                    image = utils.read_image(file_name[0])[:,:,:3]
                    mask_= cv2.imread(file_name[1],cv2.IMREAD_UNCHANGED)
                    if mask_.shape==3:
                        mask_=mask_[:,:,-1]
                dataset_dict['height']=image.shape[0]
                dataset_dict['width']=image.shape[1]
            except:
                print('data error ',dataset_dict["file_name"])
                mask_=None
        elif 'image_new' in dataset_dict :
            image = dataset_dict.pop('image_new')
        else :
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
            
        inp_image = None
        if dataset_dict['image_id'] in self.inp_dict :
            inp_image = utils.read_image(self.inp_dict[dataset_dict['image_id']], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            if mask_ is not None:
                sem_seg_gt = mask_
            else:
                sem_seg_gt= None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        if augmentations is not None :
            augmentations = T.AugmentationList(augmentations)
        else :
            augmentations = self.augmentations
        transforms = augmentations(aug_input)
        # if "inp_file_name" in dataset_dict:
        if inp_image is not None :
            inp_image = transforms.apply_image(inp_image)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        if mask_ is not None:
            mask_=sem_seg_gt
            sem_seg_gt=None
        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # if 'inp_file_name' in dataset_dict :
        if inp_image is not None :
            dataset_dict["inp_image"] = torch.as_tensor(np.ascontiguousarray(inp_image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            # dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            if mask_ is not None:
                ###
                boxes = np.zeros((1, 4))
                target = Instances(image_shape)
                target.gt_boxes = Boxes(boxes)
                classes = [dataset_dict.pop("mask_cat")]
                classes = torch.tensor(classes, dtype=torch.int64)
                target.gt_classes = classes
                mask_=(mask_>128)
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in [mask_]])
                )
                target.gt_masks = masks
                instances=target
                ###
            else:
                instances = utils.annotations_to_instances(
                    annos, image_shape, mask_format=self.instance_mask_format
                )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes or mask_ is not None:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            if not dataset_dict['instances'].has('gt_masks'):
            #   dataset_dict["instances"].gt_masks = BitMasks(np.zeros((0,image_shape[0],image_shape[1])))
                num_instances = len(dataset_dict["instances"])
                if num_instances > 0:
                    dataset_dict["instances"].gt_masks = BitMasks(np.zeros((num_instances, image_shape[0], image_shape[1])))
        return dataset_dict
