"""
Module Detials:
This module implements a multi task mask r-cnn model. The model has been
modified to contrain two roi heads, with the forward modified so that
depending on the provided inputs, either the target mask r-cnn or pseudo 
labelled mask r-cnn model is used to provide losses to be back propagated
through the network

Note, if this doesnt work, the implementation may need to be moved further back
modifying the outputs from the rpn in the same way. this should still be done to
test the difference in performance regardless
"""
# imports
# base packages
from typing import Dict, List, Optional, Tuple

# third party packages
import torch
import torch.nn as nn

import torchvision
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNNHeads, MaskRCNN
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, maskrcnn_inference, maskrcnn_loss, keypointrcnn_loss, keypointrcnn_inference

# local packages


# classes
class Multi_RoIHeads(RoIHeads):
    def __init__(self, second_box_predictor=None, second_mask_predictor=None, **kwargs):
        super().__init__(**kwargs)
        
        self.second_box_predictor = second_box_predictor
        self.second_mask_predictor = second_mask_predictor

        self.first_box_flag = True 
        self.first_mask_flag = True
        
    def forward(
        self,
        features,  
        proposals,  
        image_shapes,
        targets=None,  
    ):
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        # DOUBLE BOX HEAD MODIFICATION  <------------------------------------------------------------------------------------------
        if not self.first_box_flag:
            class_logits, box_regression = self.box_predictor(box_features)
        else:
            class_logits, box_regression = self.second_box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                # DOUBLE MASK HEAD MODIFICATION  <------------------------------------------------------------------------------------------
                if self.first_mask_flag: 
                    mask_logits = self.mask_predictor(mask_features)
                else: 
                    mask_logits = self.second_mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                if targets is None or pos_matched_idxs is None:
                    raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
            losses.update(loss_keypoint)

        return result, losses
    
class DualMaskRCNN(MaskRCNN):
    """ 
    Dual Mask R-CNN inherits mask rcnn and adds the modified roi heads and second box and mask classifier 
    prediction heads to the model using the modified roi 
    """
    def __init__(
        self,
        backbone,
        num_classes=91,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        # Mask parameters
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        # Adding Dual Mask and Box Predictors
        second_box_predictor=None,
        second_mask_predictor=None,
        **kwargs,      
        ):

        super(DualMaskRCNN, self).__init__(
            backbone,
            num_classes,
            # transform parameters
            min_size,
            max_size,
            image_mean,
            image_std,
            # RPN parameters
            rpn_anchor_generator,
            rpn_head,
            rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_score_thresh,
            # Box parameters
            box_roi_pool,
            box_head,
            box_predictor,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            # Mask parameters
            mask_roi_pool,
            mask_head,
            mask_predictor,
            # Adding Dual Mask and Box Predictors
            second_box_predictor=None,
            second_mask_predictor=None,
            **kwargs, 
            ) 
        
        # For Faster R-CNN init
        out_channels = backbone.out_channels

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        if second_box_predictor is None:
            representation_size = 1024
            second_box_predictor = FastRCNNPredictor(representation_size, num_classes)
        
        if second_mask_predictor is None:
            mask_predictor_in_channels = 256  # == mask_layers[-1]
            mask_dim_reduced = 256
            second_mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)

        # For Mask R-CNN
        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_predictor_in_channels = 256  # == mask_layers[-1]
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)

        # Custom RoI Heads
        self.roi_heads = Multi_RoIHeads(
            # Box
            box_roi_pool = box_roi_pool,
            box_head = box_head,
            box_predictor = box_predictor,
            fg_iou_thresh = box_fg_iou_thresh,
            bg_iou_thresh = box_bg_iou_thresh,
            batch_size_per_image = box_batch_size_per_image,
            positive_fraction = box_positive_fraction,
            bbox_reg_weights = bbox_reg_weights,
            score_thresh = box_score_thresh,
            nms_thresh = box_nms_thresh,
            detections_per_img = box_detections_per_img,
            mask_roi_pool = mask_roi_pool,
            mask_head = mask_head,
            mask_predictor = mask_predictor,
        )

        self.roi_heads.second_box_predictor = second_box_predictor
        self.roi_heads.second_mask_predictor = second_mask_predictor

    def forward(self, images, targets=None, mode="sup"):
        self._set_mode(mode)
        return super().forward(images, targets)

    def _set_mode(self, mode):
        if mode == "sup":
            self.roi_heads.first_box_flag = True
            self.roi_heads.first_mask_flag = True
        elif mode == "ssl":
            self.roi_heads.first_box_flag = False
            self.roi_heads.first_mask_flag = False
        else:
            raise ValueError("mode should be \"sup\" or \"ssl\" in forward")
            
# functions
def dual_mask_resnet50_fpn(cfg):
    """
    function uses the defualt mask r-cnn class with the modified roi heads to
    to add the aditional box, class, and mask prediction heads to the model.
    """
    # confing argument 
    backbone_type = cfg["params"]["backbone_type"]
    drop_out = cfg["params"]["drop_out"]
    batch_norm = cfg["params"]["batch_norm"]
    trainable_layers = cfg["params"]["trainable_layers"]
    num_classes = cfg["params"]["num_classes"]
    hidden_layers = cfg["params"]["hidden_layers"]
    min_size = cfg["params"]["min_size"]
    max_size = cfg["params"]["max_size"]

    # backbone setup
    if backbone_type == "pre-trained":
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                    backbone_name="resnet50",
                    weights="ResNet50_Weights.DEFAULT",
                    trainable_layers=trainable_layers)
    else:
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                    backbone_name="resnet50",
                    weights=False,
                    trainable_layers=trainable_layers)
    if drop_out:
        backbone.body.layer4.add_module("dropout", nn.Dropout(drop_out))

    # model setup
    model = DualMaskRCNN(backbone, num_classes)
        
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.second_box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask_features, hidden_layers, num_classes)
    model.roi_heads.second_mask_predictor = MaskRCNNPredictor(in_mask_features, hidden_layers, num_classes)

    return model