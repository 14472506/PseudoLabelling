"""
===================================================================================================
Detials
===================================================================================================
"""
# imports =========================================================================================
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# torchvision imports
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import misc as misc_nn_ops

# class ===========================================================================================
class PoliteBase(nn.Module):
    """
    Detials
    """
    def __init__(self, teacher_backbone, student_backbone, teacher_rpn, student_rpn, teacher_roi_heads, student_roi_heads, transforms):
        """
        Detials
        """
        super().__init__()
        self.transforms = transforms

        # teacher_elements
        self.teacher_backbone = teacher_backbone
        self.teacher_rpn = teacher_rpn
        self.teacher_roi_heads = teacher_roi_heads

        # student elements
        self.student_backbone = student_backbone
        self.student_rpn = student_rpn
        self.student_roi_heads = student_roi_heads
    
    # forward handling ----------------------------------------------------------------------------
    def forward(self, images, targets=None, forward_type="teacher"):
        """ 
        Detials 
        """
        # Preprocessing: retained orignal images dimensions and reshape images and targets
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transforms(images, targets)

        # execute teacher or student forward based on forward "type"
        if forward_type == "teacher":
            output = self._forward_processing(images, targets, self.teacher_backbone, self.teacher_rpn, self.teacher_roi_heads)
        elif forward_type == "student":
            output = self._forward_processing(images, targets, self.student_backbone, self.student_rpn, self.student_roi_heads)

        # post processing and return
        if self.training:
            losses = {}
            losses.update(output["proposal_losses"])
            losses.update(output["detector_losses"])
            return losses

        detections = self.transforms.postprocess(output["detections"], images.image_sizes, original_image_sizes)
        return detections
        
    def _forward_processing(self, images, targets, backbone, rpn, roi_heads):
        """
        Detials
        """
        features = backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = rpn(images, features, targets)
        detections, detector_losses = roi_heads(features, proposals, images.image_sizes, targets)

        outputs = {
            "proposals": proposals,
            "proposal_losses": proposal_losses,
            "detections": detections,
            "detector_losses": detector_losses
        }

        return outputs
    
    # Exponential Moving Average handling ---------------------------------------------------------
    def ema_update(self, alpha = 0.99):
        """
        Detials
        """
        with torch.no_grad():
            for teacher_params, student_params in zip(self._teacher_params(), self._student_params()):
                teacher_params.data.mul_(alpha).add_(student_params.data, alpha=1.0-alpha)

    def _teacher_params(self):
        """ Details """
        for params in self.teacher_backbone.parameters():
            yield params
        for params in self.teacher_rpn.parameters():
            yield params
        for params in self.teacher_roi_heads.parameters():
            yield params

    def _student_params(self):
        """ Detials """
        for params in self.student_backbone.parameters():
            yield params
        for params in self.student_rpn.parameters():
            yield params
        for params in self.student_roi_heads.parameters():
            yield params

def _default_anchorgen():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)

class PoliteTeacher(PoliteBase):
    """ Detials """
    def __init__(
        self,
        teacher_backbone,
        student_backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        teacher_rpn_anchor_generator=None,
        teacher_rpn_head=None,
        student_rpn_anchor_generator=None,
        student_rpn_head=None,
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
        teacher_box_roi_pool=None,
        teacher_box_head=None,
        teacher_box_predictor=None,
        student_box_roi_pool=None,
        student_box_head=None,
        student_box_predictor=None,
        all_box_score_thresh=0.05,
        all_box_nms_thresh=0.5,
        all_box_detections_per_img=100,
        all_box_fg_iou_thresh=0.5,
        all_box_bg_iou_thresh=0.5,
        all_box_batch_size_per_image=512,
        all_box_positive_fraction=0.25,
        all_bbox_reg_weights=None,
        # Mask head stuff
        teacher_mask_roi_pool=None,
        teacher_mask_head=None,
        teacher_mask_predictor=None,
        student_mask_roi_pool=None,
        student_mask_head=None,
        student_mask_predictor=None,
        **kwargs,
    ):
        
        # FASTER RCNN
        if not hasattr(teacher_backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        if not hasattr(student_backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        if not isinstance(teacher_rpn_anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator or None instead of {type(teacher_rpn_anchor_generator)}"
            ) 
        if not isinstance(student_rpn_anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator or None instead of {type(student_rpn_anchor_generator)}"
            ) 
        
        if num_classes is not None:
            if teacher_box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
            if student_box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if teacher_box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")
            if student_box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")
            
        teacher_out_channels = teacher_backbone.out_channels
        student_out_channels = student_backbone.out_channels

        if teacher_rpn_anchor_generator is None:
            teacher_rpn_anchor_generator = _default_anchorgen()
        if teacher_rpn_head is None:
            teacher_rpn_head = RPNHead(teacher_out_channels, teacher_rpn_anchor_generator.num_anchors_per_location()[0])
        
        if student_rpn_anchor_generator is None:
            student_rpn_anchor_generator = _default_anchorgen()
        if student_rpn_head is None:
            student_rpn_head = RPNHead(student_out_channels, student_rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        teacher_rpn = RegionProposalNetwork(
            teacher_rpn_anchor_generator,
            teacher_rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        student_rpn = RegionProposalNetwork(
            student_rpn_anchor_generator,
            student_rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if teacher_box_roi_pool is None:
            teacher_box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        if student_box_roi_pool is None:
            student_box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if teacher_box_head is None:
            teacher_resolution = teacher_box_roi_pool.output_size[0]
            teacher_representation_size = 1024
            teacher_box_head = TwoMLPHead(teacher_out_channels * teacher_resolution**2, teacher_representation_size)
        if student_box_head is None:
            student_resolution = student_box_roi_pool.output_size[0]
            student_representation_size = 1024
            student_box_head = TwoMLPHead(student_out_channels * student_resolution**2, student_representation_size)

        if teacher_box_predictor is None:
            teacher_representation_size = 1024
            teacher_box_predictor = FastRCNNPredictor(teacher_representation_size, num_classes)
        if student_box_predictor is None:
            student_representation_size = 1024
            student_box_predictor = FastRCNNPredictor(student_representation_size, num_classes)

        teacher_roi_heads = RoIHeads(
            # Box
            teacher_box_roi_pool,
            teacher_box_head,
            teacher_box_predictor,
            all_box_fg_iou_thresh,
            all_box_bg_iou_thresh,
            all_box_batch_size_per_image,
            all_box_positive_fraction,
            all_bbox_reg_weights,
            all_box_score_thresh,
            all_box_nms_thresh,
            all_box_detections_per_img,
        )

        student_roi_heads = RoIHeads(
            # Box
            student_box_roi_pool,
            student_box_head,
            student_box_predictor,
            all_box_fg_iou_thresh,
            all_box_bg_iou_thresh,
            all_box_batch_size_per_image,
            all_box_positive_fraction,
            all_bbox_reg_weights,
            all_box_score_thresh,
            all_box_nms_thresh,
            all_box_detections_per_img,
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        # Mask R-CNN
        if not isinstance(teacher_mask_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"mask_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(teacher_mask_roi_pool)}"
            )
        if not isinstance(student_mask_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"mask_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(student_mask_roi_pool)}"
            )
        
        if num_classes is not None:
            if teacher_mask_predictor is not None:
                raise ValueError("num_classes should be None when mask_predictor is specified")
        if num_classes is not None:
            if student_mask_predictor is not None:
                raise ValueError("num_classes should be None when mask_predictor is specified")  

        if teacher_mask_roi_pool is None:
            teacher_mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
        if student_mask_roi_pool is None:
            student_mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

        if teacher_mask_head is None:
            teacher_mask_layers = (256, 256, 256, 256)
            teacher_mask_dilation = 1
            teacher_mask_head = MaskRCNNHeads(teacher_out_channels, teacher_mask_layers, teacher_mask_dilation)
        if student_mask_head is None:
            student_mask_layers = (256, 256, 256, 256)
            student_mask_dilation = 1
            student_mask_head = MaskRCNNHeads(student_out_channels, student_mask_layers, student_mask_dilation)

        if teacher_mask_predictor is None:
            teacher_mask_predictor_in_channels = 256  # == mask_layers[-1]
            teacher_mask_dim_reduced = 256
            teacher_mask_predictor = MaskRCNNPredictor(teacher_mask_predictor_in_channels, teacher_mask_dim_reduced, num_classes)
        if student_mask_predictor is None:
            student_mask_predictor_in_channels = 256  # == mask_layers[-1]
            student_mask_dim_reduced = 256
            student_mask_predictor = MaskRCNNPredictor(student_mask_predictor_in_channels, student_mask_dim_reduced, num_classes)

        super().__init__(teacher_backbone, student_backbone, teacher_rpn, student_rpn, teacher_roi_heads, student_roi_heads, transform)

        self.teacher_roi_heads.mask_roi_pool = teacher_mask_roi_pool
        self.teacher_roi_heads.mask_head = teacher_mask_head
        self.teacher_roi_heads.mask_predictor = teacher_mask_predictor

        self.student_roi_heads.mask_roi_pool = student_mask_roi_pool
        self.student_roi_heads.mask_head = student_mask_head
        self.student_roi_heads.mask_predictor = student_mask_predictor

class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x
    
class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
    
class MaskRCNNHeads(nn.Sequential):
    _version = 2

    def __init__(self, in_channels, layers, dilation, norm_layer: Optional[Callable[..., nn.Module]] = None):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
            norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
        """
        blocks = []
        next_feature = in_channels
        for layer_features in layers:
            blocks.append(
                misc_nn_ops.Conv2dNormActivation(
                    next_feature,
                    layer_features,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    norm_layer=norm_layer,
                )
            )
            next_feature = layer_features

        super().__init__(*blocks)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(
            OrderedDict(
                [
                    ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
                ]
            )
        )

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)

# load functions ==================================================================================
def polite_teacher_resnet50_fpn(cfg):
    """ Detials """
    backbone_type = cfg["params"]["backbone_type"]
    trainable_layers = cfg["params"]["trainable_layers"]
    num_classes = cfg["params"]["num_classes"]
    hidden_layers = cfg["params"]["hidden_layers"]
    drop_out = cfg["params"]["drop_out"]
    pt_load = cfg["params"]["ssl_pt"]
    device = cfg["params"]["device"]

    # backbone selecting
    if backbone_type == "pre-trained":
        teacher_backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                    backbone_name="resnet50",
                    weights="ResNet50_Weights.DEFAULT",
                    trainable_layers=trainable_layers)
        student_backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                    backbone_name="resnet50",
                    weights="ResNet50_Weights.DEFAULT",
                    trainable_layers=trainable_layers)
    else:
        teacher_backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                    backbone_name="resnet50",
                    weights=False,
                    trainable_layers=trainable_layers)
        student_backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                    backbone_name="resnet50",
                    weights=False,
                    trainable_layers=trainable_layers)
  
    #if drop_out:
    #    backbone.body.layer4.add_module("dropout", nn.Dropout(drop_out))
        
    model = PoliteTeacher(teacher_backbone, student_backbone, num_classes)

    teacher_in_features = model.teacher_roi_heads.box_predictor.cls_score.in_features
    studetn_in_features = model.student_roi_heads.box_predictor.cls_score.in_features

    model.teacher_roi_heads.box_predictor = FastRCNNPredictor(teacher_in_features, num_classes)
    model.student_roi_heads.box_predictor = FastRCNNPredictor(studetn_in_features, num_classes)

    teacher_in_features_mask = model.teacher_roi_heads.mask_predictor.conv5_mask.in_channels
    student_in_features_mask = model.student_roi_heads.mask_predictor.conv5_mask.in_channels

    model.teacher_roi_heads.mask_predictor = MaskRCNNPredictor(teacher_in_features_mask, hidden_layers, num_classes)
    model.student_roi_heads.mask_predictor = MaskRCNNPredictor(student_in_features_mask, hidden_layers, num_classes)

    if pt_load:
        print("loading ssl pre trained weights")
        print("loading :" + pt_load)
        
        ssl_checkpoint = torch.load(pt_load, device)
        ssl_state_dick = ssl_checkpoint["state_dict"]
        backbone_keys = [key for key in ssl_state_dick.keys() if key.startswith("backbone")]
        backbone_state_dict = {k: ssl_state_dick[k] for k in backbone_keys}
        model.load_state_dict(backbone_state_dict, strict=False)

        print("loaded")

    return model