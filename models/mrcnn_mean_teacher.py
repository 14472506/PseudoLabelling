"""
===================================================================================================
Detials
===================================================================================================
"""
# imports =========================================================================================
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor

# class ===========================================================================================
class MeanTeacherMaskRCNN(nn.Module):
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

# load functions ==================================================================================