"""
Mask R-CNN baseline Model
"""
# imports
import torch
import torch.nn as nn
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

def maskrcnn_resnet50_fpn(cfg):
                         
    # confing argument 
    backbone_type = cfg["params"]["backbone_type"]
    trainable_layers = cfg["params"]["trainable_layers"]
    num_classes = cfg["params"]["num_classes"]
    hidden_layers = cfg["params"]["hidden_layers"]
    drop_out = cfg["params"]["drop_out"]
    pt_load = cfg["params"]["ssl_pt"]
    device = cfg["params"]["device"]

    # backbone selecting
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
        
    #if drop_out:
    #    backbone.body.layer4.add_module("dropout", nn.Dropout(drop_out))

    model = torchvision.models.detection.MaskRCNN(backbone, num_classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layers, num_classes)

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