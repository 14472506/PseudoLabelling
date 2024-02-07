"""
Detials
"""
# imports
import json
import os

import torch
import torchvision.transforms as T
import torch.utils.data as data
import torch.nn.functional as F
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

# class
class COCODataset(data.Dataset):
    """ Detials """
    def __init__(self, cfg, type, seed=42):
        self.cfg = cfg
        self.type = type
        self._extract_config()

        self._initialize_params()

    def _extract_config(self):
        """Detials"""
        self.sup_source = self.cfg["source"]
        self.type_dir = self.cfg["params"][self.type]["dir"]
        self.type_json = self.cfg["params"][self.type]["json"]

    def _initialize_params(self):
        """Detials"""
        if not self.type_dir:
            self.coco = COCO(os.path.join(self.sup_source, self.type_json))
            self.dir = self.sup_source
        else:
            self.coco = COCO(os.path.join(self.sup_source, self.type_dir, self.type_json))
            self.dir = os.path.join(self.sup_source, self.type_dir)
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        """Details"""
        # laoding image
        img_id = self.ids[idx]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.dir, file_name)).convert('RGB')

        mrcnn_target = self._coco_target_collection(img_id, idx)
        mrcnn_tensor = self._to_tensor(img)

        return mrcnn_tensor, mrcnn_target

    def _coco_target_collection(self, img_id, idx):
        """Detials"""
        # collecting ids and annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # initialise target lists and targ dict
        labels, boxes, masks_list, areas, iscrowds = [], [], [], [], []
        target = {}

        for ann in anns:
            # collecting target data
            labels.append(ann["category_id"])
            areas.append(ann["area"]) 
            boxes.append([ann["bbox"][0], ann["bbox"][1], ann["bbox"][0]+ann["bbox"][2], ann["bbox"][1]+ann["bbox"][3]])
            
            # formatting and collecting iscrowd id's
            if ann["iscrowd"]:
                iscrowds.append(1)
            else:
                iscrowds.append(0)

            # formatting mask to tensor and collecting masks
            mask = self.coco.annToMask(ann)
            mask == ann['category_id']
            masks_list.append(torch.from_numpy(mask)) 

        # building target_dict
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["masks"] = torch.stack(masks_list, 0)
        target["image_id"] = torch.tensor([idx])
        target["area"] = torch.as_tensor(areas, dtype=torch.int64)
        target["iscrowd"] = torch.as_tensor(iscrowds, dtype=torch.int64)

        return(target)

    def _to_tensor(self, img):
        transform = T.ToTensor()
        ten_img = transform(img)
        return ten_img
    
    def __len__(self):
        return len(self.ids)
      
def COCO_collate_function(batch):
    """Detials"""
    return tuple(zip(*batch)) 