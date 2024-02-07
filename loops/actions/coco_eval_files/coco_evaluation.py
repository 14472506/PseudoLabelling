"""
Details

TODO on in this file, detial what functions do so that code is easier to read
"""
# ===========================
# import libraries/packages
# ===========================
# torch imports
import torch
import torchvision

# pycocotools imports
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import pycocotools.mask as mask_util
from .custom_coco_eval import COCOeval

# general imports
import copy
import io
import numpy as np
from contextlib import redirect_stdout
import time
import json

# from packages
from . import coco_utils as utils
from .coco_utils import convert_to_coco_api

# ===========================
# evaluate function and supporting functions
# ===========================
# evaluation
@torch.inference_mode()
def evaluate(model, data_loader, device, save_path, train_flag=False, test_flag=False):
    """
    details 
    """
    # num threads for parralelizing CPU operations
    n_threads = torch.get_num_threads()
    
    # setting number of threads
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    # setting model to evaluate
    model.eval()

    # preparing coco evaluator, providing coco formated dataset and a list of iou_types
    coco = convert_to_coco_api(data_loader.dataset)
    iou_types = _get_iou_types(model)
    #iou_types.append("segm") # ADDED TO MAKE THIS WORK !!!! 
    coco_evaluator = CocoEvaluator(save_path, coco, iou_types)
    
    # carrying out evaluation over dataset
    #for images, targets, something, something_else in data_loader:
    for images, targets, _, _ in data_loader:
        # as with training loop address this is in the collate function

        images = list(img.to(device) for img in images)

        # sycronize with cuda if avaailable
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

        # getting predictions from model and loading them to gpu
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # 
        res = {target["image_id"].detach().item(): output for target, output in zip(targets, outputs)}
        del outputs
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
    
    # sychronize processes?
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()

    # getting mAP for validation whilst training
    if train_flag:
        mAP_val = coco_evaluator.mAP_return()
        return(mAP_val)
    
    # getting coco evaluation in test stage
    if test_flag:
        coco_evaluator.custom_evaluation()
        coco_evaluator.summarize()

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types
 
# ===========================
# coco evaluator class and supporting functions
# ===========================
class CocoEvaluator:
    """
    Detials
    """
    def __init__(self, save_path, coco_gt, iou_types):
        """
        Details
        """
        # checking the iou types are present and in correct format
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"This constructor expects iou_types of type list or tuple, instead  got {type(iou_types)}")
         
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(save_path, coco_gt, iouType=iou_type)

        self.save_path = save_path

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        """
        Details
        """
        # get unique ids for predicitions
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        # iterate through iou types
        for iou_type in self.iou_types:
            # format results from predictions
            results = self.prepare(predictions, iou_type)

            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)

            img_ids, eval_imgs = coco_evaluate(coco_eval)
            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        """
        details
        """
        for iou_type in self.iou_types:
            print(iou_type)
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def mAP_return(self):
        for iou_type, coco_eval in self.coco_eval.items():
            if iou_type == "segm":
                mAP_val = coco_eval.mAP_return()
                return mAP_val

    def custom_evaluation(self):
        pr_dict = {}
        for iou_type, coco_eval in self.coco_eval.items():
            pr_dict[iou_type] = coco_eval.custom_evaluation()
        
        file = self.save_path + "/precision_recall_results.json"
        with open(file, 'w') as f:
            json.dump(pr_dict, f)    

    def prepare(self, predictions, iou_type):
        """
        detials
        """
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        """
        Details
        """
        coco_results = []

        # iterate through 
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            
            # get boxes, scores, and labels data for prediction 
            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            # add current prediction to coco_results
            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        """
        Details
        """
        coco_results = []

        # iterate through predictions
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            
            # collect data for scores, labels, and masks 
            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            # reduce masks with threshold lower than 50%
            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            
            # format masks to run lenght encoding
            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            # collect rle's
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            # add current predictions to coco results
            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def merge(img_ids, eval_imgs):
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs

def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

def coco_evaluate(imgs):
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))

