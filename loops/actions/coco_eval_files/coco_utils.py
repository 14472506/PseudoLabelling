"""
Detials
"""
# imports
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import torch.distributed as dist

# functions
# ============================================ #
# added in by me
def convert_to_coco_api(ds):
    """
    details
    """
    # initialise coco requirments
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()

    # iterate through dataset
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        # get image and target in image idx
        #img, targets, _, _ = ds[img_idx]
        img, targets, _, _ = ds[img_idx]

        # getting image id from target
        image_id = targets["image_id"].item()
        
        # initialise and assemble image dict
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        
        # ad image dict to dataset dict
        dataset["images"].append(img_dict)
        
        # collect bounding box data
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        
        # collecting label, area, and iscrowd data
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()

        # collecting masks
        masks = targets["masks"]
        # make masks Fortran contiguous for coco_mask
        masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)

        # constucting anns
        num_objs = len(bboxes)
        for i in range(num_objs):
            # collecting annotation content in loop
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            
            # appending annotations to dataset
            dataset["annotations"].append(ann)
            ann_id += 1
    
    # complete assembling coco dataset dict
    dataset["categories"] = [{"id": i} for i in sorted(categories)]

    # formating dataset dict with 
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds

def all_gather(data):
    """
    Detials
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list

def get_world_size():
    """
    Detials
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():
    """
    Detials
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True