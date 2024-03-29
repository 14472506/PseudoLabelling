"""
Detials 
"""
# imports
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torchvision.transforms as T
import numpy as np
import PIL
from .oba_maker import OBAMaker
        
class InstanceWrapper(torch.utils.data.Dataset):
    """ detials """
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        """Details"""
        # Getting image
        mrcnn_tensor, mrcnn_target = self.dataset[idx]

        # converting tensors to arrays
        to_img = T.ToPILImage()        
        mrcnn_img = to_img(mrcnn_tensor)
        mrcnn_arr = np.array(mrcnn_img)

        np_masks = []
        for mask, box in zip(mrcnn_target["masks"], mrcnn_target["boxes"]): 
            mask_img = to_img(mask)
            # append values to accumulated lists
            np_masks.append(np.array(mask_img))

        # applying augmentations
        aug_data = self.transforms(image=mrcnn_arr, masks=np_masks)

        boxes_list = []
        for mask in aug_data["masks"]:
            box = self._mask_to_bbox(mask)
            if box == None:
                pass
            else:
                boxes_list.append(box)

        # extracting auged data
        mrcnn_transformed = torch.from_numpy(aug_data["image"])
        mrcnn_transformed = mrcnn_transformed.permute(2,0,1)
        mrcnn_transformed = mrcnn_transformed.to(dtype=torch.float32) / 255.0
        mrcnn_target["masks"] = torch.stack([torch.tensor(arr) for arr in aug_data["masks"]])
        mrcnn_target["boxes"] = torch.as_tensor(boxes_list, dtype=torch.float32)
        
        return mrcnn_transformed, mrcnn_target
    
    def _mask_to_bbox(self, binary_mask):
        """ Details """
        # Get the axis indices where mask is active (i.e., equals 1)
        rows, cols = np.where(binary_mask == 1)

        # If no active pixels found, return None
        if len(rows) == 0 or len(cols) == 0:
            return None

        # Determine the bounding box coordinates
        x_min = np.min(cols)
        y_min = np.min(rows)
        x_max = np.max(cols)
        y_max = np.max(rows)

        return [x_min, y_min, x_max, y_max]

    def __len__(self):
        """ Details """
        return len(self.dataset)

class OBAInstanceWrapper(torch.utils.data.Dataset):
    """ detials """
    def __init__(self, dataset, transforms, back_source, instance_source=None ,prob=0.5, type="simple"):
        self.dataset = dataset
        self.transforms = transforms
        self.oba_maker = OBAMaker(back_source, instance_source, prob)
        self.type = type

    def __getitem__(self, idx):
        """Details"""
        # Getting image
        image_tensor, target_tensors = self.dataset[idx]

        # converting tensors to arrays
        to_img = T.ToPILImage()        
        image_pil = to_img(image_tensor)

        # Just basic oba atm
        if self.type == "basic":
            image_pil = self.oba_maker.basic(image_pil, target_tensors)
        if self.type == "complex":
            image_pil, target_tensors = self.oba_maker.complex(image_pil, target_tensors)
        
        image_arr = np.array(image_pil)
        np_masks = []
        for mask, box in zip(target_tensors["masks"], target_tensors["boxes"]): 
            mask_img = to_img(mask)
            # append values to accumulated lists
            np_masks.append(np.array(mask_img))


        # applying augmentations
        aug_data = self.transforms(image=image_arr, masks=np_masks)

        boxes_list = []
        for mask in aug_data["masks"]:
            box = self._mask_to_bbox(mask)
            if box == None:
                pass
            else:
                boxes_list.append(box)

        # extracting auged data
        image_transformed = torch.from_numpy(aug_data["image"])
        image_transformed = image_transformed.permute(2,0,1)
        image_transformed = image_transformed.to(dtype=torch.float32) / 255.0
        target_tensors["masks"] = torch.stack([torch.tensor(arr) for arr in aug_data["masks"]])
        target_tensors["boxes"] = torch.as_tensor(boxes_list, dtype=torch.float32)
        
        return image_transformed, target_tensors
    
    def _mask_to_bbox(self, binary_mask):
        """ Details """
        # Get the axis indices where mask is active (i.e., equals 1)
        rows, cols = np.where(binary_mask == 1)

        # If no active pixels found, return None
        if len(rows) == 0 or len(cols) == 0:
            return None

        # Determine the bounding box coordinates
        x_min = np.min(cols)
        y_min = np.min(rows)
        x_max = np.max(cols)
        y_max = np.max(rows)

        return [x_min, y_min, x_max, y_max]

    def __len__(self):
        """ Details """
        return len(self.dataset)

def wrappers(model_type):
    """ Detials """
    transform_select = {
        "mask_rcnn": OBAInstanceWrapper, #InstanceWrapper,
        "dual_mask_multi_task": InstanceWrapper
    }
    return transform_select[model_type]

