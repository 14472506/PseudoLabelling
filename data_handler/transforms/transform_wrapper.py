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
    
class PseudodWrapper(torch.utils.data.Dataset):
    """ Detials """
    def __init__(self, dataset, transforms):
        """ Detials """
        self.dataset = dataset
        self.light_transforms = transforms["light"]
        self.heavy_transforms = transforms["heavy"]
        # tensor to PIL conversion
        self.to_img = T.ToPILImage()

    def __getitem__(self, idx):
        """Details"""
        # Getting image
        image_tensor, targets = self.dataset[idx]
        if targets == None:
            aug_img, heavy_aug_img, targets = self._unlabeled_data(image_tensor)
        else:
            aug_img, heavy_aug_img, targets = self._labeled_data(image_tensor, targets)
        return [aug_img, heavy_aug_img, targets]

    def _unlabeled_data(self, image_tensor):
        """ Detials """
        # np image data and blank for augs
        image_array = self._tensor_to_array(image_tensor)
        blank = np.zeros_like(image_array)

        # augmentations
        light_aug_data = self.light_transforms(image=image_array, masks=blank)
        heavy_aug_data = self.heavy_transforms(image=light_aug_data["image"])

        # auged_image_tensors
        light_image_tensor = torch.from_numpy(light_aug_data["image"].transpose((2,0,1))).float().div(255)
        heavy_image_tensor = torch.from_numpy(heavy_aug_data["image"].transpose((2,0,1))).float().div(255)

        return light_image_tensor, heavy_image_tensor, None

    def _labeled_data(self, image_tensor, targets):
        """ Detials """
        # mask and image data to np arrays
        np_masks = []
        for mask in targets["masks"]: 
            np_masks.append(self._tensor_to_array(mask))
        image_array = self._tensor_to_array(image_tensor)

        # augmentations
        light_aug_data = self.light_transforms(image=image_array, masks=np_masks)
        heavy_aug_data = self.heavy_transforms(image=light_aug_data["image"])

        # auged_image_tensors
        light_image_tensor = torch.from_numpy(light_aug_data["image"].transpose((2,0,1))).float().div(255)
        heavy_image_tensor = torch.from_numpy(heavy_aug_data["image"].transpose((2,0,1))).float().div(255)

        boxes_list = []
        for mask in light_aug_data["masks"]:
            box = self._mask_to_bbox(mask)
            if box == None:
                pass
            else:
                boxes_list.append(box)

        targets["masks"] = torch.stack([torch.tensor(arr) for arr in light_aug_data["masks"]])
        targets["boxes"] = torch.as_tensor(boxes_list, dtype=torch.float32)
    
        return light_image_tensor, heavy_image_tensor, targets
        
    def _tensor_to_array(self, image_tensor):
        """ Detials """
        return np.array(self.to_img(image_tensor))
    
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
    def __init__(self, dataset, transforms, back_source, prob):
        self.dataset = dataset
        self.transforms = transforms
        self.oba_maker = OBAMaker(back_source, prob)

    def __getitem__(self, idx):
        """Details"""
        # Getting image
        mrcnn_tensor, mrcnn_target = self.dataset[idx]

        # converting tensors to arrays
        to_img = T.ToPILImage()        
        mrcnn_img = to_img(mrcnn_tensor)

        # Just basic oba atm
        mrcnn_img = self.oba_maker.basic(mrcnn_img, mrcnn_target)

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

def wrappers(model_type):
    """ Detials """
    transform_select = {
        "mask_rcnn": OBAInstanceWrapper, #InstanceWrapper,
        "dual_mask_multi_task": InstanceWrapper,
        "mean_teacher_mask_rcnn": PseudodWrapper
    }
    return transform_select[model_type]

