"""
Detials
"""
# imports
import albumentations as A 
from PIL import Image

# class
class Transforms():
    """ Detials """
    def __init__(self, cfg):
        """ Detials """ 
        self.cfg = cfg
        self.model = self.cfg["model_name"]

    def transforms(self):
        """ Detials """
        transform_selector = {
            "mask_rcnn": self._maskrcnn_tranforms,
            "dual_mask_multi_task": self._polite_teacher_transforms,
            "polite_teacher_mask_rcnn": self._polite_teacher_transforms
        }
        return transform_selector[self.model]()

    def _polite_teacher_transforms(self):
        """ Detials """
        light_transforms = A.Compose([
            A.HorizontalFlip(p=0.5)
        ], p=1, 
        additional_targets={'image0': 'image'})

        heavy_transforms = A.Compose([
            A.RandomBrightnessContrast(p=0.3),
            A.ToGray(p=0.5)
        ], p=1)

        return {"light": light_transforms, "heavy": heavy_transforms}

    def _maskrcnn_tranforms(self):
        """ Detials """
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.3),
                A.ToGray(p=0.3)
            ], p=1)
            #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=25, p=0.3)
        ], p=1,
        additional_targets={'image0': 'image'})
        return transforms

    def _multitask_transforms(self):
        """ Detials """
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.3),
                A.ToGray(p=0.3)
            ], p=0.2)
            #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=25, p=0.3)
        ], p=1,
        additional_targets={'image0': 'image'})
        return transforms

