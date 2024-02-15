"""
Detials
"""
# imports
import os

import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image

# class
class UnlabeledDataLoader(data.Dataset):
    """ Detials """
    def __init__(self, cfg, type, seed=42):
        """ Detials """
        self.cfg = cfg
        self.type = type
        self._extract_config()
        self._initialize_params()

    def _extract_config(self):
        """Detials"""
        self.source = self.cfg["source"]
        
    def _initialize_params(self):
        """Detials"""
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')
        self.img_list = [f for f in os.listdir(self.source) if f.lower().endswith(image_extensions)]

    def __getitem__(self, idx):
        """Details"""
        # laoding image
        img_id = self.img_list[idx]
        img = Image.open(os.path.join(self.source, img_id)).convert('RGB')
        img_tensor = self._to_tensor(img)

        return img_tensor, None

    def _to_tensor(self, img):
        """ Detials """
        transform = T.ToTensor()
        ten_img = transform(img)
        return ten_img
    
    def __len__(self):
        """ Detials """
        return len(self.img_list)