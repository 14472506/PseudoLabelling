""" 
===================================================================================================
Detials
===================================================================================================
"""
# imports
import os 
import random
from PIL import Image, ImageDraw
import numpy as np

# class
class OBAMaker():
    """ Detials """
    def __init__(self, backg_root, p=0):
        """ detials """
        # init vars
        self.backg_root = backg_root
        self.p = p

        # generate background file list
        self.backg_img_list = self._img_dir_lister()

    
    def basic(self, image, targets):
        """ Detials """
        if random.random() <= self.p:
            # get image data
            base_width, base_height = image.size
            backg_image = Image.open(os.path.join(self.backg_root ,random.choice(self.backg_img_list)))
            back_width, back_height = backg_image.size

            base_aspect_ratio = base_width/base_height
            back_aspect_ratio = back_width/back_height
            if back_aspect_ratio != base_aspect_ratio:
                left, top, right, bottom = self._resize_aspect_dims(back_width, back_height, base_aspect_ratio)
                backg_image = backg_image.crop((left, top, right, bottom))
                backg_image = backg_image.resize((base_width, base_height), Image.ANTIALIAS)
            
            for mask in targets["masks"]:
                backg_image.paste(image, (0, 0), Image.fromarray(np.array(mask)*255))
                
            return backg_image
        else:
            return image

    def _img_dir_lister(self):
        """ detials """
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')
        return [f for f in os.listdir(self.backg_root) if f.lower().endswith(image_extensions)]
    
    def _resize_aspect_dims(self, width, height, aspect_ratio):
        """ Detials """
        copy_height = height
        copy_width = copy_height * aspect_ratio
        if copy_width > width:
            copy_width = width
            copy_height = copy_width / aspect_ratio
        
        left = (width - copy_width)/2
        top = (height - copy_height)/2
        right = (width + copy_width)/2
        bottom = (height + copy_height)/2

        return left, top, right, bottom


