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
import json
from pycocotools.mask import frPyObjects, decode
import torch
from skimage import measure
import cv2

# class
class OBAMaker():
    """ Detials """
    def __init__(self, backg_root, instance_root, p=0):
        """ detials """
        # init vars
        self.backg_root = backg_root
        self.insance_root = instance_root
        self.p = p
        self.instance_root = instance_root
        if instance_root:
            self.instance_data = self._load_json(os.path.join(instance_root, "instances.json"))

        # generate background file list
        self.backg_img_list = self._img_dir_lister()

    # BASIC =======================================================================================
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
        
    # COMPLEX =====================================================================================
    # base function
    def complex(self, image, targets):
        """ Detials """
        if random.random() <= self.p:
            # decide on injection or synthasis
            if random.random() <= 1:
                image, targets = self._injection(image, targets)
                return image, targets
            else:
                image, targets = self._synthasis(image, targets)
        return image, targets
    
    def _injection(self, image, targets):
        """ Detials """
        return self._get_new_instances(image, targets)

    def _synthasis(self, image, target):
        """ Details """
        pass

    def _get_mask_complexity(self, mask, mask_area):
        """ Detials """
        # computer rather than process to address difference in label quality
        contours = measure.find_contours(mask, 0.2)
        if contours:
            perimeters = measure.approximate_polygon(contours[0], tolerance=0).shape[0]
        complexity_score = 1 / (1 + perimeters / mask_area)
        return complexity_score

    def _get_instance_data(self, selection_log, instance_range):
        """ Detials """
        idx = None
        while idx is None:
            # propose index and check its not been used
            proposed_idx = random.randint(0, instance_range-1)   
            if proposed_idx in selection_log:
                continue
            
            # get proposed image and mask data
            proposed_mask = decode(frPyObjects(self.instance_data["annotations"][proposed_idx]["segmentation"], 
                                               self.instance_data["images"][proposed_idx]["height"], 
                                               self.instance_data["images"][proposed_idx]["width"])) 
            
            # check if mask is multiple parts if so reject, else squeeze
            if proposed_mask.shape[2] > 1:
                selection_log.append(proposed_idx)
                continue
            else:
                proposed_mask = np.squeeze(proposed_mask).astype(np.uint8)          

            # remove based on area
            proposed_mask_area = np.sum(proposed_mask)
            if proposed_mask_area <= 500:
                continue

            complexiy_score = self._get_mask_complexity(proposed_mask, proposed_mask_area)
            if complexiy_score <= 0.97:
                continue

            # checking for black masks which are faulty. this could be removed the issue as to why this happens is found
            proposed_image = Image.open(os.path.join(self.instance_root, self.instance_data["images"][proposed_idx]["file_name"]))
            proposed_img_array = np.array(proposed_image)
            check_roi = proposed_img_array[proposed_mask.astype(bool)]
            black_pixels = np.sum(np.all(check_roi == (0, 0, 0), axis=-1))
            if black_pixels == 0:
                black_percentage = 0
            else:
                black_percentage = black_pixels/check_roi.shape[0] 
            if black_percentage >= 0.5:
                selection_log.append(proposed_idx)
                continue
        
            selection_log.append(proposed_idx)
            idx = proposed_idx
            
        return proposed_image, proposed_mask
    
    def _get_new_instances(self, image, targets, max_instances=100):
        """ Detaisl """
        # instance proposal setup
        selection_count = random.randint(1, min(max_instances - len(targets["labels"]), 10))
        selection_log = []

        # instance injection setup
        instance_range = len(self.instance_data["annotations"])
        mask_array = targets["masks"].numpy()
        num_masks, array_height, array_width = mask_array.shape

        ## image resize
        #width, height =  image.size
        #if width > 640:
        #    
        #    new_size = (int(width*0.74), int(height*0.74))
        #    image = image.resize(new_size)
        #
        #    resized_slices = []
        #    for i in range(mask_array.shape[0]):
        #        slice = mask_array[i,:,:]
        #        print(slice.shape)
        #    #    slice_image = Image.fromarray((slice*255).astype(np.uint8))
        #    #    resized_slice_image = slice_image.resize(new_size, Image.NEAREST)
        #    #    resized_slices.append(np.array(resized_slice_image)/255.0)
        #    #mask_array = np.stack(resized_slices, axis=1) 

        # enter loop of random selection
        for i in range(selection_count):

            # get instance image and mask
            instance_image, instance_mask, = self._get_instance_data(selection_log, instance_range)

            injection_mask = np.zeros((array_height, array_width), dtype=np.uint8)
            fit = False
            while not fit:
                # get max_x and max_y
                max_x = injection_mask.shape[0] - instance_mask.shape[0]
                max_y = injection_mask.shape[1] - instance_mask.shape[1]

                if max_x < 0 or max_y < 0:
                    print(max_x, max_y)
                    inst_width, inst_height = instance_image.size
                    new_size = (int(inst_width*0.75), int(inst_height*0.75))
                    print("instance width, and new_size")
                    print((inst_width, inst_height))
                    print(new_size)
                    instance_image = instance_image.resize(new_size)
                    instance_mask = cv2.resize(instance_mask, new_size, interpolation=cv2.INTER_NEAREST)
                    continue
                fit = True
            
            x_start = np.random.randint(0, max_x+1) 
            y_start = np.random.randint(0, max_y+1)
            injection_mask[x_start:x_start+instance_mask.shape[0], y_start:y_start+instance_mask.shape[1]] = instance_mask

            # evaluate masks
            updated_masks = []
            for i in range(num_masks):
                origenal_mask = mask_array[i,:,:]

                # handling complete oclusion
                if np.array_equal(origenal_mask & injection_mask, origenal_mask):
                    continue
                
                # subtracting coluded region of mask
                updated_mask = origenal_mask & ~injection_mask

                # adding non empty masks 
                if np.any(updated_mask):
                    updated_masks.append(updated_mask)

            updated_masks.append(injection_mask)
            mask_array = np.array(updated_masks) if updated_masks else np.empty((0,) + origenal_mask.shape[1:])
            num_masks, _, _ = mask_array.shape             
            image.paste(instance_image, (y_start, x_start), Image.fromarray(np.array(instance_mask)*255))

        # area screening
        valid_masks = []
        valid_boxes = []
        labels = []
        iscrowd = []
        area_list = []

        for i in range(num_masks):
            mask = mask_array[i,:,:]
            bbox = self._mask_to_bbox(mask)
            box_size = (bbox[2]-bbox[0], bbox[3]-bbox[1])
            if box_size[0] < 5 or box_size[1] < 5:
                continue
            valid_masks.append(mask)            
            valid_boxes.append(bbox)
            labels.append(1)
            iscrowd.append(0)
            area_list.append(np.sum(mask == 1))

        valid_masks = np.array(valid_masks)

        ############################### VIEWING MASKS
        #combined_image = image.convert("RGBA")
        #for mask in mask_array:
        #    mask_img = Image.fromarray(np.uint8(mask * 255)).convert("L")
        #    mask_color = (random.randint(1,255), random.randint(1,255), random.randint(1,255))
        #    colored_mask = Image.new("RGBA", mask_img.size)
        #    for x in range(mask_img.width):
        #        for y in range(mask_img.height):
        #            if mask_img.getpixel((x, y)) > 0:  # Mask pixel is part of the object
        #                colored_mask.putpixel((x, y), mask_color + (100,))  # Semi-transparent
        #    combined_image = Image.alpha_composite(combined_image, colored_mask)
        #combined_image.save("injection_test.png")
        #image = image.convert("RGB")
        ############################### VIEWING MASKS

        targets["masks"] = torch.tensor(valid_masks)
        targets["boxes"] = torch.tensor(valid_boxes)
        targets["labels"] = torch.tensor(labels)
        targets["iscrowd"] = torch.tensor(iscrowd)
        targets["area"] = torch.tensor(area_list)

        return image, targets

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

    # SUPORTING ===================================================================================
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
    
    def _load_json(self, path):
        print(path)
        with open(path, "r") as file:
            return json.load(file)


