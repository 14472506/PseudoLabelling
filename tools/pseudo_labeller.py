"""
Module Detials:
This module uses a specified model to produced a pseudo labelled 
data which can be used in semi supervised training in a multi task
model training process.
"""
# import
# base packages
import os
import gc
import json

# third party packages
from PIL import Image, ImageDraw
import torch
from torchvision.transforms import functional as F
import numpy as np
from pycocotools import mask as M
from skimage import measure

# local packages
from models import Models
from loops import Logs

# class
class PseudoLabeller():
    """ Detials """
    def __init__(self, cfg, data_path):
        """ Detials """
        self.cfg = cfg
        self.data_path = data_path

        # extract config
        self._extract_cfg()

        # initialise labelling compononet
        self._initialise_model()
        self._initialise_optimiser()
        self._initialise_logs()
        self._load_model()
        
    def _extract_cfg(self):
        """ Details """
        try:
            self.device = self.cfg["loops"]["device"]
            self.model_cfg = self.cfg["model"]
            self.optimiser_cfg = self.cfg["optimiser"]
            self.logs_cfg = self.cfg["logs"]
        except KeyError as e:
            raise KeyError(f"Missing necessary key in configuration: {e}")

    def _initialise_model(self):
        """
        Initialises the model based off the provided confign
        ensuring the model is sent to the specified device for
        testing
        """
        self.model = Models(self.model_cfg).model()
        self.model.to(self.device)

    def _initialise_optimiser(self):
        """
        Initialises step attribute for edentifying if a scheduler has been
        used in the training process. This is used for determining weather
        pre step of pre and post step models need to be evaluated.
        """
        self.step = True if self.optimiser_cfg["sched_name"] else False

    def _initialise_logs(self):
        """ Initialises the logger module for testing """
        self.logger = Logs(self.logs_cfg)

    def _load_model(self):
        """ Load the trained inference model """
        #if self.step:
        model_type = "pre"
        #else:
        #    model_type = "post"
        self.logger.load_model(self.model, model_type)

    def label(self):
        """ Initialising the testing action to be carry out model testing """
        # set model to eval
        self.model.eval()

        # Initialize COCO-style data structure
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        # adding category to coco data
        category_data = {
            "id": 1,
            "name": "Pseudo Royal",
            "supercategory": "",
            "color": "#e24b1d",
            "metadata": {},
            "keypoint_colours": []
        }
        coco_data["categories"].append(category_data)

        # defining id counts
        annotation_id = 0
        image_id = 0

        # for json naming conv
        init_id = 0
        dif = 0

        for filename in os.listdir(self.data_path):
            # get an image from the target directory or skip to the next loop
            if filename.endswith(".jpg"):
                image_path = os.path.join(self.data_path, filename)
                image = Image.open(image_path).convert("RGB")
                input_tensor = F.to_tensor(image).unsqueeze(0)
                input_tensor = input_tensor.to(self.device)

                width, height = image.size 
                # get outputs from model
                with torch.no_grad():
                    prediction = self.model(input_tensor, mode="sup")
                outputs = self._filter_predictions(prediction)

                del prediction, input_tensor
                torch.cuda.empty_cache()

                # mask gen: make new image here
                ann_mask = Image.new("L", (width, height), 0)

                # make coco dataset 
                annotations_flag = False
                for i in range(len(outputs["labels"])):
                    mask = outputs['masks'][i].numpy()
                    anns, area = self._mask_to_polygon(mask)

                    if not anns:
                        continue

                    if anns:
                        annotations_flag = True

                    # get mask from annotation
                    # mask gen: comment out below line. adding a new drawn region to the existing ann mask, does this work?
                    #ann_mask = Image.new("L", (width, height), 0)
                    ImageDraw.Draw(ann_mask).polygon(anns[0], outline=255, fill=255)
                    #np_mask = np.array(ann_mask)
                    #print(np.unique(np_mask))
                    #extracted_region = Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), ann_mask)

                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(outputs['labels'][i]),
                        "segmentation": anns,
                        "area": int(area),
                        "bbox":	outputs['boxes'][i].tolist(),
                        "iscrowd": False,
                        "isbbox": False,
                        "color": "#0dd36a",
                        "keypoints": [],
                        "metadata": {}
                        }
                    coco_data["annotations"].append(annotation)
                
                    del mask, anns, area
                    torch.cuda.empty_cache()

                    annotation_id += 1
                
                if annotations_flag:
                    # mask gen: new gt image is generate from collected images
                    new_gt_img = Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), ann_mask)
                    #new_gt_img = Image.blend(image.convert("L"), ann_mask, alpha=0.5)
                    #print("Number of non-zero pixels in ann_mask:", np.sum(np.array(ann_mask)))

                    # mask gen: save the image below to a target directory along with modifications to the image data below
                    # this is yet to be done. 
                    pseudo_label_path = "data_handling/sources/pseudo_labels"
                    new_gt_img.save(os.path.join(pseudo_label_path, filename))

                    image.save("base_img.png")  # This will open the image using the default image viewer.
                    ann_mask.save("masked_img.png")
                    new_gt_img.save("maked_img.png")


                    image_data = {
                        "id": image_id,
                        "dataset_id": 1,
                        "path": self.data_path,
                        "width": width,
                        "height": height,
                        "file_name": filename
                        }
                    coco_data["images"].append(image_data)
                else:
                    print("no anns")
                
                del image
                torch.cuda.empty_cache()
                
                print(image_id)
                image_id += 1

                if (image_id - init_id) == 7000 or image_id == 10000 or image_id == 10025:
                    # save labels 
                    self._save_json(init_id, image_id, coco_data)

                    # reset cocodata
                    coco_data = {
                        "images": [],
                        "annotations": [],
                        "categories": []
                    }
                    coco_data["categories"].append(category_data)

                    # update init id
                    init_id = image_id

    def _filter_predictions(self, predictions, threshold=0.9):
        """ Filters the predicted masks to get the masks and meta data from the model """
        # get filtered masks
        masks = (predictions[0]["masks"] > 0.5).float()

        valid_indices = torch.nonzero(predictions[0]["scores"] > threshold).squeeze(1)

        masks = masks[valid_indices]
        boxes = predictions[0]["boxes"][valid_indices]
        labels = predictions[0]["labels"][valid_indices]
        scores = predictions[0]["scores"][valid_indices]

        outputs = {
            "masks": masks.detach().cpu(),
            "boxes": boxes.detach().cpu(),
            "labels": labels.detach().cpu(),
            "scores": scores.detach().cpu()
        }

        return(outputs)

    def _mask_to_polygon(self, mask):
        """ 
        requires a mask as an input and returns the polyon, in adition the function also
        returns the area.

        mask input in must be binary and numpy array 
        """
        mask = mask[0].astype(np.uint8)

        # area
        fortran_mask = np.asfortranarray(mask)
        encoded_mask = M.encode(fortran_mask)
        area = M.area(encoded_mask)

        # annotrations
        proposals = []
        contours = measure.find_contours(mask)
        annotations = []
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist(),
            proposals.append(segmentation[0])
        #print("### Props ###")
        #print(len(proposals))
        try:
            longest_proposal = max(proposals, key=len)
            annotations.append(longest_proposal)
        except ValueError:
            annotations = None
            area = None
        return annotations, area
    
    def _save_json(self, img_id, init_id, data):
        """ Detials """
        file_title = str(init_id) + "_" + str(img_id) + "_img_labels.json"
        with open(file_title, "w") as file:
            json.dump(data, file)