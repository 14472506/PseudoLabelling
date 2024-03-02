""" 
Details
"""
# imports
import os
import json
import numpy as np
from PIL import Image
from pycocotools.mask import frPyObjects, decode

# main
def main(source_dir, json_file):
    """ Detials """
    path = os.path.join(source_dir, json_file)
    with open(path, "r") as file:
        data = json.load(file)

    image_id = None
    image = None
    for anno in data["annotations"]:
        # load new image if image id has changed
        if anno["image_id"] != image_id:
            image_id = anno["image_id"]
            image = Image.open(os.path.join(source_dir ,data["images"][image_id]["file_name"]))
        
        # store annotation
        anno_poly = anno["segmentation"]
        
        # get mask from poly -> to PIL
        width, height = image.size
        rle = frPyObjects(anno["segmentation"], height, width)
        mask = decode(rle)
        mask_2d = mask[:, :, 0]
   
        # get background image
        bbox = find_bbox(mask)
        box_width = bbox[2] - bbox[0]
        box_height = bbox[3] - bbox[1]
        instance_image = Image.new("RGB", (box_width, box_height), (0, 0, 0))

        # creating instance image
        cropped_image = image.crop(bbox)
        mask_image = Image.fromarray(np.uint8(mask_2d*255)).crop(bbox).convert("L")
        instance_image.paste(cropped_image, (0,0), mask = mask_image)

        ## save image
        #id = anno["id"]
        #image_title = f"raw_instances/image_{id}.png"
        #instance_image.save(image_title)   
        #print(image_title)

        new_poly = []
        for poly in anno_poly:
            sub_poly = []
            for i in range(0, len(poly), 2):
                x_adjust = round(poly[i] - bbox[0],1)
                y_adjust = round(poly[i+1] - bbox[1],1)
                if x_adjust < 0:
                    x_adjust = 0
                if y_adjust < 0:
                    y_adjust = 0
                sub_poly.extend([x_adjust, y_adjust])
            new_poly.extend([sub_poly])

        print(new_poly)

def find_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return xmin, ymin, xmax+1, ymax+1  
  
# execute
if __name__ == "__main__":
    source_image_dir = "data_handler/sources/jersey_dataset_v4/train"
    source_json_root = "train.json"
    main(source_image_dir, source_json_root)