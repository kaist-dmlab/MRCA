import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# Paths to mask folder
# MASK_DIR = "/data3/objdet/lvis_feedback/segmented"
# OUTPUT_JSON = "/data3/objdet/lvis_feedback/annotations"

MASK_DIR = "/data3/objdet/lvis_feedback_sd3/segmented"
OUTPUT_JSON = "/data3/objdet/lvis_feedback_sd3/annotations"


# MASK_DIR = "/data3/objdet/pascal_feedback_sd3/segmented"
# OUTPUT_JSON = "/data3/objdet/pascal_feedback_sd3/annotations"

def get_image_info(file_name, image_id, height, width):
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    }

def get_annotation_info(mask, image_id, category_id, annotation_id):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:  # Ensure valid polygon
            segmentation.append(contour)
    
    if not segmentation:
        return None  # Ignore empty masks
    
    x, y, w, h = cv2.boundingRect(mask)
    area = int(np.sum(mask > 0))
    
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "bbox": [x, y, w, h],
        "area": area,
        "iscrowd": 0
    }

def create_annotations(rd = 0):
    images = []
    annotations = []
    image_id = 0
    annotation_id = 0
    cur_mask_dir = MASK_DIR
    
    for file_name in tqdm(os.listdir(cur_mask_dir)):
        if not file_name.endswith(".png"):
            continue
        
        rdNum, class_id, instance_id = file_name.split("_")
        class_id = int(class_id)
        
        mask_path = os.path.join(cur_mask_dir, file_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape[:2]
        images.append(get_image_info(file_name, image_id, height, width))
        

        
        annotation = get_annotation_info(mask, image_id, class_id, annotation_id)
        if annotation:
            annotations.append(annotation)
            annotation_id += 1
        
        image_id += 1
    
    coco_format = {
        "images": images,
        "annotations": annotations
    }

    json_rd = OUTPUT_JSON + '/rd' + str(rd) + '_annotations.json'
    json_done_path = OUTPUT_JSON + '/rd' + str(rd) + 'done.txt'

    if not os.path.exists(OUTPUT_JSON): 
        os.mkdir(OUTPUT_JSON)
    
    with open(json_rd, "w") as f:
        json.dump(coco_format, f, indent=4)
    
    open(json_done_path, 'a').close()

    
    print(f"COCO annotations saved to {json_rd}")
    
    
def create_annotations_nocum(rd = 0):
    images = []
    annotations = []
    image_id = 0
    annotation_id = 0
    cur_mask_dir = MASK_DIR
    
    for file_name in tqdm(os.listdir(cur_mask_dir)):
        if not file_name.endswith(".png"):
            continue
        
        rdNum, class_id, instance_id = file_name.split("_")
        if rdNum != f'{rd:02}':
            continue
        class_id = int(class_id)
        
        mask_path = os.path.join(cur_mask_dir, file_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape[:2]
        images.append(get_image_info(file_name, image_id, height, width))
        

        
        annotation = get_annotation_info(mask, image_id, class_id, annotation_id)
        if annotation:
            annotations.append(annotation)
            annotation_id += 1
        
        image_id += 1
    
    coco_format = {
        "images": images,
        "annotations": annotations
    }

    json_rd = OUTPUT_JSON + '/rd' + str(rd) + '_annotations.json'
    json_done_path = OUTPUT_JSON + '/rd' + str(rd) + 'done.txt'

    if not os.path.exists(OUTPUT_JSON): 
        os.mkdir(OUTPUT_JSON)
    
    with open(json_rd, "w") as f:
        json.dump(coco_format, f, indent=4)
    
    open(json_done_path, 'a').close()

    
    print(f"COCO annotations saved to {json_rd}")

