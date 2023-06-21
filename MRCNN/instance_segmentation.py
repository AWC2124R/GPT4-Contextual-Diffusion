import os
import re
import sys
import json
import math
import time
import random
import fnmatch
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT_DIR = ".\\MRCNN\\MRCNN-TF1"
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR + "\\samples\\pastel-mix")

import utils
import visualize
import model as modellib
import pastelmix
from visualize import display_images
from model import log

def find_file(directory, extension):
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.' + extension):
            return filename
    return None

def postprocess_classes(instanceSeg):
    return instanceSeg

def segment_instances(model, config, diffusionImagePath):
    instanceSeg = {
        'detectedClasses': [],
        'segmentedMasks': []
    }

    def get_ax(rows=1, cols=1, size=16):
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax

    filename = find_file(diffusionImagePath + '\\val', 'png')

    annotations = {
        filename: {
            "fileref":"",
            "filename":filename,
            "base64_img_data":"",
            "file_attributes":{
            },
            "regions":{
                "0":{
                    "shape_attributes":{
                        "name":"polygon",
                        "all_points_x": [
                            0,
                            1,
                            2
                        ],
                        "all_points_y":[
                            0,
                            1,
                            2
                        ]
                    },
                    "region_attributes": {
                        "label": "Skirt"
                    }
                }
            }
        }
    }

    jsonAnnotations = json.dumps(annotations, indent=4)
    with open(".\\images\\val\\annotationsVGG.json", "w") as outfile:
        outfile.write(jsonAnnotations)

    dataset = pastelmix.PastelMixDataset()
    dataset.load_pastelmix(diffusionImagePath, "val")
    dataset.prepare()

    image_id = dataset.image_ids[0]
    image, image_meta, gt_class_id, gt_bbox, gt_mask=modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    
    results = model.detect([image], verbose=2)

    ax = get_ax(1)
    r = results[0]
    
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")

    instanceSeg = postprocess_classes(instanceSeg)
    return instanceSeg