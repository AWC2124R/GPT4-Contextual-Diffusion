"""
GPT4 Contextual Diffusion

Module for MRCNN Instance Segmenetation.

Copyright (c) 2023 AWC2124R(Taglink).
Licensed under the MIT License (see LICENSE for details)
Written by Taehoon Hwang
"""

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
from PIL import Image

ROOT_DIR = ".\\MRCNN\\MRCNN-TF1"
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR + "\\samples\\pastel-mix")

import utils
import visualize
import model as modellib
import pastelmix
from visualize import display_images
from model import log

# Define the directory and file paths for the model weights
MODEL_DIR = ".\\MRCNN\\model_weights"
MODEL_WEIGHTS_PATH = ".\\MRCNN\\model_weights\\pastel_mix_weights.h5"

DEVICE = "/cpu:0" 
GPU_COUNT = 1
IMAGES_PER_GPU = 1

IMAGE_MAX_DIM = 1024

# Define a function to find a file with a specific extension within a directory
def find_file(directory, extension):
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.' + extension):
            return filename
    return None

# Define a function to create a matplotlib figure
def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

class MRCNNModel:
    def __init__(self, model_dir=MODEL_DIR, model_weights_path=MODEL_WEIGHTS_PATH,
                 device=DEVICE, gpu_count=GPU_COUNT, images_per_gpu=IMAGES_PER_GPU,
                 image_max_dim=IMAGE_MAX_DIM):

        self.model_dir = model_dir
        self.model_weights_path = model_weights_path
        self.device = device
        self.config = pastelmix.PastelMixConfig()

        self.image_max_dim = image_max_dim
        
        # Set GPU count and images per GPU for the inference configuration
        class InferenceConfig(self.config.__class__):
            GPU_COUNT = gpu_count
            IMAGES_PER_GPU = images_per_gpu
        
        self.config = InferenceConfig() # Replace the config with the inference configuration

        self.model = 0

    def initialize_model(self):
        with tf.device(DEVICE):
            self.model = modellib.MaskRCNN(mode="inference", model_dir=self.model_dir, config=self.config) # Initialize the model in inference mode

        weights_path = self.model_weights_path
        self.model.load_weights(weights_path, by_name=True) # Load the model weights
    
    def postprocess_classes(self, instanceSeg):
        pInstanceSeg = {
            'detectedClasses': [],
            'segmentedMasks': []
        }

        # print(instanceSeg['segmentedMasks'][:, :, 0].shape)

        masks = np.copy(instanceSeg['segmentedMasks'])

        for detClass, segMaskIdx in zip(instanceSeg['detectedClasses'], np.arange(masks.shape[2])):
            if detClass in pInstanceSeg['detectedClasses']:
                idx = pInstanceSeg['detectedClasses'].index(detClass)
                pInstanceSeg['segmentedMasks'][idx] = [[a or b for a, b in zip(sublist1, sublist2)] 
                                                       for sublist1, sublist2 in zip(instanceSeg['segmentedMasks'][:, :, segMaskIdx], pInstanceSeg['segmentedMasks'][idx])]
            else:
                pInstanceSeg['detectedClasses'].append(detClass)
                pInstanceSeg['segmentedMasks'].append(instanceSeg['segmentedMasks'][:, :, segMaskIdx])

        return pInstanceSeg

    def instance_segment(self, diffusionImagePath):
        filename = find_file(diffusionImagePath + '\\val', 'png') # Find the PNG image in the given path

        # Create a dictionary with polaceholder annotations
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

        # Convert annotations dictionary to JSON format and save it to a file
        jsonAnnotations = json.dumps(annotations, indent=4)
        with open(".\\images\\val\\annotationsVGG.json", "w") as outfile:
            outfile.write(jsonAnnotations)

        # Load the dataset using PastelMix dataset class
        dataset = pastelmix.PastelMixDataset()
        dataset.load_pastelmix(diffusionImagePath, "val")
        dataset.prepare()

        # Load image and its ground truth data
        image_id = dataset.image_ids[0]
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, self.config, image_id,
                                                                                  use_mini_mask=False)

        results = self.model.detect([image], verbose=2) # Run object detection with the model

        ax = get_ax(1) 
        r = results[0]

        # Display the object detection results
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    dataset.class_names, r['scores'], ax=ax,
                                    title="Predictions")

        # Create a dictionary containing instance segmentation results
        instanceSeg = {
            'detectedClasses': r['class_ids'],
            'segmentedMasks': r['masks']
        }

        pinstanceSeg = self.postprocess_classes(instanceSeg) # Post-process the instance segmentation results

        return pinstanceSeg