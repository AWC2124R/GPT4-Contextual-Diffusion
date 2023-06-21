import os
import sys
import PIL
import keras
import numpy as np
import skimage
import tensorflow as tf

from SD import diffusion_model
from GPT4 import llm_call
from MRCNN import instance_segmentation

ROOT_DIR = ".\\MRCNN\\MRCNN-TF1"
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR + "\\samples\\pastel-mix")

import utils
import visualize
import model as modellib
import pastelmix
from visualize import display_images
from model import log

RECURSION_DEPTH = 1
INITIAL_USER_PROMPT = "1girl, full body, city background"
DIFFUSION_IMAGE_PATH = '.\\images'
MRCNN_MODEL_DIR = ".\\MRCNN\\model_weights"
MRCNN_MODEL_WEIGHTS_PATH = ".\\MRCNN\\model_weights\\pastel_mix_weights.h5"

config = pastelmix.PastelMixConfig()
class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
DEVICE = "/cpu:0"

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MRCNN_MODEL_DIR, config=config)
weights_path = MRCNN_MODEL_WEIGHTS_PATH
model.load_weights(weights_path, by_name=True)

diffusion_model.initialize_model_options()
diffusionImage, imageInfo = diffusion_model.generate_new_image(INITIAL_USER_PROMPT)
diffusionImage.save('images\\val\\initial_diffusion_generation.png', pnginfo = imageInfo)

for depth in range(RECURSION_DEPTH):
    instanceSeg = instance_segmentation.segment_instances(model, config, DIFFUSION_IMAGE_PATH)

    # subPrompts = llm_call.call_gpt4(instanceSeg['detectedClasses'], INITIAL_USER_PROMPT)

    # for segmentedMask, subPrompt in zip(instanceSeg['segmentedMasks'], subPrompts):
    #     diffusionImage = diffusion_model.inpaint_image(subPrompt, segmentedMask, diffusionImage)

    # diffusionImage.save('images\\val\\initial_diffusion_generation.png', pnginfo = imageInfo)