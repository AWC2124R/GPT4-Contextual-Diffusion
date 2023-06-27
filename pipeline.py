"""
GPT4 Contextual Diffusion

Main Pipeline Module for LLM based Contextual Diffusion.

Copyright (c) 2023 AWC2124R(Taglink).
Licensed under the MIT License (see LICENSE for details)
Written by Taehoon Hwang
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import io
import sys
import PIL
from PIL import Image
import base64
import keras
import numpy as np
import skimage
import tensorflow as tf
import base64
import shutil

from SD import diffusion_model
from GPT4 import llm_call
from MRCNN import instance_segmentation

# Define a configuration class for the overall pipeline
class PipelineConfig:
    IMAGE_SAVE_PATH = ".\\images"  # Directory where all images, both interim and final will be saved
    
    RECURSION_DEPTH = 1

    IMAGE_WIDTH = 1280
    IMAGE_HEIGHT = 720

    USE_UPSCALING = True
    UPSCALE_FACTOR = 2

# Define a configuration class for the Stable Diffusion model
class SDConfig:
    OPTION_PAYLOAD = {
        'sd_model_checkpoint': "pastelmix-better-vae-fp32.ckpt [943a810f75]"  # Specify the model checkpoint for the SD model
    }

    GENERATION_PAYLOAD = {
        'steps': 20,  # Number of steps for the SD model to take during generation

        'width': PipelineConfig.IMAGE_WIDTH / PipelineConfig.UPSCALE_FACTOR if PipelineConfig.USE_UPSCALING else PipelineConfig.IMAGE_WIDTH,
        'height': PipelineConfig.IMAGE_HEIGHT / PipelineConfig.UPSCALE_FACTOR if PipelineConfig.USE_UPSCALING else PipelineConfig.IMAGE_HEIGHT,

        'enable_hr': PipelineConfig.USE_UPSCALING,  # Whether to enable high-resolution generation
        'hr_scale': PipelineConfig.UPSCALE_FACTOR,
        'hr_second_pass_steps': 20,

        'denoising_strength': 0.6  # Strength of the denoising to apply during generation
    }

    INPAINT_PAYLOAD = {
    }

# Define a configuration class for the MRCNN model
class MRCNNConfig:
    MODEL_DIR = ".\\MRCNN\\model_weights"  # Directory where the MRCNN model weights are stored
    MODEL_WEIGHTS_PATH = ".\\MRCNN\\model_weights\\pastel_mix_weights.h5"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    IMAGE_MAX_DIM = 1024

    SEGMENT_CLASSES = ["Skirt", "Shirt", "Hat", "Bag", "Shoe", "Cars", "Buildings",
               "Alleyshops", "Jacket", "Dress", "BodyofWater", "Sky", "Lightsource",
               "Shelfs", "Residentialcomplex", "Brightwindow"]


# Function that maps 2D Boolean Mask to a base64 string
def mask2str(mask):
    maskImg = Image.fromarray((255 * np.array(mask)).astype(np.uint8))
    maskImg.save("test.png")
    
    if PipelineConfig.IMAGE_WIDTH >= PipelineConfig.IMAGE_HEIGHT:
        maskImg = maskImg.resize((PipelineConfig.IMAGE_WIDTH, PipelineConfig.IMAGE_WIDTH))
        maskImg = maskImg.crop((0,
                                PipelineConfig.IMAGE_WIDTH / 2 - PipelineConfig.IMAGE_HEIGHT / 2,
                                PipelineConfig.IMAGE_WIDTH,
                                PipelineConfig.IMAGE_WIDTH / 2 + PipelineConfig.IMAGE_HEIGHT / 2))
    else:
        maskImg = maskImg.resize((PipelineConfig.IMAGE_HEIGHT, PipelineConfig.IMAGE_HEIGHT))
        maskImg = maskImg.crop((PipelineConfig.IMAGE_HEIGHT / 2 - PipelineConfig.IMAGE_WIDTH / 2,
                                0,
                                PipelineConfig.IMAGE_HEIGHT / 2 + PipelineConfig.IMAGE_WIDTH / 2,
                                PipelineConfig.IMAGE_HEIGHT))
    
    bufferBytes = io.BytesIO()
    maskImg.save(bufferBytes, format="PNG")
    img_str = base64.encodebytes(bufferBytes.getvalue()).decode()

    return img_str


# Create instances of the SD and MRCNN models with the appropriate configurations
stableDiffusion = diffusion_model.SDModel(option_payload=SDConfig.OPTION_PAYLOAD,
                                          generation_payload=SDConfig.GENERATION_PAYLOAD,
                                          inpaint_payload=SDConfig.INPAINT_PAYLOAD)
stableDiffusion.initialize_model()  # Initialize the SD model

mrcnn = instance_segmentation.MRCNNModel(model_dir=MRCNNConfig.MODEL_DIR,
                                         model_weights_path=MRCNNConfig.MODEL_WEIGHTS_PATH,
                                         gpu_count=MRCNNConfig.GPU_COUNT,
                                         images_per_gpu=MRCNNConfig.IMAGES_PER_GPU,
                                         image_max_dim=MRCNNConfig.IMAGE_MAX_DIM)
mrcnn.initialize_model()  # Initialize the MRCNN model

# Generate an image using the SD model and save the image information
initialPrompt = "1girl, full body, city background"
diffusionImage, imageInfo, imageStr = stableDiffusion.generate_image(initialPrompt)
diffusionImage.save(PipelineConfig.IMAGE_SAVE_PATH + "\\val\\buffer_image.png", pnginfo = imageInfo)
diffusionImage.save(PipelineConfig.IMAGE_SAVE_PATH + "\\initial_image.png", pnginfo = imageInfo)

# Run the instance segmentation and inpainting for as many times as specified by RECURSION_DEPTH
for depth in range(PipelineConfig.RECURSION_DEPTH):
    instanceSeg = mrcnn.instance_segment(PipelineConfig.IMAGE_SAVE_PATH) # Perform instance segmentation using the MRCNN model  
    
    # subPrompts = llm_call.call_gpt4(instanceSeg['detectedClasses'], INITIAL_USER_PROMPT)

    dir = PipelineConfig.IMAGE_SAVE_PATH + "\\" + str(depth + 1)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    for idx, segMask in zip(np.arange(len(instanceSeg['detectedClasses'])), instanceSeg['segmentedMasks']):
        mask_str = mask2str(segMask)
        img, imageInfo, imageStr = stableDiffusion.inpaint_image("", mask_str, imageStr)
        img.save(dir + "\\" + str(idx) + "-" + str(instanceSeg['detectedClasses'][idx]) + ".png", pnginfo=imageInfo)

    img.save(PipelineConfig.IMAGE_SAVE_PATH + "\\val\\buffer_image.png", pnginfo = imageInfo)