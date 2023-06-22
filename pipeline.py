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

# Define a configuration class for the overall pipeline
class PipelineConfig:
    IMAGE_SAVE_PATH = ".\\images"  # Directory where all images, both interim and final will be saved
    RECURSION_DEPTH = 1

# Define a configuration class for the Stable Diffusion model
class SDConfig:
    OPTION_PAYLOAD = {
        'sd_model_checkpoint': "pastelmix-better-vae-fp32.ckpt [943a810f75]"  # Specify the model checkpoint for the SD model
    }

    GENERATION_PAYLOAD = {
        'steps': 20,  # Number of steps for the SD model to take during generation
        'enable_hr': True,  # Whether to enable high-resolution generation
        'hr_second_pass_steps': 20,
        'denoising_strength': 0.6  # Strength of the denoising to apply during generation
    }

    INPAINT_PAYLOAD = {
    }

# Define a configuration class for the MRCNN model
class MRCNNConfig:
    MODEL_DIR = ".\\MRCNN\\model_weights"  # Directory where the MRCNN model weights are stored
    MODEL_WEIGHTS_PATH = ".\\MRCNN\\model_weights\\pastel_mix_weights.h5"

    IMAGES_PER_GPU = 1

# Create instances of the SD and MRCNN models with the appropriate configurations
stableDiffusion = diffusion_model.SDModel(option_payload=SDConfig.OPTION_PAYLOAD,
                                          generation_payload=SDConfig.GENERATION_PAYLOAD,
                                          inpaint_payload=SDConfig.INPAINT_PAYLOAD)
stableDiffusion.initialize_model()  # Initialize the SD model

mrcnn = instance_segmentation.MRCNNModel(model_dir=MRCNNConfig.MODEL_DIR,
                                         model_weights_path=MRCNNConfig.MODEL_WEIGHTS_PATH,
                                         images_per_gpu=MRCNNConfig.IMAGES_PER_GPU)
mrcnn.initialize_model()  # Initialize the MRCNN model

# Generate an image using the SD model and save the image information
initialPrompt = "1girl, full body, city background"
diffusionImage, imageInfo = stableDiffusion.generate_image(initialPrompt)
diffusionImage.save(PipelineConfig.IMAGE_SAVE_PATH + "\\val\\initial_diffusion_generation.png", pnginfo = imageInfo)

# Run the instance segmentation and potential inpainting for as many times as specified by RECURSION_DEPTH
for depth in range(PipelineConfig.RECURSION_DEPTH):
    instanceSeg = mrcnn.instance_segment(PipelineConfig.IMAGE_SAVE_PATH) # Perform instance segmentation using the MRCNN model

    # subPrompts = llm_call.call_gpt4(instanceSeg['detectedClasses'], INITIAL_USER_PROMPT)

    # for segmentedMask, subPrompt in zip(instanceSeg['segmentedMasks'], subPrompts):
    #     diffusionImage = diffusion_model.inpaint_image(subPrompt, segmentedMask, diffusionImage)

    # diffusionImage.save('images\\val\\initial_diffusion_generation.png', pnginfo = imageInfo)