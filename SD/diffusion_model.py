import json
import requests
import io
import base64
import PIL
from PIL import Image

# Define the URL for the SD API
SD_URL = 'http://127.0.0.1:7860'

# Set up default payloads for options, image generation and inpainting
OPTION_PAYLOAD_DEFAULT = {
    'sd_model_checkpoint': "pastelmix-better-vae-fp32.ckpt [943a810f75]"  # The checkpoint for the SD model to use
}

GENERATION_PAYLOAD_DEFAULT = {
    'styles': ["Anime-Image Baseline"],  # The style to use for image generation
    'sampler_index': "DPM++ 2M Karras",  # The sampler to use for image generation
    'steps': 20,  # Number of steps for the generation process
    'width': 640,
    'height': 360,
    'enable_hr': True,  # Whether to generate the image in high resolution
    'hr_upscaler': "Latent",  # The upscaler to use for high resolution
    'hr_scale': 2,  # The scale factor for high resolution
    'hr_second_pass_steps': 20,
    'denoising_strength': 0.6  # Strength of denoising to apply to the image
}

INPAINT_PAYLOAD_DEFAULT = {
    'styles': ["Anime-Image Baseline"],
    'resize_mode': 0,
    'mask_blur': 10,
    "inpainting_mask_invert": 0,
    "inpainting_fill": 1,
    "inpaint_full_res": True,
    "inpaint_full_res_padding": 32,
    'steps': 40,
    'sampler_index': "DPM++ 2M Karras",
    'denoising_strength': 0.6
}

# Define the SDModel class
class SDModel:
    # Initialize the class with the provided payloads and API URL
    def __init__(self, option_payload=OPTION_PAYLOAD_DEFAULT, generation_payload=GENERATION_PAYLOAD_DEFAULT,
                inpaint_payload=INPAINT_PAYLOAD_DEFAULT, api_url=SD_URL):
        
        # Combine the default and provided payloads
        self.option_payload = dict(OPTION_PAYLOAD_DEFAULT, **option_payload)

        self.generation_payload = dict(GENERATION_PAYLOAD_DEFAULT, **generation_payload)
        self.inpaint_payload = dict(INPAINT_PAYLOAD_DEFAULT, **inpaint_payload)

        self.api_url = api_url # Set the API URL
    
    # Initialize the SD model by sending a POST request to the
    # API's options endpoint with the options payload
    def initialize_model(self):
        response = requests.post(url=self.api_url + '/sdapi/v1/options', json=self.option_payload)
        return response

    # Generate an image by sending a POST request to the API's txt2img endpoint
    # with the generation payload and the user prompt
    def generate_image(self, userPrompt):
        diffusionPayload = self.generation_payload
        diffusionPayload['prompt'] = userPrompt  # Add the user prompt to the payload

        txt2imgResponse = requests.post(url=self.api_url + '/sdapi/v1/txt2img', json=diffusionPayload)  # Send the POST request
        rJson = txt2imgResponse.json()

        # Decode the base64-encoded image and open it as a PIL Image object
        image = Image.open(io.BytesIO(base64.b64decode(rJson['images'][0].split(",", 1)[0])))

        # Define a payload for the PNG info request, including the base64-encoded image
        PNGPayload = {
            "image": "data:image/png;base64," + rJson['images'][0]
        }

        # Send a POST request to the API's png-info endpoint with the PNG payload, and parse the response
        PNGInfoResponse = requests.post(url=self.api_url + '/sdapi/v1/png-info', json=PNGPayload)
        PNGInfo = PIL.PngImagePlugin.PngInfo()
        PNGInfo.add_text("parameters", PNGInfoResponse.json().get("info"))  # Add the info from the response to the PNGInfo object

        # Return the image and PNGInfo object
        return (image, PNGInfo, rJson['images'][0])

    def inpaint_image(self, subPrompt, segmentedMask, diffusionImage):
        diffusionPayload = self.inpaint_payload
        diffusionPayload['prompt'] = subPrompt
        diffusionPayload['init_images'] = [ "data:image/png;base64," + diffusionImage ]
        diffusionPayload['mask'] = "data:image/png;base64," + segmentedMask

        img2imgResponse = requests.post(url=self.api_url + '/sdapi/v1/img2img', json=diffusionPayload)  # Send the POST request
        rJson = img2imgResponse.json()
        
        image = Image.open(io.BytesIO(base64.b64decode(rJson['images'][0].split(",", 1)[0])))

        # Add PNG INFO POST Request

        return image