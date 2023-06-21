import json
import requests
import io
import base64
import PIL

SD_URL = 'http://127.0.0.1:7860'

DEFAULT_OPTION_PAYLOAD = {
    'sd_model_checkpoint': "pastelmix-better-vae-fp32.ckpt [943a810f75]"
}

DEFAULT_DIFFUSION_PAYLOAD_GENERATION = {
    'prompt': "1girl, full body",
    'styles': ["Anime-Image Baseline"],

    'sampler_index': "DPM++ 2M Karras",
    'steps': 20,
    'width': 640,
    'height': 360,

    'enable_hr': True,
    'hr_upscaler': "Latent",
    'hr_scale': 2,
    'hr_second_pass_steps': 20,
    'denoising_strength': 0.6
}

DEFAULT_DIFFUSION_PAYLOAD_INPAINT = {
    'resize_mode': 'just_resize',
    'mask_mode': 'inpaint_masked',
    'masked_content': 'original',
    'inpaint_area': 'whole picture',
    'padding_pixels': -1,
    'sampling_method': 'DPM++2MKarras',
    'sampling_steps': 20,
    'restore_faces': False,
    'tiling': False,
    'resize_by': 1,
    'batch_count': 1,
    'batch_size': 1,
    'CFG_scale': 7,
    'denoising_strength': 0.6,
    'seed': -1,
    'styles': 'Anime-Image Baseline'
}

def initialize_model_options():
    response = requests.post(url=SD_URL + '/sdapi/v1/options', json=DEFAULT_OPTION_PAYLOAD)
    return response

def generate_new_image(userPrompt):
    diffusionPayload = DEFAULT_DIFFUSION_PAYLOAD_GENERATION
    diffusionPayload['prompt'] = userPrompt

    response = requests.post(url=SD_URL + '/sdapi/v1/txt2img', json=diffusionPayload)
    r = response.json()

    image = PIL.Image.open(io.BytesIO(base64.b64decode(r['images'][0].split(",",1)[0])))

    PNGPayload = {
        "image": "data:image/png;base64," + r['images'][0]
    }

    responsePNGInfo = requests.post(url=SD_URL + '/sdapi/v1/png-info', json=PNGPayload)
    PNGInfo = PIL.PngImagePlugin.PngInfo()
    PNGInfo.add_text("parameters", responsePNGInfo.json().get("info"))

    return (image, PNGInfo)

def inpaint_image(subPrompt, segmentedMask, diffusionImage):
    pass