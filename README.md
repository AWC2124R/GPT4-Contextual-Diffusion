# GPT4-Contextual-Diffusion

This project is an end-to-end pipeline that integrates Language Learning Models (LLMs), Stable Diffusion Models, and an instance segmentation model (Mask R-CNN) to provide a rich context for generated images.

## Overview

The goal is to improve the interpretability and contextual relevance of images produced by Stable Diffusion models by incorporating instance-specific context from LLMs. This project employs an instance segmentation model, Mask R-CNN, to recognize and segment distinct objects within the generated images. This data is then used to supply context for each object through LLMs. Subsequently, the Stable Diffusion Model can utilize this context to inpaint the current image with additional detail and coherence.

The pipeline is broken down into the following stages:

1. **Stable Diffusion Model Generation**: An initial image prompt is passed to a Stable Diffusion model, which generates a contextually relevant image.
2. **Instance Segmentation with Mask R-CNN**: The generated image is passed to a Mask R-CNN model, which performs instance segmentation to identify and isolate different objects within the image.
3. **LLM Context Generation**: For each segmented object, an LLM(GPT-4 in this case) provides context that can be used in subsequent runs of the pipeline to generate more precise and contextually relevant images.

A general overview of the pipeline is shown in the image below. To view a more detailed pipeline structure, refer to the assets folder.

![Pipeline Overview](https://github.com/AWC2124R/GPT4-Contextual-Diffusion/blob/master/Assets/pipeline_overview_pseudo.png)

## Examples
The following is an example illustrating the enhancements achieved through the use of a GPT-4 integrated context pipeline for Stable Diffusion, with 1 recursion step.

![Example](https://github.com/AWC2124R/GPT4-Contextual-Diffusion/blob/master/Assets/result1.png)

## Notes
The repository includes the MRCNN code to accommodate a specifically trained MRCNN model, while the Stable Diffusion Model operates locally on a separate instance for access to the SD API. For the pipeline to function, users must run their own instance of the Stable Diffusion Model.

This project employs a custom-trained MRCNN model to supply object masks. Users can train their own MRCNN model to detect varying objects from assorted stylized Stable Diffusion Models. Specifically, an effective application of custom-trained MRCNN models would be to maintain separate models for different recursion depths. For instance, the initial recursion loop could identify relatively large elements of the image, like humans, skies, and buildings, thus enabling the LLM to generate more generalized details for the image such as human poses and building structures. Following recursions could then detect smaller individual objects like clothes, jackets, and cars, granting the LLM a more comprehensive and detailed influence over the image composition.

## Shortcomings
Given the current configuration of the MRCNN model, the pipeline cannot accommodate significant alterations, such as modifying character position or background within previously generated images. However, it's worth highlighting that [recent research from UC Berkeley & UCSF](https://arxiv.org/pdf/2305.13655.pdf) demonstrates that even the initial generation of an image can be enhanced by LLM-generated context, vastly improving the capacity to effect macro-level changes on diffusion model outputs. This potential for "LLM-grounded Diffusion," as evidenced by implementations from various sources, suggests that this limitation can indeed be overcome. Other issues encountered during development, such as the effectiveness of image inpainting on smaller masks, seem capable of being resolved through tuning or more precise MRCNN training.

## Development Environment
- Python 3.7.16
- TensorFlow 1.13.0
- Keras 2.0.8
- PIL
- Matplotlib
- Numpy
- Scikit-Image

The Python and Tensorflow versions have been intentionally downgraded to align with the TF-1 based MRCNN code. It is recommended to establish a distinct environment for this project (or the MRCNN model), given that migrating the original code to TF2 can be challenging.

## License
This project is licensed under the terms of the MIT license.

## Acknowledgements & Credits

In particular, this project expresses its gratitude to the developers of the Stable Diffusion, Mask R-CNN, and GPT models, whose tools form the backbone of this project.

This project includes code from the following sources:
- [MRCNN](https://github.com/matterport/Mask_RCNN) - Obtained Code / Initial Training Weights for Mask RCNN