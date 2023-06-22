# GPT4-Contextual-Diffusion

This project is an end-to-end pipeline that integrates Language Learning Models (LLMs), Stable Diffusion Models, and an instance segmentation model (Mask R-CNN) to provide a rich context for generated images. **Note that it is still in development.**

## Overview

The aim is to enhance the interpretability and contextual relevance of the images generated using Stable Diffusion models by incorporating instance-specific context from LLMs. To achieve this, we utilize an instance segmentation model, Mask R-CNN, to identify and segment different objects within the generated images. This information is then used to provide context for each object via LLMs.

The pipeline is broken down into the following stages:

1. **Stable Diffusion Model Generation**: An initial image prompt is passed to a Stable Diffusion model, which generates a contextually relevant image.
2. **Instance Segmentation with Mask R-CNN**: The generated image is passed to a Mask R-CNN model, which performs instance segmentation to identify and isolate different objects within the image.
3. **LLM Context Generation**: For each segmented object, an LLM provides context that can be used in subsequent runs of the pipeline to generate more precise and contextually relevant images.

## Notes
While the MRCNN code is contained within the repository in order to compensate for a specifically trained MRCNN model, the Stable Diffusion Model is run locally on a different instance, in order to access the SD API. Using the local SD instance, users will have to run their own SD instance in order for the pipeline to function.

## Development Environment
- Python 3.5.6
- TensorFlow 1.3.0
- Keras 2.0.8
- PIL
- Matplotlib
- Numpy
- Scikit-Image

The Python & Tensorflow versions were intentionally lowered to match the TF-1 based MRCNN code. It is suggested to create a separate environment for this project (or the MRCNN model), as migrating code to TF2 can be finicky.

## License
This project is licensed under the terms of the MIT license.

## Acknowledgements & Credits

In particular, this project expresses its gratitude to the developers of the Stable Diffusion, Mask R-CNN, and GPT models, whose tools form the backbone of this project.

This project includes code from the following sources:
- [MRCNN](https://github.com/matterport/Mask_RCNN) - Obtained Code / Initial Training Weights for Mask RCNN