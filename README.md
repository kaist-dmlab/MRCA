# Sample-Efficient Multi-Round Generative Data Augmentation for Long-Tail Instance Segmentation

![MRCA](assets/overview.png)

## Description

Data synthesis has become increasingly crucial for long-tail instance segmentation tasks to mitigate class imbalance and high annotation costs. We propose a **collaborative** approach that incorporates feedback from an instance segmentation model to guide the augmentation process. Specifically, the diffusion model uses feedback to generate objects that exhibit high uncertainty. The number and size of synthesized objects for each class are dynamically adjusted based on the model state to improve learning in underrepresented classes. This augmentation process is further strengthened by running **multiple rounds**, allowing feedback to be refined throughout training. In summary, **multi-round collaborative augmentation (MRCA)** enhances sample efficiency by providing optimal synthetic data at the right moment. 

## Requirements
Follow [X-Paste](https://github.com/yoctta/XPaste) for basic requirements.

```
pip install -r requirements.txt
```
Download [LVIS](https://www.lvisdataset.org/dataset), [OpenImages](https://storage.googleapis.com/openimages/web/visualizer/index.html), and [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) datasets.


Set your access_token from [StableDiffusion3](https://huggingface.co/stabilityai/stable-diffusion-3-medium) to use the model.


Modify the pipeline_stable_diffusion_3.py file in the diffusers library to this [file](https://github.com/kaist-dmlab/MRCA/blob/main/generator/guided_diffusion/pipeline_stable_diffusion_3.py)


In the case of generating with stable diffusion 1.5, modify the pipeline_stable_diffusion.py file in the diffusers library to this [file](https://github.com/kaist-dmlab/MRCA/blob/main/generator/guided_diffusion/pipeline_stable_diffusion.py) 



## Getting Started 


1. Generate images with stablediffusion3:  

```
cd generator

python generate.py
```


2. Segment foreground objects and filter low-quality objects:
```
cd diSegmenter

python segmentAndFilter.py
```


3. Train instance segmentation model:
```
bash launch.sh --config configs/MRCA/MRCA_R50.yaml 
```

4. Test with given checkpoint:
```
bash launch.sh --config configs/MRCA/MRCA_R50.yaml --eval-only

```

## Models



## References & Acknowledgements
We use code from
[Detectron2](https://github.com/facebookresearch/detectron2),
[StableDiffusion3](https://huggingface.co/stabilityai/stable-diffusion-3-medium),
[CenterNet2](https://github.com/xingyizhou/CenterNet2),
[X-Paste](https://github.com/yoctta/XPaste), and
[BSGAL](https://github.com/aim-uofa/DiverGen/tree/main/BSGAL)


