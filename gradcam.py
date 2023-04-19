import copy
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from define_simclr import eval_model
import numpy as np
import torchvision
from PIL import Image
# Load model resnet18
model2=eval_model.simclr.encoder
# Pick up layers for visualization
target_layers = [model2.layer4[-1]]

import os
# directory="/home/ccet/SIH/UCMD/100/test/aeroplane/"
# for filename in os.listdir(directory):
#path = os.path.join(directory, filename)
path = "/home/ccet/SIH/SIRI-WHU/100/train/overpass/1386.tif"

rgb_img = Image.open(path).convert('RGB')
# Max min normalization
rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
# Create an input tensor image for your model
input_tensor = torchvision.transforms.functional.to_tensor(rgb_img).unsqueeze(0).float()
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model2, target_layers=target_layers, use_cuda=True)
# cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)
# cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=False)
grayscale_cam = cam(input_tensor=input_tensor)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
Image.fromarray(visualization, 'RGB').save('/home/ccet/SIH/notebooks/simclr-12cls/'+'mlrs'+'.png')