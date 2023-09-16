# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:53

@author: mick.yi

入口类

"""
import argparse
import os
import re

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models,transforms

from interpretability.grad_cam import *
# from interpretability.guided_back_propagation import GuidedBackPropagation

def gradcam_1(args, model, inputs, class_id, device, layer_name):

    inputs = inputs.to(device)
    class_id = class_id.to(device)
    # 网络
    net = model
    # Grad-CAM
    grad_cam = GradCAM_1(net, layer_name)
    mask0 = grad_cam(inputs, class_id, device)  # cam mask
    grad_cam.remove_handlers()
    return mask0

