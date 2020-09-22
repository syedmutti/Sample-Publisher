#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
import numpy as np
import os

import models as model
import load_models





#Creating the main (Combined) model with all the pre-trained loaded weights
model = load_models.load_models()


#  Function to Display Outputs with each long bone 

def display(output):

	x, pred_LL, pred_UL, pred_LA, pred_UA = output

	print('Stature: ', x.item(), 'LL: ', pred_LL.item(),'UL: ', pred_UL.item(),'LA: ', pred_LA.item(),'UA: ', pred_UA.item())
    

def do_inference_ll():

	## Inference Using Knee_Height
	# Test value 
	measurement = 511.0
	flag = 'lower_leg' #176.88

	# Converting value to a pytorch tensor
	x1 = torch.from_numpy(np.array([measurement], dtype='float32'))

	# Visualising in cm
	#print('Input lower_leg :', x1.item()/10)

	# Reshaping to networks input size
	x1.view((1,1))

	# Passing through the main model with flag to generate height and also other segments
	output = model(x1/592.0, flag)

	#display(output)

	return flag, output, measurement



def do_inference_ul():

	## Inference Using upper_leg
	# Test value 
	measurement = 44.0
	flag = 'upper_leg'  # 174.8

	# Converting value to a pytorch tensor
	x1 = torch.from_numpy(np.array([measurement], dtype='float32'))

	# Printing input value in cm
	#print('\nInput upper_leg :', x1.item())


	# Reshaping to networks input size
	x1.view((1,1))        

	# Passing through the main model with flag to generate height and also other segments
	output = model(x1/55.5, flag)  

	#display(output)
	return flag, output, measurement*10

def do_inference_la():

	## Inference Using Lower Arm
	# Test value 
	measurement = 266.52
	flag = 'lower_arm'  # 192.076

	# Converting value to a pytorch tensor
	x1 = torch.from_numpy(np.array([measurement], dtype='float32'))

	# Printing input value in cm
	#print('\nInput lower_arm :', x1.item())

	# Reshaping to networks input size
	x1.view((1,1))        

	# Passing through the main model with flag to generate height and also other segments
	output = model(x1/280.15, flag)

	#display(output)
	return flag, output, measurement

def do_inference_ua():

	## Inference Using Sitting Height Under 18
	# Test value 
	measurement = 358.96
	flag = 'upper_arm'  # 182.67

	# Converting value to a pytorch tensor
	x1 = torch.from_numpy(np.array([measurement], dtype='float32'))

	# Printing input value in cm
	#print('\nInput upper_arm :', x1.item())

	# Reshaping to networks input size
	x1.view((1,1))        

	# Passing through the main model with flag to generate height and also other segments
	output = model(x1/389.098, flag)
	#display(output)
	return flag, output, measurement





