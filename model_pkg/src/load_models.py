

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import models as model


def load_models():

	# Create models and load state_dicts    

	model_lower_leg = model.MyModelA()
	model_upper_leg = model.MyModelA()
	model_lower_arm = model.MyModelA()
	model_upper_arm = model.MyModelA()

	#Virtual Segments Reconstruction Model

	model_height_to_lowerleg = model.MyModelA()
	model_height_to_upper_leg = model.MyModelA()
	model_height_to_lower_arm = model.MyModelA()
	model_height_to_upper_arm = model.MyModelA()

	# Load state dicts

	# Load state dicts
	PATH_lower_leg = '/mrtstorage/users/rehman/catkin_ws/src/mytest/src/Trained_models/lower_leg_pytorch_/ansur_model_2.6701'
	PATH_upper_leg = '/mrtstorage/users/rehman/catkin_ws/src/mytest/src/Trained_models/upper_leg_model_pytorch_/upper_leg_5.84564208984375'
	PATH_lower_arm = '/mrtstorage/users/rehman/catkin_ws/src/mytest/src/Trained_models/lower_arm_pytorch_/lower_arm_pytorch_3.489548921585083'
	PATH_upper_arm = '/mrtstorage/users/rehman/catkin_ws/src/mytest/src/Trained_models/upper_arm_pytorch_/upper_arm_pytorch4.744106292724609'

	# Virtual Models

	PATH_model_height_to_lowerleg =  '/mrtstorage/users/rehman/catkin_ws/src/mytest/src/Virtual_Segments/Trained_models/height_lower_leg_5.075006484985352'
	PATH_model_height_to_upper_leg = '/mrtstorage/users/rehman/catkin_ws/src/mytest/src/Virtual_Segments/Trained_models/height_upper_leg_18.44318962097168'
	PATH_model_height_to_lower_arm = '/mrtstorage/users/rehman/catkin_ws/src/mytest/src/Virtual_Segments/Trained_models/height_lower_arm_9.933013916015625'
	PATH_model_height_to_upper_arm = '/mrtstorage/users/rehman/catkin_ws/src/mytest/src/Virtual_Segments/Trained_models/Height_upper_arm_9.007071495056152'

	# Loading Learned Weights

	model_lower_leg.load_state_dict(torch.load(PATH_lower_leg))
	model_upper_leg.load_state_dict(torch.load(PATH_upper_leg))
	model_lower_arm.load_state_dict(torch.load(PATH_lower_arm))
	model_upper_arm.load_state_dict(torch.load(PATH_upper_arm))

	# Loading Learned Weights for Virtual Segments

	model_height_to_lowerleg.load_state_dict(torch.load(PATH_model_height_to_lowerleg))
	model_height_to_upper_leg.load_state_dict(torch.load(PATH_model_height_to_upper_leg))
	model_height_to_lower_arm.load_state_dict(torch.load(PATH_model_height_to_lower_arm))
	model_height_to_upper_arm.load_state_dict(torch.load(PATH_model_height_to_upper_arm))


	#Load Model
	full_model = model.MyEnsemble(model_lower_leg, model_upper_leg, model_lower_arm, model_upper_arm,
		            model_height_to_lowerleg, model_height_to_upper_leg,
		            model_height_to_lower_arm, model_height_to_upper_arm)
	return full_model 
