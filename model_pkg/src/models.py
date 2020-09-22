

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
import numpy as np
import os


class MyModelA(nn.Module):
    def __init__(self):
        super(MyModelA, self).__init__()
        self.fc1 = nn.Linear(1, 4)
        self.fc2 = nn.Linear(4, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
       

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC, modelD, modelH2LL, modelH2UL, modelH2LA, modelH2UA):
        super(MyEnsemble, self).__init__()
        
        self.model_lower_leg = modelA
        self.model_upper_leg = modelB
        self.model_lower_arm = modelC
        self.model_upper_arm = modelD
        
        # Models for Virtual Reconstruction
        
        self.height_to_lowerleg = modelH2LL
        self.height_to_upper_leg = modelH2UL
        self.height_to_lower_arm = modelH2LA
        self.height_to_upper_arm = modelH2UA
        

                
    def forward(self, x1, flag):
        
        def reconstruct_Segments(height):

            pred_LL  = self.height_to_lowerleg(height/201.513) * 49.319  # 201.513741064 493.195872265
            pred_UL = self.height_to_upper_leg(height/201.913) * 58.554   # 201.913788186 585.546772618
            pred_LA = self.height_to_lower_arm(height/204.805) * 31.670  # 204.805580317 316.702168533
            pred_UA = self.height_to_upper_arm(height/204.104) * 42.5100   # 204.104986515 425.100857276

            return pred_LL, pred_UL, pred_LA, pred_UA
        
        if flag == 'lower_leg':
            x = self.model_lower_leg(x1)
            x*=195.5    
            
        if flag == 'upper_leg':
            x = self.model_upper_leg(x1)
            x*=204.1
            
        if flag == 'lower_arm':
            x = self.model_lower_arm(x1)
            x*=204.68
            
        if flag == 'upper_arm':
            x = self.model_upper_arm(x1)
            x*= 209.2665
            
        #print('Height predicted from {} : is {} cm'.format(flag, x.item()))
        
        pred_LL, pred_UL, pred_LA, pred_UA = reconstruct_Segments(x)
        
        return x, pred_LL, pred_UL, pred_LA, pred_UA



