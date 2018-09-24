import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F



def dense(in_dim,out_dim,act_fn=nn.Sigmoid()):


    layers = nn.Sequential(

            nn.Linear(in_dim,out_dim),
            act_fn
            )
            
    return layers