import torch 
from torch import nn
import torchvision
from blocks import *

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()

        resnet = torchvision.models.resnet152(pretrained=True)
        in_num_fltrs  = resnet.fc.in_features
        print(in_num_fltrs)

        #Freezing weights
        # for param in resnet.parameters():
        #     param.requires_grad = False
            
        #Change last layer
        resnet.fc = dense(in_num_fltrs,1)
        #resnet.fc = nn.Linear(in_num_fltrs,1)
        
        
        

        outputs = []
        def hook(module, input, output):
            outputs.append(output)

        for name, child in resnet.named_children():
            for name2, params in child.named_parameters():
                print(name, name2)
        self.resnet = resnet
        #model.layer2.conv1.register_forward_hook (hook)

    

    def forward(self,inputs):
        """Define forward pass
        
        Args:
            inputs ([type]): [description]
        
        Returns:
            [type]: [description]
        """

        output = self.resnet(inputs)
        

        return output


#         
# ResNet()