import torch 
from torch import nn
import torchvision

class ResNet(nn.Module):

    def __init__(self):
        super(ResNet,self).__init__()

        resnet = torchvision.models.resnet152(pretrained=True)
        in_num_fltrs  = resnet.fc.in_features
        print(in_num_fltrs)
        resnet.fc = nn.Linear(in_num_fltrs,2)


    def forward(self,inputs):
        """Define forward pass
        
        Args:
            inputs ([type]): [description]
        
        Returns:
            [type]: [description]
        """

        output = self.resnet(inputs)
        


        return output






        for name, child in resnet.named_children():
            for name2, params in child.named_parameters():
                print(name, name2)
ResNet()