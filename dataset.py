import numpy as np
import torch
import torch.utils.data as data
from glob import glob
import os
import os.path
import pandas as pd
import pandas as pd
import numpy as np
from scipy.ndimage import zoom
import torchvision.utils as v_utils
from utils import *

class Prostate_Dataset(data.Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.dataframe = create_df(img_dir)
        #print(self.dataframe)
        self.dataframe.to_csv('df.csv',sep='\t')
    

    def __getitem__(self, index):

        label = self.dataframe.iloc[index,1] 
        img = np.load(os.path.join(self.img_dir,self.dataframe.iloc[index,0]))
        #print(img.shape)      
        img = torch.from_numpy(img)
        img = img.float()
        img = torch.stack([img,img,img],0)
        sample = {'image':img, 'label':float(label)}
        
        return sample 

    def __len__(self):
            return len(self.dataframe)
