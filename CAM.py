import torch
from model import * 
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models, transforms
from torch.nn import functional as F
import cv2
import numpy as np
from scipy.misc import imsave


net = Model().cuda()
net = torch.load('./model/Resnet.pkl')

net.eval()

finalconv_name = 'layer4'

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

print(net.resnet._modules.get(finalconv_name))
net.resnet._modules.get(finalconv_name).register_forward_hook(hook_feature)
# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        #print(cam.shape)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        #print(cam_img.shape)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

casename = 'Case09_27'

np_arr = np.load('./data/slice_data/{}.npy'.format(casename))
print(np_arr.shape)
mask_arr = np.load('./data/slice_data/{}_segmentation.npy'.format(casename))
mask_arr = np.uint8(255 * mask_arr)


#np_arr = np.resize(np_arr,[224,224])
tensor = torch.from_numpy(np_arr)
tensor = torch.stack([tensor,tensor,tensor],0)
#tensor = preprocess(tensor)
img_variable = Variable(tensor.unsqueeze(0)).float().cuda()
logit = net(img_variable)

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.data.cpu().numpy()
idx = idx.data.cpu().numpy()
CAMs = returnCAM(features_blobs[0], [weight_softmax], [idx])

print(CAMs[0].shape)

img = np_arr
img = np.stack([img,img,img],axis=2)
print(img.shape)

imsave('test_img.png',np_arr)


height, width,_ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.2
cv2.imwrite('CAM.jpg', result)
cv2.imwrite('mask.jpg',mask_arr)



