import numpy as np
import SimpleITK as sitk
import os
import pandas as pd
import cv2
from skimage.transform import resize
from scipy.misc import imsave

def load_raw(filename):
    """Load images with .mhd, .raw format
    
    Args:
        filename (string): full path to the .mhd file
    
    Returns:
        img_array: np array containing image data
        origin (list): origina coodinates
        spacing (list): [z,x,y] spacing
    """
    itk_image = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(itk_image)

    origin = np.array(list(reversed(itk_image.GetOrigin())))# x y z origin in world coordinates
    spacing = np.array(list(reversed(itk_image.GetSpacing())))# Spacing in world coordinates

    return img_array,origin,spacing

def create_df(img_dir):

    
    """Create pd dataframe for dataloader
    
    Args:
        img_dir (str): directory containing the images
    
    Returns:
        df: pd dataframe 
    """ 

    img_lis = os.listdir(img_dir)
    case_lis = []
    label_lis = []

    #print(img_lis)
    
    for filename in img_lis:
        if filename.endswith('.npy') and 'segmentation' not in filename:

            #print(filename)        
            full_path = os.path.join(img_dir,filename)

            mask_name = filename.replace('.npy','_segmentation.npy')
            mask_path = os.path.join(img_dir,mask_name)

            mask = np.load(mask_path)
            #print(mask_path)

            #print(np.count_nonzero(mask))
            if np.count_nonzero(mask) != 0:
                label_lis.append(1)
            else:
                label_lis.append(0)
                #print('Does not contaion')


            case_lis.append(filename)
    
    
    df = pd.DataFrame({
    'casename':case_lis,'label':label_lis
    })


    return df


def create_slice(img_dir):
    

    """Split 3D volume into 2D slices 
    """

    save_dir = os.path.join(img_dir,'slice_data')


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    img_lis = os.listdir(img_dir)

    for filename in img_lis:
        if filename.endswith('.mhd') and 'segmentation' not in filename:

            full_path = os.path.join(img_dir,filename)
            mask_name = filename.replace('.mhd','_segmentation.mhd')
            mask_path = os.path.join(img_dir,mask_name)

            volume,origin,spacing = load_raw(full_path)
            mask,mask_origin, mask_spacing = load_raw(mask_path)
            print(volume.shape) #[z, h, w]

            # For each volume, save each slice as numpy array

            for z_slice in range(0,volume.shape[0]):
                #print(np.count_nonzero(mask[z_slice,:,:]))
                slice_name = filename.replace('.mhd','_') + str(z_slice) + '.npy'
                slice_ = resize(volume[z_slice,:,:], (224, 224), anti_aliasing=True,preserve_range=True)



                np.save(os.path.join(save_dir,slice_name),slice_)
                #imsave('test_img.png',slice_)


                mask_slice_name = filename.replace('.mhd','_') + str(z_slice) + '_segmentation.npy'
                mask_slice = resize(mask[z_slice,:,:], (224, 224), anti_aliasing=True,preserve_range=True)
                np.save(os.path.join(save_dir,mask_slice_name),mask_slice)

                #imsave('test_seg.png',mask_slice)
    



