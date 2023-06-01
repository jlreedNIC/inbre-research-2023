# ------------
# @author   Jordan Reed
# @date     5/23/23
# @brief    This file is to test an image mask with basic 'custom' thresholding
# ------------


try:
    from nd2reader import ND2Reader
    import matplotlib.pyplot as plt
    import numpy as np
    # import datetime as dt
    import cv2
except Exception as e:
    print('issue with import')
    print('You may need to install importlib package if issue with nd2reader.')
    print(e)

import custom_color_map as ccm

folder_loc = '../nd2_files/'
file_names = [
    'SciH-Whole-Ret-4C4-Redd-GFP-DAPI005.nd2', # 4gb
    'Undamaged-structual-example.nd2', # 58mb 
    'S2-6dpi-uoi2506Tg-4R-#13-sxn2003.nd2', #40mb
    'S2-6dpi-uoi2506Tg-1R-#13-sxn2002.nd2' # 70mb
]

def get_imgs_from_channel(nd2img:ND2Reader, channel_name:str):
    """Extracts all images from the specified channel

    Args:
        nd2img (ND2Reader Object): open nd2 file from which data is extracted
        channel_name (string): name of channel in nd2 file

    Returns:
        numpy array: array of images (array of arrays)
    """
    
    nd2img.iter_axes = 'cz'
    channel_num = nd2img.metadata['channels'].index(channel_name)+1
    print(f'{channel_name} at channel {channel_num}')
    z_img_num = nd2img.sizes['z']

    new_arr = np.zeros((z_img_num, nd2img.metadata['height'], nd2img.metadata['width']))
    index = 0
    for i in range( (channel_num-1)*z_img_num, channel_num*z_img_num ):
        new_arr[index] = nd2img[i]
        index += 1
    
    return new_arr

def compress_stack(img_stack):

    # looping like this slows the program down
    # keeping it to try min, avg, sum, etc
    # new_arr = np.zeros((img_stack[0].shape))

    # for i in range(new_arr.shape[0]):
    #     for j in range(new_arr.shape[1]):
    #         new_arr[i][j] = np.max(img_stack[:,i,j])

    # new array created by taking max value of all imgs in stack
    new_arr = np.amax(img_stack, 0)

    return new_arr

# open nd2 file
with ND2Reader(folder_loc + file_names[2]) as sample_image:
    
    # iterate over all channels and entire z stack
    num_channels = sample_image.sizes['c']
    img_per_channel = sample_image.sizes['z']
    sample_image.iter_axes = 'cz'
    print(f'total number of images to show: {sample_image.shape[0]}')
    print(f'total channels: {num_channels}')
    print(f'images per channel: {img_per_channel}')
    print(f'size of images: {sample_image.shape[1]} x {sample_image.shape[2]}')

    pcna_imgs = get_imgs_from_channel(sample_image, 'far red')
    neuron_imgs = get_imgs_from_channel(sample_image, 'DAPI')

img = compress_stack(pcna_imgs)


blur_img = cv2.GaussianBlur(img,(7,7),0)
adj = cv2.convertScaleAbs(blur_img, alpha=.15, beta=100)

maxval = np.max(adj)
mask = .425 * maxval
background_mask = maxval * .4
background_layer_mask = maxval * .65 #(maxval + background_mask)/2

np_bckgrnd_img_arr = np.asarray(adj) <= background_mask
adj[np_bckgrnd_img_arr] = 0
np_prev_layer_arr = (np.asarray(adj) > 0) & (np.asarray(adj) <= background_layer_mask)
adj[np_prev_layer_arr] = maxval * .25
# print(adj)

# original image
# plt.subplot(1,2,1)
# plt.imshow(img, vmax=np.max(img), cmap=ccm.purple_channel)
# plt.show()

# gaussian blur image
# plt.subplot(1,2,2)
# print(np.max(adj), mask)
plt.imshow(adj,vmax=maxval, cmap=ccm.purple_channel)
plt.show()
    
    

'''
above code does ok. think it might do better with gaussian blur applied
'''

    


    
