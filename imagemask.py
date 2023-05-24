# ------------
# @author   Jordan Reed
# @date     5/23/23
# @brief    This file is to test an image mask
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

file_names = [
    'nd2_files/SciH-Whole-Ret-4C4-Redd-GFP-DAPI005.nd2', # 4gb
    'nd2_files/Undamaged-structual-example.nd2', # 58mb 
    'nd2_files/S2-6dpi-uoi2506Tg-4R-#13-sxn2003.nd2', #40mb
    'nd2_files/S2-6dpi-uoi2506Tg-1R-#13-sxn2002.nd2' # 70mb
]


# open nd2 file
with ND2Reader(file_names[3]) as sample_image:
    
    # iterate over all channels and entire z stack
    num_channels = sample_image.sizes['c']
    img_per_channel = sample_image.sizes['z']
    sample_image.iter_axes = 'cz'
    print(f'total number of images to show: {sample_image.shape[0]}')
    print(f'total channels: {num_channels}')
    print(f'images per channel: {img_per_channel}')
    print(f'size of images: {sample_image.shape[1]} x {sample_image.shape[2]}')

    # orig = sample_image[11]
    ch = 2
    for i in range((ch-1)*img_per_channel, ch*img_per_channel): #sample_image.shape[0]): #sample_image.shape[0]):
        
        img = sample_image[i]
        blur_img = cv2.GaussianBlur(img,(7,7),0)
        '''
        try doing enhance contrast
        gaussian blur
        unsharp mask
        '''
    
        maxval = np.max(blur_img)
        mask = .425 * maxval
        background_mask = maxval * .4
        background_layer_mask = maxval * .65 #(maxval + background_mask)/2
    
        np_bckgrnd_img_arr = np.asarray(blur_img) <= background_mask
        blur_img[np_bckgrnd_img_arr] = 0
        np_prev_layer_arr = (np.asarray(blur_img) > 0) & (np.asarray(blur_img) <= background_layer_mask)
        blur_img[np_prev_layer_arr] = maxval * .25
        print(blur_img)
    
        # original image
        plt.subplot(1,2,1)
        plt.imshow(sample_image[i], vmax=np.max(sample_image[i]), cmap=ccm.purple_channel)
        # plt.show()
        
        # gaussian blur image
        plt.subplot(1,2,2)
        print(np.max(blur_img), mask)
        plt.imshow(blur_img,vmax=maxval, cmap=ccm.purple_channel)
        plt.show()
    
    

'''
above code does ok. think it might do better with gaussian blur applied
'''

    


    
