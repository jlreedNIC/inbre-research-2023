# ------------
# @author   Jordan Reed
# @date     6/1/23
# @brief    This file is testing adaptive thresholding
# ------------


try:
    from nd2reader import ND2Reader
    import matplotlib.pyplot as plt
    import numpy as np
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

def show_image_using_scalar(img, color_channel=ccm.green_channel):
    """Uses matplotlib to show the given image (using the given scalar values) using the given color channel

    Args:
        img (array of ints): array of ints representing grayscale pixels
        color_channel (color map, optional): Color map object. Custom color map preferred. Defaults to ccm.green_channel.
    """
    maxval = np.max(img)
    plt.imshow(img,vmax=maxval, cmap=color_channel)
    plt.show()

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

with ND2Reader(folder_loc + file_names[2]) as imgs:
    # grab imgs to work with
    pcna_imgs = get_imgs_from_channel(imgs, "far red")
    neuron_imgs = get_imgs_from_channel(imgs, "DAPI")

# start with single img to work with
img = compress_stack(pcna_imgs)
blur_img = cv2.GaussianBlur(img, (7,7), 0) # think about enhancing contrast
adj = cv2.convertScaleAbs(blur_img, alpha=.15, beta=100)

# show_image_using_scalar(img, ccm.purple_channel)
# show_image_using_scalar(blur_img, ccm.purple_channel)
# show_image_using_scalar(adj, ccm.purple_channel)
cv2.imshow('adjusted original', adj)

# otsu's thresholding
T, otsus_method = cv2.threshold(adj, 0,np.max(adj), cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
otsu_mask = otsus_method == 255
adj[otsu_mask] = 0
# cv2.imshow('otsus method mask', adj)
# show_image_using_scalar(otsus_method)
# separates out background better

import copy
adapt_adj = copy.copy(adj)
adapt_thresh = cv2.adaptiveThreshold(adj, np.max(adj), cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=9, C=4)
mask = adapt_thresh == 255
adapt_adj[mask] = 0
# cv2.imshow('with adaptive mean mask', adapt_adj)
# show_image_using_scalar(adapt_adj, ccm.purple_channel)


# adaptive with gaussian mean
gauss_thresh = cv2.adaptiveThreshold(adj, np.max(adj), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=15, C=5)
gmask = gauss_thresh == 255
adj[gmask] = 0
# cv2.imshow('adaptive with gaussian', adj)
# cv2.waitKey(0)
# show_image_using_scalar(adj, ccm.purple_channel)

# --- apply morphological techniques

# testing erosion
# for i in range(0,3):
#     eroded = cv2.erode(adapt_adj.copy(), None, iterations=i+1)
#     cv2.imshow(f'eroded {i+1} times', eroded)
#     cv2.waitKey(0)

# ---- testing opening
# kernelSizes = [(3, 3), (5, 5), (7, 7)]
# # loop over the kernels sizes
# for kernelSize in kernelSizes:
# 	# construct a rectangular kernel from the current size and then
# 	# apply an "opening" operation
# 	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
# 	opening = cv2.morphologyEx(adapt_adj, cv2.MORPH_OPEN, kernel)
# 	cv2.imshow("Opening: ({}, {})".format(
# 		kernelSize[0], kernelSize[1]), opening)
# 	cv2.waitKey(0)


# testing gradient
# loop over the kernels a final time
# for kernelSize in kernelSizes:
	# construct a rectangular kernel and apply a "morphological
	# gradient" operation to the image
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    # gradient = cv2.morphologyEx(adapt_adj, cv2.MORPH_GRADIENT, kernel)
    # cv2.imshow("Gradient: ({}, {})".format(
    #     kernelSize[0], kernelSize[1]), gradient)
    # cv2.waitKey(0)

# ---- testing watershed from skimage

# from skimage.segmentation import watershed
# new_img = watershed(adapt_adj)
# print(type(new_img), new_img)
# show_image_using_scalar(new_img)

# cv2.imshow('watershed', new_img)
# cv2.waitKey(0)