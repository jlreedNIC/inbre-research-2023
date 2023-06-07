# ------------
# @author   Jordan Reed
# @date     6/2/23
# @brief    This file is testing edge detection
# ------------

from nd2reader import ND2Reader
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime as dt
import copy

from skimage import filters, feature, color, measure

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

def use_subplots(imgs:list, titles = [], ncols = 1, nrows=1):
    num = len(imgs)

    if num == 0:
        print("you must pass images in list form")
        return
    if num > (ncols*nrows):
        print("num images don't match rows and columns")
        return
    
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True, figsize=(10,10))
    ax = axes.ravel()

    for i in range(0, len(imgs)):
        ax[i].imshow(imgs[i], cmap='gray')

        if i >= len(titles):
            title = f'figure {i}'
        else:
            title = titles[i]

        ax[i].set_title(title)

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

def apply_otsus_threshhold(img):
    blur_img = cv2.GaussianBlur(img, (7,7), 0) # apply blur
    # blur_img = filters.gaussian(img, sigma = 3.0)
    adj = cv2.convertScaleAbs(blur_img, alpha=.1, beta=100) # enhance contrast and brightness
    

    # otsu's thresholding
    T, otsus_method = cv2.threshold(adj, 0,np.max(adj), cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    otsu_mask = otsus_method > 0
    adj[otsu_mask] = 0

    return adj, otsu_mask

def apply_adaptive_threshold(img, block=9, c=4):
    adapt_thresh = cv2.adaptiveThreshold(img, np.max(img), cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=block, C=c)
    mask = adapt_thresh == 255
    img[mask] = 0

    return img

def custom_threshold(img, val=0):
    temp = copy.copy(img)
    if val == 0:
        val = np.mean(img)
    
    mask = img <= val
    temp[mask] = 0
    # img[mask] = 0

    return temp

def create_transparent_img(orig):
    transparent = np.zeros((orig.shape[0], orig.shape[1], 4))

    for i in range(0, len(orig)):
        for j in range(0, len(orig[0])):
            isBlack = True # for checking if all rgb values => black
            for k in range(0,3):
                transparent[i][j][k] = orig[i][j][k]

                if transparent[i][j][k] != 0:
                    isBlack = False

            if not isBlack:
                transparent[i][j][3] = 1
    
    return transparent

def combine_imgs(orig, new_img):
    # orig must be a scalar
    gray_rgb = [.5,.5,.5]
    orig = orig/255 # put into 255 range

    combine = np.zeros((orig.shape[0], orig.shape[1], 4))

    for i in range(0, len(orig)):
        for j in range(0, len(orig[0])):
            if new_img[i][j][3] == 1:
                # use new_img pixel
                combine[i][j] = new_img[i][j]
            else:
                # convert scalar orig into rgb and put into new
                for k in range(0,3):
                    combine[i][j][k] = orig[i][j]*gray_rgb[k]
                combine[i][j][3] = 1
    
    return combine

def create_image_overlay(colored_cells_img, orig_img):
    trans_image = create_transparent_img(colored_cells_img)
    combined = combine_imgs(orig_img, trans_image)

    return combined
folder_loc = 'nd2_files/'
file_names = [
    'SciH-Whole-Ret-4C4-Redd-GFP-DAPI005.nd2', # 4gb
    'Undamaged-structual-example.nd2', # 58mb 
    'S2-6dpi-uoi2506Tg-4R-#13-sxn2003.nd2', #40mb
    'S2-6dpi-uoi2506Tg-1R-#13-sxn2002.nd2', # 70mb
    '6dpi-uoi2500Tg-3R-#17-sxn6006.nd2'
]

with ND2Reader(folder_loc + file_names[2]) as imgs:
    pcna_imgs = get_imgs_from_channel(imgs, 'far red')

# img = compress_stack(pcna_imgs)
img = pcna_imgs[1]

# apply edge detection to original image
edge_canny = feature.canny(img, sigma=1.5)

# apply otsus thresholding to original image
new_img, inv_mask = apply_otsus_threshhold(img)

# overlay otsus binary mask on edge detection to get rid of background
edge_canny[inv_mask] = True

# overlay edges onto otsus threshold image to outline individual cells
new_img[edge_canny] = 0

# use to apply basic thresholding after edge detection
new_img = custom_threshold(new_img, 150)

# apply erosion and dialation to 'open' image and get cells
kernel = np.ones((5,5))
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
opening = cv2.morphologyEx(new_img, cv2.MORPH_OPEN, kernel, iterations=1)

cells = opening != 0
opening[cells] = 255

# count cells
labeled_image, count = measure.label(opening, connectivity=1, return_num=True)
print(count)

# color the counted cells a random color
colored_labeled_img = color.label2rgb(labeled_image, bg_label=0)

# create transparent image

combine_img = create_image_overlay(colored_labeled_img, img)

# don't show counts overlayed on original image
# use_subplots([img, new_img, opening, colored_labeled_img], 
#              ['original', 'after edge detection applied', 'after opening applied', f'final result: {count} cells'],
#              ncols=4, nrows=1
# )

# overlay counted cells on top of original image
use_subplots([img, new_img, opening, combine_img],
             ['original', 'after edge detection', 'after opening', f'counted {count} cells'],
             ncols=4)