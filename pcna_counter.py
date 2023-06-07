# ------------
# @author   Jordan Reed
# @date     6/7/23
# @brief    Program to find and count cells on given channels using edge 
#           detection, thresholding, and morphological techniques.
# ------------

from nd2reader import ND2Reader
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime as dt
import copy

from skimage import filters, feature, color, measure

def get_imgs_from_channel(nd2img:ND2Reader, channel_name:str):
    """Get all images in the stack for a specified channel in an open ND2 file.

    :param ND2Reader nd2img: Open ND2 file using ND2Reader
    :param str channel_name: channel name given in string format. Ex - 'far red' or 'DAPI'
    :return list: list of images in a numpy array
    """
    # get all channels and all z
    nd2img.iter_axes = 'cz'
    # get index of channel
    channel_num = nd2img.metadata['channels'].index(channel_name)+1

    print(f'{channel_name} is at channel number {channel_num}')

    # get number of images in channel
    z_img_num = nd2img.sizes['z']

    new_arr = np.zeros((z_img_num, nd2img.metadata['height'], nd2img.metadata['width']))
    # put all images into new array
    index = 0
    for i in range( (channel_num-1)*z_img_num, channel_num*z_img_num ):
        new_arr[index] = nd2img[i]
        index += 1
    
    return new_arr

def compress_stack(img_stack:list):
    """Combines all images in an array into one image by taking the maximum value of the images for each pixel.

    :param list img_stack: list of gray-scale images
    :return numpy array: a numpy array containing a single image
    """
    # new array created by taking max value of all imgs in stack
    new_arr = np.amax(img_stack, 0)

    return new_arr

def use_subplots(imgs:list, titles = [], ncols = 2, nrows=1):
    """Show images in a matplotlib subplot format.

    :param list imgs: List containing images to show.
    :param list titles: List containing titles of images, defaults to []
    :param int ncols: number of columns in subplot, defaults to 2
    :param int nrows: number of rows in subplot, defaults to 1
    """
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
    """Applies Otsu's Thresholding method to an image. 
    First applies a Gaussian Blur, then enhances the brightness and 
    contrast of the image before applying thresholding.

    :param list img: Image in an array format
    :return array(s): Array with thresholding applied, as well as the binary mask that was applied to image
    """
    blur_img = cv2.GaussianBlur(img, (7,7), 0) # apply blur
    # blur_img = filters.gaussian(img, sigma = 3.0)
    adj = cv2.convertScaleAbs(blur_img, alpha=.1, beta=100) # enhance contrast and brightness
    

    # otsu's thresholding
    T, otsus_method = cv2.threshold(adj, 0,np.max(adj), cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    otsu_mask = otsus_method > 0
    adj[otsu_mask] = 0

    return adj, otsu_mask

def apply_adaptive_threshold(img, block=9, c=4):
    """Applies adaptive thresholding to an image using OpenCV's adaptive threshold.

    :param array img: image in an array format
    :param int block: block size to break image into, defaults to 9
    :param int c: c for adaptive threshold, defaults to 4
    :return array: array after thresholding applied
    """
    adapt_thresh = cv2.adaptiveThreshold(img, np.max(img), cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=block, C=c)
    mask = adapt_thresh == 255
    img[mask] = 0

    return img

def custom_threshold(img, val=0):
    """Applies a basic thresholding by zeroing any pixel less than the given value

    :param array img: image in array format
    :param int val: threshold value to 0 out, defaults to 0
    :return array: image after thresholding applied
    """
    temp = copy.copy(img)
    if val == 0:
        val = np.mean(img)
    
    mask = img <= val
    temp[mask] = 0

    return temp

def create_transparent_img(orig):
    """Converts an RGB image into RGBA and makes all black pixels transparent.

    :param array orig: image in array format with shape (h,w,3)
    :return array: image with shape (h,w,4)
    """
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
    """Combines an image in gray scale with an image with a transparent background into a single image to create an overlay.

    :param array orig: image in array format with shape (h,w)
    :param array new_img: image in array with shape (h,w,4)
    :return array: image with shape (h,w,4)
    """
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
    """Calls 2 functions to make an image have a transparent background, then overlay the two images

    :param array colored_cells_img: array of shape(h,w,4)
    :param array orig_img: array of shape (h,w)
    :return array: array of shape (h,w,4)
    """
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