# ------------
# @author   Jordan Reed
# @date     6/2/23
# @brief    This file is testing edge detection
# ------------

from nd2reader import ND2Reader
import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage import filters, feature
# from skimage.data import camera
# from skimage.util import compare_images

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

folder_loc = 'nd2_files/'
file_names = [
    'SciH-Whole-Ret-4C4-Redd-GFP-DAPI005.nd2', # 4gb
    'Undamaged-structual-example.nd2', # 58mb 
    'S2-6dpi-uoi2506Tg-4R-#13-sxn2003.nd2', #40mb
    'S2-6dpi-uoi2506Tg-1R-#13-sxn2002.nd2' # 70mb
]

with ND2Reader(folder_loc + file_names[2]) as imgs:
    pcna_imgs = get_imgs_from_channel(imgs, 'DAPI')

# img = compress_stack(pcna_imgs)
img = pcna_imgs[1]

edge_prewitt = filters.prewitt(img)
edge_scharr = filters.scharr(img)
edge_sobel = filters.sobel(img)
edge_canny = feature.canny(img, sigma=1.5)

fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True,
                         figsize=(10, 10))

ax = axes.ravel()

ax[0].imshow(img, cmap='gray')
ax[0].set_title('original')

ax[1].imshow(edge_scharr, cmap=plt.cm.gray)
ax[1].set_title('scharr Edge Detection')

ax[2].imshow(edge_sobel, cmap=plt.cm.gray)
ax[2].set_title('Sobel Edge Detection')

ax[3].imshow(edge_canny, cmap=plt.cm.gray)
ax[3].set_title('canny Edge Detection')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()