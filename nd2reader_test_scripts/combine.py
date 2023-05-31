# ------------
# @author   Jordan Reed
# @date     5/25/23
# @brief    This file is to test different methods of combining channels to run through pcna.py
# ------------


try:
    from nd2reader import ND2Reader
    import matplotlib.pyplot as plt
    import numpy as np
    import datetime as dt
    import cv2
    from sklearn.cluster import KMeans
except Exception as e:
    print('issue with import')
    print('You may need to install importlib package if issue with nd2reader.')
    print(e)

import custom_color_map as ccm
folder_loc = 'nd2_files/'
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

def get_imgs_from_channel(nd2img, channel_name):
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

def apply_threshold_segm(img, bottom_perc=.2, top_perc=.3):
    blurred = cv2.GaussianBlur(img, (7,7), 0)

    # get max value of image to base mask off of
    maxval = np.max(blurred)
    # num to filter foreground from background
    background_mask = maxval * bottom_perc
    # num to filter out objects
    object_mask = maxval * top_perc

    # apply masks
    background_filter = np.asarray(blurred) <= background_mask
    blurred[background_filter] = 0

    object_filter = (np.asarray(blurred) > 0) & (np.asarray(blurred) <= object_mask)
    blurred[object_filter] = maxval * .15

    return blurred

def apply_clustering(img, k=5):
    blurred = cv2.GaussianBlur(img, (7,7), 0)

    maxval = np.float16(np.max(blurred))
    pic = blurred/maxval
    pic_n = pic.reshape(pic.shape[0]*pic.shape[1], 1)

    kmeans = KMeans(k, random_state=0).fit(pic_n)
    pic2show = kmeans.cluster_centers_[kmeans.labels_]
    cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1])

    return cluster_pic


# ---- main ----
with ND2Reader(folder_loc + file_names[2]) as nd2img:
    print(nd2img.metadata)
    
    pcna_imgs = get_imgs_from_channel(nd2img, 'far red')
    dapi_imgs = get_imgs_from_channel(nd2img, 'DAPI')
    

# print(pcna_imgs[:,0,2])
pcna_compressed = compress_stack(pcna_imgs)
dapi_compressed = compress_stack(dapi_imgs)

# --------------------
# tried thresholding on imgs separately, then see where the maxes are together
pcna_seg = apply_threshold_segm(pcna_compressed)
dapi_seg = apply_threshold_segm(dapi_compressed,.3,.5)

show_image_using_scalar(pcna_seg, ccm.purple_channel)
show_image_using_scalar(dapi_seg, ccm.blue_channel)

maxval = np.max(pcna_seg)
channel_composite = np.zeros(pcna_seg.shape)

ind_match = (np.asarray(pcna_seg) <= (maxval*.4)) & (np.asarray(dapi_seg) <= (maxval*.4))
channel_composite[ind_match] = 0
opp_match = np.invert(ind_match)
channel_composite[opp_match] = 1

show_image_using_scalar(channel_composite)
# --------------------------

# -------------
# tried kmeans clustering on image composite with no blurring, then with blurring
# channel_composite = pcna_compressed + dapi_compressed
# seg_img = apply_clustering(channel_composite)

# show_image_using_scalar(seg_img, ccm.purple_channel)
# ---------------

# -------------
# tried thresholding on image composite
# channel_composite = pcna_compressed + dapi_compressed

# seg_img = apply_threshold_segm(channel_composite)
# show_image_using_scalar(channel_composite)
# show_image_using_scalar(seg_img, ccm.purple_channel)
# think about trying thresholding then kmeans
# ---------------

# -----------
# try clustering on each image, then combine to see where values are
# pcna_cluster = apply_clustering(pcna_imgs[1],5)
# dapi_cluster = apply_clustering(dapi_imgs[1],5)

# show_image_using_scalar(pcna_cluster)
# show_image_using_scalar(dapi_cluster)

# maxval = np.max(pcna_cluster)
# channel_composite = np.zeros(pcna_cluster.shape)

# ind_match = (np.asarray(pcna_cluster) <= (.6)) & (np.asarray(dapi_cluster) <= (.6))
# channel_composite[ind_match] = 0
# opp_match = np.invert(ind_match)
# channel_composite[opp_match] = 1

# show_image_using_scalar(channel_composite)
# ----------------

