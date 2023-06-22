# ------------
# @author   Jordan Reed
# @date     6/7/23
# @brief    Image segmentation functions
#
# ------------

from nd2reader import ND2Reader
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime as dt
from copy import copy

from skimage import filters, feature, color, measure, morphology

    
def open_nd2file(filepath:str, channel_name=['far red', 'DAPI']):
    """
    Opens the nd2 file at the filepath and grabs all images in the specified channels. The default channels are 'far red' (pcna marker) and 'DAPI' (neurons).

    :param filepath: String containing the file path of the nd2 file to open.
    :param channel_name: List of strings containing channel names to get images from, defaults to ['far red', 'DAPI']

    :return: list of list of images, i.e. far red images will be in img_stacks[0]
    """
    img_stacks = []

    # try:
    #     imgs = ND2Reader(filepath)
    # except Exception as e:
    #     # print(f'Error opening file: {filepath}.')
    #     # print(e)
    #     raise(Exception(f'Error opening file: {filepath}'))

    # for i in range(len(channel_name)):
    #     img_stacks.append(get_imgs_from_channel(imgs, channel_name[i]))
    
    # imgs.close()

    with ND2Reader(filepath) as imgs:
        for i in range(len(channel_name)):
            img_stacks.append(get_imgs_from_channel(imgs, channel_name[i]))
    
    return img_stacks

def get_imgs_from_channel(nd2img:ND2Reader, channel_name:str, debug=False):
    """Get all images in the stack for a specified channel in an open ND2 file.

    :param ND2Reader nd2img:    Open ND2 file using ND2Reader
    :param str channel_name:    channel name given in string format. Ex - 'far red' or 'DAPI'
    :param boolean debug:  if True, show print statements to console

    :return list: list of images in a numpy array
    """
    # get all channels and all z
    nd2img.iter_axes = 'cz'
    # get index of channel
    channel_num = nd2img.metadata['channels'].index(channel_name)+1
    if debug:
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

def select_roi(img, debug=False):
    temp = copy(img)
    if np.max(temp) > 1:
        temp /= np.max(temp)

    if debug:
        print('\nMaking ROI selection...')
    
    # select rectangular roi
    window = 'ROI Selection'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    res = cv2.selectROI(window, temp, showCrosshair=False)
    res = np.array(res)
    cv2.destroyAllWindows()

    if debug:
        print(f'\nPoints: {res}')

    # make copy of image and fill roi with white
    img_copy = copy(temp)
    cv2.rectangle(img_copy, res, 1, -1)

    # create mask to show roi
    mask = img_copy != 1
    temp[mask] = 0

    # show masked image
    print('\nPress any key to close the window. DO NOT CLOSE THE WINDOW YOURSELF.')
    cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
    cv2.imshow('ROI', temp)
    cv2.waitKey(0)

    cv2.destroyWindow('ROI')
    
    return temp, mask

def get_edges(img):
    """
    Applies canny edge detection to image and returns boolean image containing edges found.
    Sigma applied for Gaussian step is set to 2.5

    :param img: Image to apply edge detection to. Does not need blur
    :return: boolean image of edges
    """
    edges = feature.canny(img, sigma=2.5)

    return edges

def apply_unsharp_filter(img):
    """
    Applies the unsharp mask filter with a radius of 5 and amount of 20 to image.
    Then applies gaussian blur (sigma 1.5) to clean it up

    :param img: image to apply filter to, numpy array preferred

    :return:    image, numpy array
    """
    unsharp = filters.unsharp_mask(img, radius = 5, amount = 20.0)
    # put blur on unsharp
    blurred = filters.gaussian(unsharp, sigma=1.5)

    return blurred

def apply_local_threshold(img):
    """
    applies gaussian blur (sigma 1.5), then applies a local thresholding technique.
    Block size for local is 3, method is 'mean', and offset is 0.0. 

    :param img: image, numpy array preferred

    :return:    image with local threshold mask applied
                mask produced by local threshold
    """
    img = filters.gaussian(img, sigma=1.5)
    local = filters.threshold_local(img, block_size=3, method='mean', offset = 0.0, mode="mirror")
    mask = img > local

    # apply mask
    img[mask==0] = 0

    return img, mask

def apply_otsus_threshold(img):
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

def apply_skimage_otsu(img):
    """
    Creates copy of image to apply Otsu's thresholding implemented by skimage. 
    Applies the Otsu mask to the image

    :param img: Image to apply algo to, numpy array preferred

    :return:    image with mask applied
                binary mask
    """
    temp = copy(img)
    otsu_1 = filters.threshold_otsu(temp)
    mask = temp <= otsu_1 
    temp[mask] = 0

    return temp, mask

def apply_multi_otsu(img, c = 3):
    """
    Applies Multi Otsu thresholding to break image up into 3 sections (background, middle ground, foreground). 
    Applies mask to hide anything not in the foreground

    :param img: image, numpy array preferred
    :param c:   classes or section to break image up into, defaults to 3

    :return:    image with mask applied
                binary mask
    """
    thresholds = filters.threshold_multiotsu(img, c)
    regions = np.digitize(img, bins=thresholds)
    mask = regions!=(c-1)
    img[mask] = 0

    return img, mask

def custom_threshold(img, threshold=0):
    """Applies a basic thresholding by zeroing any pixel less than the given value

    :param array img: image in array format
    :param int val: threshold value to 0 out, defaults to 0

    :return array: image after thresholding applied
    """
    temp = copy(img)
    if threshold == 0:
        threshold = np.mean(img[img!=0]) * .9
    
    print(f'threshold used: {threshold}')
    
    mask = img <= threshold
    temp[mask] = 0

    return temp

def use_subplots(imgs:list, titles = [], ncols = 2, nrows=1, figure_title = None):
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

    if figure_title != None:
        fig.canvas.manager.set_window_title(figure_title)

    plt.tight_layout()
    plt.show()

def create_image_overlay(labeled_image, orig_img):
    """Combines two images by first creating an rbg image from the labeled image, 
    then overlaying the original image onto the colored image wherever the colored image is black.
    Continuation of process image function.

    :param array labeled_image: array containing labels from count function; array of shape(h,w)
    :param array orig_img: array of shape (h,w)

    :return array: array of shape (h,w,3)
    """
    # color the counted cells a random color

    # TO DO:
    # FIX RGB VALUES SO BETWEEN 0 AND 1
    colored_labeled_img = color.label2rgb(labeled_image, bg_label=0)

    # convert original image to grayscale and scale down brightness
    gray_img = orig_img/np.max(orig_img)
    if np.mean(gray_img) < .3:
        gray_img = color.gray2rgb(orig_img)/np.max(orig_img) * 1.5
    else:
        gray_img = color.gray2rgb(orig_img)/np.max(orig_img)
    # colored_labeled_img = colored_labeled_img

    # find where colors are
    colored = colored_labeled_img != [0,0,0]
    # find where colors aren't by performing OR on above
    non_c = np.any(colored, 2)
    non_c = np.invert(non_c)

    # don't have to create new array
    # set non colored parts to original image
    colored_labeled_img[non_c] = gray_img[non_c]

    return colored_labeled_img

def new_imp_process(img, debug=False, mask=None):
    """
    Applies filters to image in order to count the cells in the image. Can also return images of each step applied

    :param img: numpy array of scalar values; shape (h,w)
    :param save_steps: boolean whether or not to return intermediary images, defaults to False
    :return: labeled counted image as numpy array
                list of intermediary images (OPT)
                list of image titles (OPT)
    """
    steps = []  # for debugging use
    titles = [] # for debugging use

    if debug:
        steps.append(copy(img))
        titles.append('original')
        print('\nStarting processing.')

    # unsharp to original
    progress_img = apply_unsharp_filter(img)
    if debug:
        steps.append(copy(progress_img))
        titles.append('unsharp on orig')
        print('Unsharp filter applied.')
    
    # blur orig, find edges on orig, then apply to progress
    blur_img = filters.gaussian(img, sigma=2.0)
    edges = get_edges(blur_img)
    progress_img[edges] = 0
    if debug:
        steps.append(copy(progress_img))
        titles.append('edges on orig to progress')
        print('Edges found.')

    # apply otsu to orig, then apply to progress
    _, otsu_mask = apply_skimage_otsu(blur_img)
    progress_img[otsu_mask] = 0
    if debug:
        steps.append(copy(progress_img))
        titles.append('otsu on orig to progress')
        print("Otsu's threshold applied.")
    
    # apply local on orig, then to progress (gets rid of more background)
    _, local_mask = apply_local_threshold(blur_img)
    progress_img[local_mask==0] = 0
    if debug:
        steps.append(copy(progress_img))
        titles.append('local to progress')
        print("Local threshold applied.")

    # apply opening morph to separate cells better
    # progress_img = morphology.opening(progress_img)#, morphology.disk(3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    progress_img = cv2.morphologyEx(progress_img, cv2.MORPH_OPEN, kernel, iterations=1)
    if debug:
        steps.append(copy(progress_img))
        titles.append('opening applied')
        print("Morphological opening applied.")

    # apply multi otsu on opened, then to opened
    progress_img, _ = apply_multi_otsu(progress_img)
    progress_img[progress_img!=0] = 1
    if debug:
        steps.append(copy(progress_img))
        titles.append('multi otsu applied')
        print("Multi Otsu threshold applied (separated background from foreground).")
    
    if mask is not None:
        progress_img[mask] = 0

    # count binary image
    final_image, count = measure.label(progress_img, connectivity=1, return_num=True)
    if debug:
        steps.append(copy(final_image))
        titles.append(f'counted: {count}')
        print(f'Image segmented. {count} cells counted.')
    
    # remove small objects and relabel array
    size = 10
    final_image = morphology.remove_small_objects(final_image, min_size=size)
    final_image, count = measure.label(final_image, connectivity=1, return_num=True)
    if debug:
        steps.append(copy(final_image))
        titles.append(f'artifacts removed <= {size}. Count: {count}')
        print(f'Removed artifacts smaller than {size} and recounted: {count} cells.')

    # if save_steps:
    return final_image, steps, titles
    # else:
    #     return final_image

def combine_channels(pcna_img, dapi_img, debug = False):
    # pcna and dapi images must be labeled!
    steps = []
    titles = []

    # find areas where both pictures have content
    img_and = np.logical_and(pcna_img, dapi_img)
    if debug:
        steps.append(copy(img_and))
        titles.append('pcna AND dapi')
        print('Combined images using logical AND operation.')

    # label and count
    img_and, count = measure.label(img_and, connectivity=1, return_num=True)
    if debug:
        steps.append(copy(img_and))
        titles.append(f'counted {count}')
        print(f'Counted image: {count} cells')

    # remove small artifacts
    size = 7
    img_and = morphology.remove_small_objects(img_and, min_size=size)
    # relabel
    img_and, count = measure.label(img_and, connectivity=1, return_num=True)
    if debug:
        steps.append(copy(img_and))
        titles.append(f'removed artifacts < {size}')
        print(f'Removed artifacts smaller than {size} and recounted: {count} cells')

    # if debug:
    return img_and, steps, titles
    # else:
    #     return img_and

def get_cell_sizes(img, filename:str, debug=False):
    """
    Will count the size of each object in the given image and output it to the given file.

    :param img: image that is a labeled format
    :param filename: file to output data, a csv file
    :param debug: if True, show print statements regarding process of function
    """
    # loop through number of cells and sum total pixels

    num_cells = np.max(img)

    if debug:
        print(f'\nSaving cell size info in - {filename} - ...')

    try:
        f = open(filename, 'x')
    except FileExistsError as e:
        f = open(filename, 'w')

    cells = []

    f.write('cell,size,x-coordinate,y-coordinate,\n')

    for i in range(num_cells):
        # get size of cell
        size = (img == (i+1)).sum()
        # print(f'cell #{i+1}: {size} pixels')

        # get a single index in cell
        index = np.where(img == (i+1))
        x = index[0][0]
        y = index[1][0]

        # print(f'{i+1}: {x},{y} = {img[x][y]}')
        f.write(f'{i+1},{size},{x},{y},\n')
        cells.append(size)
    
    f.close()

    if debug:
        print(f'Size of image: {img.shape[0]} x {img.shape[1]}')
        print(f'Number of cells: {num_cells}')
        print(f'Average cell size: {np.mean(cells)} pixels')