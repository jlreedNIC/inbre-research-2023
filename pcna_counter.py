# ------------
# @author   Jordan Reed
# @date     6/7/23
# @brief    Program to find and count cells on given channels using edge 
#           detection, thresholding, and morphological techniques.
# ------------

import seg_functions as sf

from nd2reader import ND2Reader
import numpy as np

# testing command line filepath passing
# import argparse

# parser = argparse.ArgumentParser(description="Program to count PCNA")
# parser.add_argument('filepath', type=str, help='path to nd2 file')
# parser.add_argument('-d', '--demo', action='store_true')
# args = parser.parse_args()

# print(args.filepath, args.demo)
# get all files from nd2 files dir
# all_files = os.listdir(args.filepath)

folder_loc = 'nd2_files/'
file_names = [
    'S2-6dpi-uoi2506Tg-4R-#13-sxn2003.nd2', #40mb
    '6dpi-uoi2505Tg-2R-#17-sxn3003.nd2',
    '6dpi-uoi2505Tg-2R-#17-sxn3002.nd2',
    'gl22-6dpi-3R-#12-sxn3P002.nd2', # not catching all cells
    '6dpi-uoi2500Tg-3R-#17-sxn6001.nd2'
]

demoMode = True         # quickly show original and result image
singleChannel = False   # only show a single channel?
channel = None          # if single channel is True, which channel do you want to see
                        # use these variables WITH quotations:    'far red'    or     'DAPI'

with ND2Reader(folder_loc + file_names[4]) as imgs:
    pcna_imgs = sf.get_imgs_from_channel(imgs, 'far red')
    neuron_imgs = sf.get_imgs_from_channel(imgs, 'DAPI')

# NOTE: current implementation does not combine channels together

if demoMode:
    # ----------- demo mode -----------
    # img = sf.compress_stack(pcna_imgs)
    img = pcna_imgs[1]
    img2 = neuron_imgs[1]

    labeled_image= sf.new_imp_process(img)
    labeled2 = sf.new_imp_process(img2)

    # create transparent image of cell counts
    combine_img = sf.create_image_overlay(labeled_image, img)
    combine2 = sf.create_image_overlay(labeled2, img2)

    # overlay counted cells on top of original image
    sf.use_subplots(
        [img, combine_img, img2, combine2],
        ['original', f'counted {np.max(labeled_image)} cells', 'neurons', f'counted {np.max(labeled2)}'],
        ncols=2, nrows=2)
else:
    # ------------ debugging use ----------------
    img = sf.compress_stack(neuron_imgs)
    # img = sf.compress_stack(pcna_imgs)
    labeled_image, steps, titles = sf.new_imp_process(img, save_steps=True)

    combine_img = sf.create_image_overlay(labeled_image, img)

    steps.append(combine_img)
    titles.append(f'final result {np.max(labeled_image)}')

    # overlay counted cells on top of original image

    cols = 3 * round(len(steps)/3)
    if cols < len(steps):
        cols += 3

    sf.use_subplots(
        steps,
        titles,
        ncols=int(cols/3), nrows=3)
