# ------------
# @author   Jordan Reed
# @date     6/7/23
# @brief    Program to find and count cells on given channels using edge 
#           detection, thresholding, and morphological techniques.
# ------------

import seg_functions as sf

from nd2reader import ND2Reader
import numpy as np

folder_loc = 'nd2_files/'
file_names = [
    'SciH-Whole-Ret-4C4-Redd-GFP-DAPI005.nd2', # 4gb
    'Undamaged-structual-example.nd2', # 58mb 
    'S2-6dpi-uoi2506Tg-4R-#13-sxn2003.nd2', #40mb
    'S2-6dpi-uoi2506Tg-1R-#13-sxn2002.nd2', # 70mb
    '6dpi-uoi2500Tg-3R-#17-sxn6006.nd2'
]

with ND2Reader(folder_loc + file_names[2]) as imgs:
    pcna_imgs = sf.get_imgs_from_channel(imgs, 'far red')
    neuron_imgs = sf.get_imgs_from_channel(imgs, 'DAPI')

# img = sf.compress_stack(pcna_imgs)
img = pcna_imgs[1]
img2 = neuron_imgs[1]

labeled_image = sf.process_image(img)
labeled2 = sf.process_image(img2)

# create transparent image of cell counts
combine_img = sf.create_image_overlay(labeled_image, img)
combine2 = sf.create_image_overlay(labeled2, img2)

# overlay counted cells on top of original image
sf.use_subplots(
    [img, combine_img, img2, combine2],
    ['original', f'counted {np.max(labeled_image)} cells', 'neurons', f'coutned {np.max(labeled2)}'],
    ncols=2, nrows=2)