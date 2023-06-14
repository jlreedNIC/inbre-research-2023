# ------------
# @author   Jordan Reed
# @date     6/12/23
# @brief    testing multiple otsu threshold.
#
#           turned into unsharp mask, edge detection, mult otsu threshold, etc
#
# ------------

import seg_functions as sf
from nd2reader import ND2Reader
from skimage import filters, morphology, measure
from copy import copy

folder_loc = 'nd2_files/'
file_names = [
    'S2-6dpi-uoi2506Tg-4R-#13-sxn2003.nd2', #40mb
    '6dpi-uoi2505Tg-2R-#17-sxn3003.nd2',
    '6dpi-uoi2505Tg-2R-#17-sxn3002.nd2'
]

demoMode = False

with ND2Reader(folder_loc + file_names[2]) as imgs:
    pcna_imgs = sf.get_imgs_from_channel(imgs, 'far red')
    neuron_imgs = sf.get_imgs_from_channel(imgs, 'DAPI')

img = pcna_imgs[1]

# ---- unsharp mask and edge detection

if demoMode:
    labeled_image = sf.new_imp_process(img)
    count = sf.np.max(labeled_image)
    colored_img = sf.create_image_overlay(labeled_image, img)

    sf.use_subplots([img, colored_img], ['original', f'final result: {count}'])
else:
    labeled_image, steps, titles = sf.new_imp_process(img, True)
    # overlay colored counts on orig image
    colored_img = sf.create_image_overlay(labeled_image, img)
    count = sf.np.max(labeled_image)
    steps.append(colored_img)
    titles.append(f'final image {count}')
    print(len(steps))

    cols = 3 * round(len(steps)/3)
    if cols < len(steps):
        cols += 3
    sf.use_subplots(steps,titles, ncols=int(cols/3), nrows=3)