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
from skimage import filters

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

# ----- comparing opencv otsu to skimage otsu
# otsu_current, _ = sf.apply_otsus_threshold(img)

# img = sf.filters.gaussian(img, sigma=1.5)

# otsu_1 = sf.filters.threshold_otsu(img)
# mask = img <= otsu_1 
# img[mask] = 0
# print(otsu_1)

# sf.use_subplots([otsu_current, img])
# --------------

# ----- testing multi otsu
# img = filters.gaussian(img, sigma=1.5)
# numc=4
# thresholds = filters.threshold_multiotsu(img,classes=numc)
# regions = sf.np.digitize(img, bins=thresholds)
# mask = regions==(numc-1)
# import copy
# temp = copy.copy(img)
# temp[mask] = 0

# sf.use_subplots([img, regions, temp], ncols=3)
# -------------

# ------- testing skimage local threshold
# img = filters.gaussian(img, sigma=1.5)


# -------- 

# ---- unsharp mask and edge detection

# img = filters.gaussian(img, sigma=1.5)
unsharp_1 = filters.unsharp_mask(img, radius = 5, amount=20.0)
blurred_unsharp = filters.gaussian(unsharp_1, sigma = 1.5)
edge_1 = sf.feature.canny(blurred_unsharp)
edge_2 = sf.filters.sobel(blurred_unsharp)

# sf.use_subplots([img, unsharp_1, blurred_unsharp, edge_1, edge_2], ncols=5)

img = filters.gaussian(img, sigma=1.5)
edge_3 = sf.feature.canny(img)
blurred_unsharp[edge_3] = 0

# sf.use_subplots([img, blurred_unsharp, edge_3], ncols=3)

# apply opening after unsharp edge detection
# from skimage import morphology

# opened = morphology.opening(blurred_unsharp)
# sf.use_subplots([img, blurred_unsharp, opened], ncols=3)

otsu_2 = sf.filters.threshold_otsu(img)
mask_2 = img <= otsu_2
blurred_unsharp[mask_2] = 0

# sf.use_subplots([img, blurred_unsharp], ncols=3)

from skimage import morphology, measure

opened = morphology.opening(blurred_unsharp)
# sf.use_subplots([img, blurred_unsharp, opened], ncols=3)

thresholds = filters.threshold_multiotsu(opened,classes=3)
regions = sf.np.digitize(opened, bins=thresholds)
mask = regions!=(3-1)
import copy
temp = copy.copy(opened)
temp[mask] = 0
temp[temp!=0] = 1

labeled_image, count = measure.label(temp, connectivity=1, return_num=True)
print(count)

colored_img = sf.create_image_overlay(labeled_image, img)

sf.use_subplots([img, blurred_unsharp, opened, temp, colored_img], ncols=5)