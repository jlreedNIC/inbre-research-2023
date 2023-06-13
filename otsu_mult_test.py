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

imgs = []
titles = []


imgs.append(copy(img))
titles.append("original")

# apply unsharp mask to original image
blurred_unsharp = sf.apply_unsharp_filter(img)
imgs.append(copy(blurred_unsharp))
titles.append("blur unsharp")

# blur original image, find edges, then apply to unsharped blurred image
img = filters.gaussian(img, sigma=1.5)
edges = sf.get_edges(img)
imgs.append(copy(edges))
titles.append('edges on orig')

blurred_unsharp[edges] = 0
imgs.append(copy(blurred_unsharp))
titles.append('unsharp with edge')

# apply otsu to original image, then use that threshold to mask the blurred unsharp
_, otsu_mask = sf.apply_skimage_otsu(img)
blurred_unsharp[otsu_mask] = 0
imgs.append(copy(blurred_unsharp))
titles.append('otsu to unsharped')

# apply local threshold on orig then apply to blurred unsharp to get rid of more background
_, local_mask = sf.apply_local_threshold(img)
blurred_unsharp[local_mask==0] = 0

imgs.append(copy(local_mask))
titles.append("local threshold on orig")

imgs.append(copy(blurred_unsharp))
titles.append("local threshold applied to unsharped")


# apply opening morph function to separate cells better
opened = morphology.opening(blurred_unsharp)
imgs.append(copy(opened))
titles.append("opened unsharped")

# apply multi otsu on opened image, then apply mask to opened
opened, _ = sf.apply_multi_otsu(opened)
# anything not black is now white
opened[opened!=0] = 1
imgs.append(copy(opened))
titles.append("multi otsu applied to opened")

# count binary image
labeled_image, count = measure.label(opened, connectivity=1, return_num=True)
print(count)

# overlay colored counts on orig image
colored_img = sf.create_image_overlay(labeled_image, img)
imgs.append(copy(colored_img))
titles.append(f'final result: {count}')

print(len(imgs))

sf.use_subplots(imgs,titles, ncols=round(len(imgs)/3)+1, nrows=3)