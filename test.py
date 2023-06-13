# ------------
# @author   Jordan Reed
# @date     6/13/23
# @brief    testing file
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
# local_1 = filters.threshold_local(img, block_size = 3, method = 'mean', offset = 0.0, mode = 'mirror')#, param = 1.5)
# image_1 = img > local_1

# # apply binary mask
# img[image_1==0]=0
# sf.use_subplots([img, image_1])
# -------- 

# ------------ testing edge masking
# _, mask = sf.apply_skimage_otsu(img)
# otsu_mask = sf.copy.copy(mask)
# otsu_mask[mask==0] = 1
# otsu_mask[mask==1] = 0
# edges_1 = sf.feature.canny(img, sigma=1.5)
# edges_2 = sf.feature.canny(img, sigma=1.5, mask=otsu_mask)


# sf.use_subplots([img, edges_1, edges_2], ncols=3)
# ---------------

# --------- testing edge sigma values
# edges_1 = sf.feature.canny(img, sigma=0.5)
# edges_2 = sf.feature.canny(img, sigma=1.5)
# edges_3 = sf.feature.canny(img, sigma=2.5)


# sf.use_subplots([img, edges_1, edges_2, edges_3], ncols=4)
# -------------------------------

# ---------- testing edge on orig, unsharped, blurred, thresholded
# img = filters.gaussian(img, sigma=2.0)

# otsu_img, otsu_mask = sf.apply_skimage_otsu(img)
# local_img, local_mask = sf.apply_local_threshold(img)
# unsharp_img = sf.apply_unsharp_filter(img)

# edge_1 = sf.get_edges(img)
# edge_2 = sf.get_edges(otsu_img)
# edge_3 = sf.get_edges(local_img)
# edge_4 = sf.get_edges(unsharp_img)

# sf.use_subplots(
#     [img, otsu_img, local_img, unsharp_img, edge_1, edge_2, edge_3, edge_4],
#     ['orig', 'otsu', 'local', 'unsharp', 'orig edge', 'otsu edge', 'local edge', 'unsharp edge'],
#     ncols=4, nrows=2
# )

# ------------------
