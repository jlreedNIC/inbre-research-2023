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

import os

# get all files from nd2 files dir
all_files = os.listdir(folder_loc)

demoMode = False

with ND2Reader(folder_loc + file_names[0]) as imgs:
    pcna_imgs = sf.get_imgs_from_channel(imgs, 'far red')
    neuron_imgs = sf.get_imgs_from_channel(imgs, 'DAPI')

# ---- looping through every image
# if demoMode:
#     for i in range(0, len(pcna_imgs)):
#         img = pcna_imgs[i]
#         img2 = neuron_imgs[i]

#         labeled_image = sf.new_imp_process(img)
#         labeled_image2 = sf.new_imp_process(img2)
#         count = sf.np.max(labeled_image)
#         count2 = sf.np.max(labeled_image2)
#         colored_img = sf.create_image_overlay(labeled_image, img)
#         colored_img2 = sf.create_image_overlay(labeled_image2, img2)

#         sf.use_subplots([img, colored_img, img2, colored_img2], ['pcna orig', f'final result: {count}', 'DAPI orig', f'final: {count2}'], nrows = 2)

# ---- currently loops through all files in nd2 folder
print(f'Found {len(all_files)} files. Starting now...\n')
for i in range(len(all_files)):
# for i in range(1):
    print(f'Now opening - {all_files[i]} -')
    start = sf.dt.datetime.now()

    with ND2Reader(folder_loc + all_files[i]) as imgs:
        try:
            pcna_imgs = sf.get_imgs_from_channel(imgs, 'far red')
            neuron_imgs = sf.get_imgs_from_channel(imgs, 'DAPI')
        except Exception as e:
            print(e)
            continue

    # compress stack into one image (provides best result so far)
    dapi_stack = sf.compress_stack(neuron_imgs)
    pcna_stack = sf.compress_stack(pcna_imgs)

    pcna_stack, roi_mask = sf.select_roi(pcna_stack, True)

    # put images through multiple levels of filtering and count resulting binary image
    dapi_labeled = sf.new_imp_process(dapi_stack, mask=roi_mask)
    pcna_labeled = sf.new_imp_process(pcna_stack, mask=roi_mask)
    # AND images together and count again
    result = sf.combine_channels(pcna_labeled, dapi_labeled)

    # get counts from each image
    dapi_count = sf.np.max(dapi_labeled)
    pcna_count = sf.np.max(pcna_labeled)
    f_count = sf.np.max(result)

    # overlay the colored counts onto original images
    dapi_colored = sf.create_image_overlay(dapi_labeled, dapi_stack)
    pcna_colored = sf.create_image_overlay(pcna_labeled, pcna_stack)
    f_color = sf.create_image_overlay(result, pcna_stack)

    stop = sf.dt.datetime.now()
    print(f'Time taken to count: {stop-start}')

    import os
    cell_folder = 'cell_sizes'
    try:
        os.mkdir(cell_folder)
    except:
        pass
    sf.get_cell_sizes(result, cell_folder + '/' + all_files[i] + '-cell-counts.csv')

    # show images
    sf.use_subplots([dapi_stack, dapi_colored, pcna_stack, pcna_colored, f_color], 
                    ['dapi orig', f'dapi result: {dapi_count}', 'pcna orig', f'pcna result: {pcna_count}', f'final result: {f_count}'],
                    ncols=5, nrows=1, figure_title=f'{all_files[i]}-stack')

# TO DO:
# put all user changeable variables into separate file
# for testing arg parse!
# import argparse
# import os

# parser = argparse.ArgumentParser(description="Program to count PCNA")
# parser.add_argument('filepath', type=str, help='path to nd2 file folder')
# parser.add_argument('-d', '--demo', action='store_true')
# args = parser.parse_args()

# print(args.filepath, args.demo)

# # get all files from nd2 files dir
# all_files = os.listdir(args.filepath)

# folder_loc = args.filepath + '/'
# demoMode = args.demo