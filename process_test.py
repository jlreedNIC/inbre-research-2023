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

demoMode = True

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
# for i in range(len(all_files)):
for i in range(1):
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

    # put images through multiple levels of filtering and count resulting binary image
    dapi_labeled = sf.new_imp_process(dapi_stack)
    pcna_labeled = sf.new_imp_process(pcna_stack)
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
    folder_loc = 'cell_sizes'
    os.mkdir(folder_loc)
    sf.get_cell_sizes(result, folder_loc + '/' + all_files[i] + '-cell-counts.csv')

    # show images
    sf.use_subplots([dapi_stack, dapi_colored, pcna_stack, pcna_colored, f_color], 
                    ['dapi orig', f'dapi result: {dapi_count}', 'pcna orig', f'pcna result: {pcna_count}', f'final result: {f_count}'],
                    ncols=5, nrows=1, figure_title=f'{all_files[i]}-stack')

# -------- testing list for processing

# demoMode = True         # quickly show original and result image
# singleChannel = False   # only show a single channel?
# channel = None          # if single channel is True, which channel do you want to see
#                         # use these variables WITH quotations:    'far red'    or     'DAPI'

# img_stacks = []
# if singleChannel:
#     if channel == None:
#         print('You must change the channel name.')
#         print("Example: 'far red' or 'DAPI'")
#         exit(1)
    
#     img_stacks.append(sf.open_nd2file(folder_loc + file_names[4], channel_name=[channel]))
#     img_titles = [channel]
# else:
#     img_stacks = sf.open_nd2file(folder_loc + file_names[4])
#     img_titles = ['pcna', 'dapi']

# img_steps = []          # imgs to output to screen
# labeled_counts = []     # counted labeled images

# # count cells in each channel
# for i in range(len(img_stacks)):
#     # get compressed img
#     img = sf.compress_stack(img_stacks[i])
#     img_steps.append(img)
#     img_titles.append(img_titles.pop(0)) # get name and put it in correct spot

#     # process and count img
#     labeled_counts.append(sf.new_imp_process(img))
#     # color image and add it to steps
#     img_steps.append(sf.create_image_overlay(labeled_counts[i], img))
#     img_titles.append(f'counted {np.max(labeled_counts[i])}')

# labeled_counts.append(sf.combine_channels(labeled_counts[0], labeled_counts[1]))
# img_steps.append(sf.create_image_overlay(img_steps[0], labeled_counts[-1]))
# img_titles.append(f'final count: {np.max(labeled_counts[-1])}')

# sf.use_subplots(
#     img_steps,
#     img_titles,
#     ncols=len(img_steps)
# )
# # NOTE: current implementation does not combine channels together

# # if demoMode:
# #     # ----------- demo mode -----------
# #     # img = sf.compress_stack(pcna_imgs)
# #     img = pcna_imgs[1]
# #     img2 = neuron_imgs[1]

# #     labeled_image= sf.new_imp_process(img)
# #     labeled2 = sf.new_imp_process(img2)

# #     # create transparent image of cell counts
# #     combine_img = sf.create_image_overlay(labeled_image, img)
# #     combine2 = sf.create_image_overlay(labeled2, img2)

# #     # overlay counted cells on top of original image
# #     sf.use_subplots(
# #         [img, combine_img, img2, combine2],
# #         ['original', f'counted {np.max(labeled_image)} cells', 'neurons', f'counted {np.max(labeled2)}'],
# #         ncols=2, nrows=2)
# # else:
# #     # ------------ debugging use ----------------
# #     img = sf.compress_stack(neuron_imgs)
# #     # img = sf.compress_stack(pcna_imgs)
# #     labeled_image, steps, titles = sf.new_imp_process(img, save_steps=True)

# #     combine_img = sf.create_image_overlay(labeled_image, img)

# #     steps.append(combine_img)
# #     titles.append(f'final result {np.max(labeled_image)}')

# #     # overlay counted cells on top of original image

# #     cols = 3 * round(len(steps)/3)
# #     if cols < len(steps):
# #         cols += 3

# #     sf.use_subplots(
# #         steps,
# #         titles,
# #         ncols=int(cols/3), nrows=3)
