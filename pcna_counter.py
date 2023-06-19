# ------------
# @author   Jordan Reed
# @date     6/7/23
# @brief    Program to find and count cells on given channels using edge 
#           detection, thresholding, and morphological techniques.
# ------------

import seg_functions as sf
import numpy as np

# # testing command line filepath passing
import argparse
import os

# TO DO:
# put all user changeable variables into separate file

# parser = argparse.ArgumentParser(description="Program to count PCNA")
# parser.add_argument('filepath', type=str, help='path to nd2 file folder')
# parser.add_argument('-d', '--demo', action='store_true')
# args = parser.parse_args()

# print(args.filepath, args.demo)

# # get all files from nd2 files dir
# all_files = os.listdir(args.filepath)


folder_loc = 'nd2_files/'
cell_folder = 'cell_sizes/'

all_files = os.listdir(folder_loc)

# create folder to store cell counts
try:
    os.mkdir(cell_folder)
except FileExistsError as e:
    # print(f'{cell_folder} exists!')
    pass

file_names = [
    'S2-6dpi-uoi2506Tg-4R-#13-sxn2003.nd2', #40mb
    '6dpi-uoi2505Tg-2R-#17-sxn3003.nd2',
    '6dpi-uoi2505Tg-2R-#17-sxn3002.nd2',
    'gl22-6dpi-3R-#12-sxn3P002.nd2', # not catching all cells
    '6dpi-uoi2505Tg-5R-#18-sxn1004.nd2'
]

# quickly show original and result image
show_steps = True

# only show a single channel
singleChannel = True

# if single channel is True, which channel do you want to see
# use these variables WITH quotations:    'far red'    or     'DAPI'
channel = 'far red'          


# file = file_names[4]
# folder_loc = args.filepath + '/'
file = all_files[1]
# demoMode = args.demo

if singleChannel:
    if channel == None:
        print('You must change the channel name.')
        print("Example: 'far red' or 'DAPI'")
        exit(1)
    
    print(f'Opening file: - {file} -\n')
    try:
        img_stack = sf.open_nd2file(folder_loc + file, channel_name=[channel])
    except Exception as e:
        print(f'Error opening {file}.')
        print(e)
        exit(1)
    
    # rename for file conventions
    if channel == 'far red':
        channel = 'pcna'

    # compress stacks to single img
    img = sf.compress_stack(img_stack[0])
    print('\nImages compressed.')

    # process and count imgs
    print('\nApplying filters...')
    labeled, steps, titles = sf.new_imp_process(img, debug=True)

    # get counts from each image
    count = np.max(labeled)

    # color each image and overlay with original
    print('\nColoring image and overlaying with original...')
    color = sf.create_image_overlay(labeled, img)

    # Output results to terminal
    print(f'\nFinal count:   {count} cells')

    # count cell sizes of the result image
    sf.get_cell_sizes(labeled, cell_folder + file[:-4] + '-' + channel + '-cell-counts.csv', debug=True)

    # show images
    print('\nDisplaying image...')
    # show steps
    cols = 3 * round(len(steps)/3)
    if cols < len(steps):
        cols += 3

    sf.use_subplots(
        steps,
        titles,
        ncols=int(cols/3), nrows=3,
        figure_title= file[:-4] + '-' + channel + '-RESULTS.nd2'
    )
else:
    # 2 channels
    print(f'Opening file: - {file} -\n')
    
    try:
        pcna_imgs, dapi_imgs = sf.open_nd2file(folder_loc + file)
    except Exception as e:
        print(f'Error opening {file}.')
        print(e)
        exit(1)

    # compress stacks to single img
    pcna_img = sf.compress_stack(pcna_imgs)
    dapi_img = sf.compress_stack(dapi_imgs)
    if show_steps:
        print('\nImages compressed.')

    # # select roi
    # _, mask = sf.select_roi(pcna_img, True)

    # process and count imgs
    # pcna_steps and pcna_titles will be empty lists if show_steps is selected
    pcna_labeled, pcna_steps, pcna_titles = sf.new_imp_process(pcna_img, debug=show_steps)
    dapi_labeled, dapi_steps, dapi_titles = sf.new_imp_process(dapi_img, debug=show_steps)
    if show_steps:
        print('\nApplying filters...')

    # perform AND combination
    # result_steps and result_titles will be empty lists if show_steps is selected
    result_labeled, result_steps, result_titles = sf.combine_channels(pcna_labeled, dapi_labeled, debug=show_steps)
    if show_steps:
        print('\nCombining channels...')

    # get counts from each image
    pcna_count = np.max(pcna_labeled)
    dapi_count = np.max(dapi_labeled)
    result_count = np.max(result_labeled)

    # color each image and overlay with original
    if show_steps:
        print('\nColoring images and overlaying with original...')
    pcna_color = sf.create_image_overlay(pcna_labeled, pcna_img)
    dapi_color = sf.create_image_overlay(dapi_labeled, dapi_img)
    result_color = sf.create_image_overlay(result_labeled, pcna_img)

    # Output results to terminal
    print(f'\nPCNA image:   {pcna_count} cells')
    print(f'DAPI image:   {dapi_count} cells')
    print(f'Final result: {result_count} cells\n')

    # count cell sizes of the result image
    sf.get_cell_sizes(result_labeled, cell_folder + file + '-cell-counts.csv', debug=show_steps)

    if show_steps:
        print('Showing PCNA image filter steps...\n')
        # show pcna steps
        cols = 3 * round(len(pcna_steps)/3)
        if cols < len(pcna_steps):
            cols += 3

        sf.use_subplots(
            pcna_steps,
            pcna_titles,
            ncols=int(cols/3), nrows=3,
            figure_title= file[:-4] + '-PCNA-STEPS.nd2'
        )

        print('Showing DAPI image filter steps...\n')
        # show dapi steps
        cols = 3 * round(len(dapi_steps)/3)
        if cols < len(dapi_steps):
            cols += 3

        sf.use_subplots(
            dapi_steps,
            dapi_titles,
            ncols=int(cols/3), nrows=3,
            figure_title= file[:-4] + '-DAPI-STEPS.nd2'
        )

        print('Showing final image results...\n')
    # show results
    sf.use_subplots(
        [dapi_img, dapi_color, pcna_img, pcna_color, result_color],
        [ 'DAPI orig', f'DAPI: {dapi_count}', 'PCNA orig', f'PCNA: {pcna_count}',f'final: {result_count}'],
        ncols=5,
        figure_title= file[:-4] + '-RESULTS.nd2'
    )
