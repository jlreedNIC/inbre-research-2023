# ------------
# @author   Jordan Reed
# @date     6/7/23
# @brief    Program to find and count cells on given channels using edge 
#           detection, thresholding, and morphological techniques.
# ------------

import seg_functions as sf
import numpy as np
import os

# -------------------------
# user changeable variables
# -------------------------

# name of folder/directory that contains nd2 files for analyzing
# may have issue if there is a file that is not .nd2 in there
# TO DO: handle not .nd2 files
folder_loc = 'nd2_files/'

# name of folder/directory to create to hold all .csv files that have the sizes of cells
cell_folder = 'cell_sizes/'

# Whether or not to show the intermediary steps in cell count process
# this is the default when looking at a single channel
showSteps = True

# only show a single channel
singleChannel = False

# if single channel is True, which channel do you want to see
# use these variables WITH quotations:    'far red'    or     'DAPI'
# can work with any actual channel name
channel = 'far red'

# Whether or not to show a SINGLE user specified file at a time
# still subject to above settings
singleImage = False

# if singleImage is True, only process the file given below
user_file = 'gl22-6dpi-3R-#12-sxn3P002.nd2'

# -------------------
# DO NOT CHANGE ANY CODE BELOW THIS LINE
# -------------------


# create folder to store cell counts
try:
    os.mkdir(cell_folder)
except FileExistsError as e:
    # print(f'{cell_folder} exists!')
    pass
    
# file = all_files[1]

if singleImage:
    all_files = [user_file]
else:
    all_files = os.listdir(folder_loc)

# loop through every file in list of files
for file in all_files:
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
        if showSteps:
            print('\nImages compressed.')

        # # select roi
        # _, mask = sf.select_roi(pcna_img, True)

        # process and count imgs
        # pcna_steps and pcna_titles will be empty lists if showSteps is selected
        pcna_labeled, pcna_steps, pcna_titles = sf.new_imp_process(pcna_img, debug=showSteps)
        dapi_labeled, dapi_steps, dapi_titles = sf.new_imp_process(dapi_img, debug=showSteps)
        if showSteps:
            print('\nApplying filters...')

        # perform AND combination
        # result_steps and result_titles will be empty lists if showSteps is selected
        result_labeled, result_steps, result_titles = sf.combine_channels(pcna_labeled, dapi_labeled, debug=showSteps)
        if showSteps:
            print('\nCombining channels...')

        # get counts from each image
        pcna_count = np.max(pcna_labeled)
        dapi_count = np.max(dapi_labeled)
        result_count = np.max(result_labeled)

        # color each image and overlay with original
        if showSteps:
            print('\nColoring images and overlaying with original...')
        pcna_color = sf.create_image_overlay(pcna_labeled, pcna_img)
        dapi_color = sf.create_image_overlay(dapi_labeled, dapi_img)
        result_color = sf.create_image_overlay(result_labeled, pcna_img)

        # Output results to terminal
        print(f'\nPCNA image:   {pcna_count} cells')
        print(f'DAPI image:   {dapi_count} cells')
        print(f'Final result: {result_count} cells\n')

        # count cell sizes of the result image
        sf.get_cell_sizes(result_labeled, cell_folder + file + '-cell-counts.csv', debug=showSteps)

        if showSteps:
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
