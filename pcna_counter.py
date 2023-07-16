# ------------
# @author   Jordan Reed
# @date     6/7/23
# @brief    Program to find and count cells on given channels using edge 
#           detection, thresholding, and morphological techniques.
# ------------

import seg_functions as sf
import numpy as np
import os
from roi_class import ROI

# -------------------------
# user changeable variables located in config.yaml
# DO NOT CHANGE THIS CODE (unless you know what you're doing)
# -------------------------

config = sf.get_config()

# name of folder/directory that contains nd2 files for analyzing
folder_loc = config['folder_loc']

# name of folder/directory to create to hold all .csv files that have the sizes of cells
cell_size_folder = config['cell_size_folder']

# Whether or not to show the intermediary steps in cell count process
# this is the default when looking at a single channel
showSteps = config['showSteps']

# only show a single channel
singleChannel = config['singleChannel']

# which channel numbers to look at
channel_nums = config['channel_num']

# if single channel is True, which channel do you want to see
channel = config['channel']

# Whether or not to show a SINGLE user-specified file at a time
# still subject to above settings
singleImage = config['singleImage']

# if singleImage is True, only process the file given below
user_file = config['user_file']

# minimum size of cells to count
artifact_size = config['artifact_size']

# batch number of files
batch_num = config['batch_number']

# create folder to store cell counts
try:
    os.mkdir(cell_size_folder)
except FileExistsError as e:
    # print(f'{cell_size_folder} exists!')
    pass

if singleImage:
    all_files = [user_file]
else:
    all_files = os.listdir(folder_loc)

all_files.sort()

print(f'Found {len(all_files)} files. Starting process now...')
# loop through every file in list of files
for file in all_files:
    
    if singleChannel:
        if channel == None:
            print('You must change the channel name.')
            print("Example: 'far red' or 'DAPI'")
            exit(1)
        
        print(f'Opening file: - {file} -\n')
        try:
            img_stack, p_microns = sf.open_nd2file(folder_loc + file, channel_name=[channel], channel_num=channel_nums, debug=showSteps)
        except Exception as e:
            print(f'Error opening {file}.')
            print(e)

            # put file name in an output file for later review
            with open('files_wont_open.txt','a') as f:
                f.write(f'{file}\n')
            # exit(1)
            continue
        
        # rename for file conventions
        if channel == 'far red':
            channel_name = 'pcna'
        else:
            channel_name = channel
        channel_name.swapcase() # make it all capitals

        # find middle slice of image stack
        mid_slice = int(len(img_stack)/2)
        img = img_stack[mid_slice]
        print('\nMiddle slice of stack found.')

        # get roi selection
        roi = ROI(img)
        roi.get_roi()
        print('\nCreating ROI mask...')
        roi_mask = roi.create_roi_mask()

        # process and count imgs
        print('\nApplying filters...')
        if channel == 'DAPI':
            labeled, steps, titles = sf.dapi_process(img, debug=True, mask=roi_mask, artifact_size=artifact_size)
        else:
            labeled, steps, titles = sf.pcna_process(img, debug=True, mask=roi_mask, artifact_size=artifact_size)

        # get counts from each image
        count = np.max(labeled)

        # color each image and overlay with original
        print('\nColoring image and overlaying with original...')
        color = sf.create_image_overlay(labeled, img)

        # show roi lines on finished image
        roi.draw_lines_on_image(color, (255,0,0))
        steps.append(color)
        titles.append(f'Final count: {count}')

        # Output results to terminal
        print(f'\nFinal count:   {count} cells')

        # get roi size
        roi_size = roi.get_roi_size()

        # count cell sizes of the result image
        sf.get_cell_sizes(labeled, nd2_filename=file, save_filename=f'{cell_size_folder}cell-counts-batch-{batch_num}.csv', roi_pcount=roi_size, pixel_conv=p_microns, debug=True)

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
            figure_title= file[:-4] + '-' + channel_name + '-RESULTS.nd2'
        )
    else:
        # 2 channels
        print(f'Opening file: - {file} -')
        
        try:
            pcna_imgs, dapi_imgs, p_microns = sf.open_nd2file(folder_loc + file, channel_num=channel_nums, debug=showSteps)
        except Exception as e:
            print(f'Error opening {file}.')
            print(e)

            # put file name in an output file for later review
            with open('files_wont_open.txt','a') as f:
                f.write(f'{file}\n')
            # exit(1)
            continue

        # compress stacks to single img
        mid_slice = int(len(pcna_imgs)/2)
        pcna_img = pcna_imgs[mid_slice]
        dapi_img = dapi_imgs[mid_slice]

        if showSteps:
            # print('\nImages compressed.')
            print(f'\nMiddle slice found at {mid_slice}.')

        # select roi
        roi = ROI(pcna_img)
        roi.get_roi()
        roi_mask = roi.create_roi_mask()

        # process and count imgs
        # pcna_steps and pcna_titles will be empty lists if showSteps is selected
        # pcna_labeled, pcna_steps, pcna_titles = sf.new_imp_process(pcna_img, debug=showSteps, mask=roi_mask)
        pcna_labeled, pcna_steps, pcna_titles = sf.pcna_process(pcna_img, debug=showSteps, mask=roi_mask, artifact_size=artifact_size)
        dapi_labeled, dapi_steps, dapi_titles = sf.dapi_process(dapi_img, debug=showSteps, mask=roi_mask, artifact_size=artifact_size)
        if showSteps:
            print('\nApplying filters...')

        # perform AND combination
        # result_steps and result_titles will be empty lists if showSteps is selected
        result_labeled, result_steps, result_titles = sf.combine_channels(pcna_labeled, dapi_labeled, debug=showSteps, artifact_size=artifact_size)
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
        # if showSteps:
        #     pcna_steps.append(pcna_color)
        #     pcna_titles.append('Final PCNA')
        #     dapi_steps.append(dapi_color)
        #     dapi_titles.append('Final DAPI')

        # Output results to terminal
        print(f'\nPCNA image:   {pcna_count} cells')
        print(f'DAPI image:   {dapi_count} cells')
        print(f'Final result: {result_count} cells\n')

        # get roi size
        roi_size = roi.get_roi_size()

        # count cell sizes of the result image and put into file
        sf.get_cell_sizes(result_labeled, nd2_filename=file, save_filename=f'{cell_size_folder}cell-counts-batch-{batch_num}.csv', roi_pcount=roi_size,pixel_conv=p_microns, debug=showSteps)

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
        
        # put roi lines on final image
        roi.draw_lines_on_image(dapi_color, color=(255,0,0))
        roi.draw_lines_on_image(pcna_color, color=(255,0,0))
        roi.draw_lines_on_image(result_color, color=(255,0,0))
        
        # show results
        sf.use_subplots(
            [dapi_img, dapi_color, pcna_img, pcna_color, result_color],
            [ 'DAPI orig', f'DAPI: {dapi_count}', 'PCNA orig', f'PCNA: {pcna_count}',f'final: {result_count}'],
            ncols=5,
            figure_title= file[:-4] + '-RESULTS.nd2'
        )

print('\nEnd of file list.')
print('Thank you for using the PCNA-Counter!')