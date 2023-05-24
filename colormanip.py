# ------------
# @author   Jordan Reed
# @date     5/22/23
# @brief    This file is to test getting the right color variations out of the nd2 files
#
#           This was tested in Spyder IDE in order to see the pictures 
#           alongside the code. If executing in another IDE, changes should be
#           made to save the pictures shown one after another so the images 
#           are not lost.
# ------------


try:
    from nd2reader import ND2Reader
    import matplotlib.pyplot as plt
    import numpy as np
    import datetime as dt
except Exception as e:
    print('issue with import')
    print('You may need to install importlib package if issue with nd2reader.')
    print(e)

import custom_color_map as ccm

file_names = [
    'nd2_files/SciH-Whole-Ret-4C4-Redd-GFP-DAPI005.nd2', # 4gb
    'nd2_files/Undamaged-structual-example.nd2', # 58mb 
    'nd2_files/S2-6dpi-uoi2506Tg-4R-#13-sxn2003.nd2', #40mb
    'nd2_files/S2-6dpi-uoi2506Tg-1R-#13-sxn2002.nd2' # 70mb
]

def find_max(table):
    maxes = []
    for i in range(len(table)):
        val = np.max(table[i])
        maxes.append(val)
    val = np.max(maxes)

    return val

def turn_into_percentage(table, maxval):
    new_table = np.zeros(table.shape)
    for i in range(0, len(table)):
        for j in range(0, len(table[0])):
            new_table[i][j] = float(table[i][j]) / float(maxval)
    
    return new_table

def perc_to_rgb(table, rgb_val):
    new_img = np.zeros((table.shape[0], table.shape[1], 3))
    for i in range(0, len(table)):
        for j in range(0, len(table[0])):
            # new_rgb = [0,0,0]
            # for k in range(3):
            #     new_rgb[k] = table[i][j]*rgb_val[k]
            new_img[i][j] = np.asarray(rgb_val) * table[i][j]
    
    return new_img

color_channels = {
    'green': (0.0, 1.0, 0.0),
    'blue': (0.0, 0.0, 1.0),
    'red': (1.0, 0.0, 0.0),
    'pink': (1.0, 0.75, 0.80)
}

def show_image_using_scalar(img, color_channel=ccm.green_channel):
    maxval = np.max(img)
    plt.imshow(img,vmax=maxval, cmap=color_channel)
    plt.show()

# open nd2 file
with ND2Reader(file_names[3]) as sample_image:

    # iterate over all channels and entire z stack
    num_channels = sample_image.sizes['c']
    img_per_channel = sample_image.sizes['z']
    sample_image.iter_axes = 'cz'
    print(f'total number of images to show: {sample_image.shape[0]}')
    print(f'total channels: {num_channels}')
    print(f'images per channel: {img_per_channel}')
    print(f'size of images: {sample_image.shape[1]} x {sample_image.shape[2]}')

    start = dt.datetime.now()
    # following code will output a picture using different color maps and the imshow function
    color_map_list = [ccm.green_channel, ccm.red_channel, ccm.purple_channel, ccm.blue_channel]
    cindex = 0
    for i in range(0, sample_image.shape[0]):
        if i != 0 and i%img_per_channel==0:
            cindex += 1
        
        # maxval = np.max(sample_image[i])
        # plt.imshow(sample_image[i],vmax=maxval, cmap=color_map_list[cindex])
        # plt.show()
        show_image_using_scalar(sample_image[i], color_map_list[cindex])
    stop = dt.datetime.now()
    print(f'time taken for imshow with colormap: {stop-start}')

    # maxval = np.max(sample_image[2])
    # plt.imshow(sample_image[2],vmax=maxval, cmap=ccm.green_channel)
    # plt.show()

    # start = dt.datetime.now()
    # color_options = ['green', 'red', 'blue', 'pink']
    # ci = 0
    # # apply different custom channel to each set of 5 images
    # for i in range(sample_image.shape[0]):
    #     if i != 0 and i%img_per_channel == 0:
    #         ci += 1
    #     maxval = np.max(sample_image[i])
    #     perc_img = turn_into_percentage(sample_image[i], maxval)
    #     new_img = perc_to_rgb(perc_img, color_channels[color_options[ci]])
    #     plt.imshow(new_img)
    #     plt.show()
    # stop = dt.datetime.now()
    # print(f'time taken for manual color adjustment: {stop-start}')


    
