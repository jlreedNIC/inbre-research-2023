# ------------
# @author   Jordan Reed
# @date     5/18/23
# @brief    This file is just to test how to open and display an nd2 file
#
#           This was tested in Spyder IDE in order to see the pictures 
#           alongside the code. If executing in another IDE, changes should be
#           made to save the pictures shown one after another so the images 
#           are not lost.
# ------------


try:
    from nd2reader import ND2Reader
except Exception as e:
    print('issue with nd2reader import')
    print('You may need to install importlib package.')
    print(e)

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print('issue with matplotlib import')
    print('may need to install matplotlib package')
    print(e)

try:
    import numpy as np
except Exception as e:
    print('issue with numpy import')
    print('may need to install numpy package')
    print(e)

file_names = [
    'nd2_files/SciH-Whole-Ret-4C4-Redd-GFP-DAPI005.nd2', # 4gb
    'nd2_files/Undamaged-structual-example.nd2', # 58mb
    'nd2_files/S2-6dpi-uoi2506Tg-4R-#13-sxn2003.nd2' #40mb
]

# open nd2 file
with ND2Reader(file_names[1]) as sample_image:
    print(sample_image)
    
    print(f'\nnd2 file metadata: \n {sample_image.metadata}')
    print(f'\naxes to iterate over: \n {sample_image.sizes}')
    
    # iterate over all channels and entire z stack
    sample_image.iter_axes = 'cz'
    print(f'total number of images to show: {sample_image.shape[0]}')
    print(f'size of images: {sample_image.shape[1]} x {sample_image.shape[2]}')
    
    # create new array of size [z, channels, img pixels (2)]
    # new_img_arr = np.zeros((sample_image.sizes['z'], sample_image.sizes['c'], sample_image.shape[1], sample_image.shape[2]))
    
    # show all images
    for i in range(sample_image.shape[0]):  
        """ next 2 lines are for putting all images into a single subplot
        NOTE: this makes the images hard to see
        num rows = num images along z axis
        num cols = num img per channel
        this has trouble running with bigger images and more images
        """
        # plt.subplot(sample_image.sizes['z'], sample_image.sizes['c'], (i+1))        
        # plt.imshow(sample_image[i])
        
        # following code will output images one after the other and save it to file
        plt.imshow(sample_image[i])
        # plt.savefig(f'images/whole_retina_img{i}.png')
        plt.show()
        # plt.close()
        
        # this code will put images into another array of size [z, channels, img height, img width]
        # x = int(i/2)
        # y = i%2
        # new_img_arr[x][y] = sample_image[i]
    
# print(new_img_arr.shape)
        

'''
This file will successfully open the nd2 file and show the images requested.
Program will also save images and move data to another array
'''