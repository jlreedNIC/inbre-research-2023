# ------------
# @author   Jordan Reed
# @date     5/23/23
# @brief    This file is to test creating a custom color map for matplotlib.plotly.imshow()
#
#           https://matplotlib.org/stable/gallery/color/custom_cmap.html
#           Documentation on creating color map for showing images.
# ------------

import matplotlib.colors as mcolors

'''
Creates a dictionary that specifies the intensity for each rgb value 
in a linear fashion. For example, red stays at 0. Green goes from 
0% to 100%. Blue stays at 0.
'''
green_dict = {
    'red': (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0)
    ),
    'green': (
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0)
    ),
    'blue': (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0)
    )
}

red_dict = {
    'red': (
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0)
    ),
    'green': (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0)
    ),
    'blue': (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0)
    )
}

blue_dict = {
    'red': (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0)
    ),
    'green': (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0)
    ),
    'blue': (
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0)
    )
}

pink_dict = {
    'red': (
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0)
    ),
    'green': (
        (0.0, 0.0, 0.0),
        (1.0, 0.6, 0.6)
    ),
    'blue': (
        (0.0, 0.0, 0.0),
        (1.0, 0.8, 0.8)
    )
}

'''
Creates an actual color map that imshow() understands for each dict specified above.
'''
green_channel = mcolors.LinearSegmentedColormap('green_channel', green_dict)
red_channel = mcolors.LinearSegmentedColormap('red_channel', red_dict)
blue_channel = mcolors.LinearSegmentedColormap('blue_channel', blue_dict)
pink_channel = mcolors.LinearSegmentedColormap('pink_channel', pink_dict)