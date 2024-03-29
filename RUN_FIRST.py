# ------------
# @author   Jordan Reed
# @date     6/15/23
# @brief    File to run first to let user know which libraries need to be installed
#
# ------------

everythingInstalled = True

print("Testing imports...")

try:
    from nd2reader import ND2Reader
except ModuleNotFoundError as e:
    print(f'\n{e}')
    print('Please type: pip3 install nd2reader')
    everythingInstalled = False

try:
    import numpy as np
except ModuleNotFoundError as e:
    print(f'\n{e}')
    print('Please type: pip3 install numpy')
    everythingInstalled = False

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    print(f'\n{e}')
    print('Please type: pip3 install matplotlib')
    everythingInstalled = False

try:
    import cv2
except ModuleNotFoundError as e:
    print(f'\n{e}')
    print('Please type: pip3 install opencv-python')
    everythingInstalled = False

try:
    import datetime as dt
except ModuleNotFoundError as e:
    print(f'\n{e}')
    print('Please type: pip3 install DateTime')
    everythingInstalled = False

try:
    from skimage import filters, feature, color, measure, morphology
except ModuleNotFoundError as e:
    print(f'\n{e}')
    print('Please type: pip3 install scikit-image')
    everythingInstalled = False

try:
    import yaml
except ModuleNotFoundError as e:
    print(f'\n{e}')
    print('Please type: pip3 install pyyaml')
    everythingInstalled = False

try:
    from matplotlib_scalebar import scalebar
except ModuleNotFoundError as e:
    print(f'\n{e}')
    print('Please type: pip3 install matplotlib-scalebar')
    everythingInstalled = False

if everythingInstalled:
    print("\nYou are good to start the program! Type: python3 pcna_counter.py")
    exit(0)
else:
    print("\nPlease install the required modules and try again.")
    exit(1)
