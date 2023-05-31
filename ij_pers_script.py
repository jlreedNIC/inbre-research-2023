from ij.io import OpenDialog
from ij.plugin import ZProjector
from ij.plugin import ContrastEnhancer
from ij.plugin import ChannelSplitter
from ij.plugin import GaussianBlur3D as gb
from ij.plugin.frame import RoiManager
from ij.plugin import RGBStackMerge
from ij.plugin import ImageCalculator

from ij.gui import WaitForUserDialog

from ij import IJ
from ij import ImagePlus

# -------------------------------
# @author   Jordan Reed (some code taken from original writer of pcna.py)
# @date     
# @brief    This code will run in ImageJ to separate out 2 channels of interest,
#           apply image filtering (gaussian blur, contrast enhancing, etc) and thresholding
#           to each image. A mask of one image is then applied to the other image to see 
#           where the cells overlap. Then it will run the 'Analyze Particles' command to count 
#           and measure the overlapping cells.
#
#           The limitations come with the thresholding, as the cells are not sufficiently separated out.
#
#           NOTE: ImageJ does not accept comments on the first line.
# -------------------------------



def create_composite(imp, channel_indeces = None):
    """Creates a composite image of two (or more) images. Takes either a list of the images to combine, or an image stack with the channel indeces to combine.

    :param ImagePlus imp: Either a single image or a list of images to combine
    :param list of integers channel_indeces: If provided, list of channels to combine for composite image, defaults to None
    :return ImagePlus: composite image
    """
    if channel_indeces == None and type(imp) != type([]):
        print("if no indeces are given, imp must be a list of images in their separate channels")
        return
    
    if channel_indeces != None:
        # split channels of given image and add each specified channel to a list of images
        channels = ChannelSplitter.split(imp)
        
        img_list = []
        for i in channel_indeces:
            img_list.append(channels[i])
    else:
        img_list = imp

    # merge each image into a single image
    new_img = RGBStackMerge.mergeChannels(img_list, False)
    
    new_img.show()
    
    return new_img

def apply_mask(img, imask):
    """Applies a binary mask of one image to the other image and returns the result

    :param ImagePlus img: Image to apply mask to
    :param ImagePlus imask: Image to create mask out of
    :return ImagePlus: image with binary mask applied
    """
    mask = imask.getProcessor()
    target = img.getProcessor()
    target.setValue(0)
    target.fill(mask)
    return img

def apply_img_filtering(imp):
    """Applies some image filtering in an effort to make cells stand out better.
    Enhance contrast, apply gaussian blur, apply unsharp mask

    :param ImagePlus imp: Image to apply filtering to
    :return ImagePlus: Filtered image
    """
    IJ.run(imp, "Enhance Contrast", "saturated=0.35")
    IJ.run(imp, "Gaussian Blur...", "sigma=3")
    IJ.run(imp, "Unsharp Mask...", "radius=4 mask=0.60")

    return imp

def process_image():
    """Original function from pcna.py that will process a single image with 'Analyze Particles' and get an estimate of counts.
    """
    imp = IJ.getImage() # gets active image
    imp2 = ZProjector.run(imp,"max") # compress all z into one image (for now)
    imp.close() # close orig img
    channels = ChannelSplitter.split(imp2)
    imp2.close()
    imp3 = channels[3] # gets specific channel
    ContrastEnhancer().equalize(imp3) # enhances contrast
    gb().blur(imp3, 1.5,1.5,1.5) # applies gaussian blur
    imp3.show()
    
    rm = RoiManager().getInstance()
    WaitForUserDialog("Create ROI's, then click Add.").show()
    IJ.setAutoThreshold(imp3, "Default dark") # what is this doing
    IJ.run("Threshold...")
    
    WaitForUserDialog("Set threshold, then click apply.").show()
    
    # following function will tell the ROI manager to select the first ROI in the list
    rm.select(0)

    IJ.run(imp3, "Fill Holes", "")
    IJ.run(imp3, "Watershed", "")    
    IJ.run(imp3, "Analyze Particles...", "size=5-Infinity show=Outlines display clear include summarize");
    WaitForUserDialog("Count your cells, then click OK.").show()
    #rm.runCommand(imp2,"Delete")
    IJ.selectWindow("Results")
    IJ.run("Close")

    # close roi manager
    rm.close()

    IJ.selectWindow("Summary")
    IJ.run("Close")
    imp3.close()

def process_composite_img():
    """Function to process 2 channels of a given image to see where the overlap on cells is and count using 'Analyze Particles'. Gives approximation.
    """
    imp = IJ.getImage() # gets active image

    # imp2 = ZProjector.run(imp,"max") # compress all z into one image (for now)
    
    # imp.show()
    channels = ChannelSplitter.split(imp)
    imp.close() # close orig img
    
#    imp3 = create_composite(imp2, [2,3])
    # imp2.close()
    
    # get channels of interest
    ch2imp = channels[2]
    ch3imp = channels[3]
    
    # compress all z in each channel into one image (for now)
    ch2imp = ZProjector.run(ch2imp, "max")
    ch3imp = ZProjector.run(ch3imp, "max")

    ch2imp = apply_img_filtering(ch2imp)
    ch3imp = apply_img_filtering(ch3imp)
    
    ch2imp.show()
    ch3imp.show()
    
    rm = RoiManager().getInstance()
    WaitForUserDialog("Create ROI on one image, then click Add.").show()
    IJ.setAutoThreshold(ch2imp, "Default dark") 
    IJ.setAutoThreshold(ch3imp, "Default dark") 
    IJ.run("Threshold...")
    
    WaitForUserDialog("Set threshold for each image, then click apply.").show()
    
    new_img = apply_mask(ch2imp, ch3imp)
    
    new_img.updateImage()
    
    # select ROI and apply to masked image
    rm.select(new_img, 0)

    IJ.run(new_img, "Fill Holes", "")
    IJ.run(new_img, "Watershed", "") # approximation to separate out cells from blobs
    IJ.run(new_img, "Analyze Particles...", "size=5-Infinity show=Outlines display clear include summarize")
    
    WaitForUserDialog("Count your cells, then click OK to close.").show()
    IJ.selectWindow("Results")
    IJ.run("Close")

    # close roi manager
    rm.close()

    IJ.selectWindow("Summary")
    IJ.run("Close")
    new_img.close()
    ch3imp.close()
    
    

# ------------- main ------------------------------

od = OpenDialog("Images", "")

# following opens a single file at a time
fileName = od.getFileName()
directory = od.getDirectory()
IJ.run("Bio-Formats Importer", "open=[" + directory + fileName + "] autoscale color_mode=Default view=Hyperstack stack_order=XYCZT series_list=")
print(fileName)

process_composite_img()

# following will open all images in the directory in alphabetical order

#firstDir = od.getDirectory()
#fileList = os.listdir(firstDir)
#
#if "DisplaySettings.json" in fileList:
#    fileList.remove("DisplaySettings.json")
#if ".DS_Store" in fileList:  
#    fileList.remove(".DS_Store")  
#if "comments.txt" in fileList:
#    fileList.remove("comments.txt")
##print(firstDir + fileList[0])
#
#fileList.sort()
#for fileName in fileList:
#    IJ.run("Collect Garbage")
#    currentFile = directory + fileName
#    IJ.run("Bio-Formats Importer", "open=[" + currentFile + "] autoscale color_mode=Default view=Hyperstack stack_order=XYCZT series_list=")
#    process_composite_img()