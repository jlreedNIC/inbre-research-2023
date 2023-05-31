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

def create_composite(imp, channel_indeces = None):
    """
    channels is an array of ints representing channels to merge
    imp is image plus object representing main image
    """
    if channel_indeces == None and type(imp) != type([]):
        print("if no indeces are given, imp must be a list of images in their separate channels")
        return
    
    if channel_indeces != None:
        channels = ChannelSplitter.split(imp)
        
        img_list = []
        for i in channel_indeces:
            img_list.append(channels[i])
    else:
        img_list = imp

    new_img = RGBStackMerge.mergeChannels(img_list, False)
    
    new_img.show()
    
    return new_img

def apply_mask(img, imask):
#    masked_img = ImageCalculator.run(img, imask, "multiply")
#    
#    if masked_img == None:
#        return img
#    else:
#        return masked_img
    mask = imask.getProcessor()
    target = img.getProcessor()
    target.setValue(0)
    target.fill(mask)
    return img

def process_image():
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
    # ---------

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
    imp = IJ.getImage() # gets active image

    imp2 = ZProjector.run(imp,"max") # compress all z into one image (for now)
    
    imp2.show()
    channels = ChannelSplitter.split(imp)
    imp.close() # close orig img
    
#    imp3 = create_composite(imp2, [2,3])
    imp2.close()
    
    ch2imp = channels[2]
    ch3imp = channels[3]
    
    ch2imp = ZProjector.run(ch2imp, "max")
    ch3imp = ZProjector.run(ch3imp, "max")

    IJ.run(ch2imp, "Enhance Contrast", "saturated=0.35")
    IJ.run(ch2imp, "Gaussian Blur...", "sigma=3")
    IJ.run(ch2imp, "Unsharp Mask...", "radius=4 mask=0.60")
    
    IJ.run(ch3imp, "Enhance Contrast", "saturated=0.35")
    IJ.run(ch3imp, "Gaussian Blur...", "sigma=3")
    IJ.run(ch3imp, "Unsharp Mask...", "radius=4 mask=0.60")
    
    ch2imp.show()
    ch3imp.show()
    
    rm = RoiManager().getInstance()
    WaitForUserDialog("Create ROI's, then click Add.").show()
    IJ.setAutoThreshold(ch2imp, "Default dark") # what is this doing
    IJ.setAutoThreshold(ch3imp, "Default dark") # what is this doing
    IJ.run("Threshold...")
    
    WaitForUserDialog("Set threshold, then click apply. Create ROI for channel 3 first").show()
    
    
#    new_img = create_composite([ch2imp, ch3imp])
    new_img = apply_mask(ch2imp, ch3imp)
#    new_img.show()
    try:
        new_img.updateImage()
    except:
        print("no new img to show")
    else:
        new_img.show()
    
    # following function will tell the ROI manager to select the first ROI in the list
    rm.select(0)
    # ---------

    IJ.run(new_img, "Fill Holes", "")
    IJ.run(new_img, "Watershed", "")
    IJ.run(new_img, "Analyze Particles...", "size=5-Infinity show=Outlines display clear include summarize")
    
#    IJ.run(ch3imp, "Fill Holes", "")
#    IJ.run(ch3imp, "Watershed", "")
#    IJ.run(ch3imp, "Analyze Particles...", "size=5-Infinity show=Outlines display clear include summarize")
    
    
    WaitForUserDialog("Count your cells, then click OK.").show()
    #rm.runCommand(imp2,"Delete")
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
print("Working with", fileName)


process_composite_img()

#imp = O.openImage(directory,fileName)
#print(imp)
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
#    currentFile = firstDir + fileName
#    #print(firstDir)
#    #print(currentFile)
#    IJ.run("Bio-Formats Importer", "open=[" + currentFile + "] autoscale color_mode=Default view=Hyperstack stack_order=XYCZT series_list=")
#    PCNA_semi_automated()