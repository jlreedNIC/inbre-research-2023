from ij.io import OpenDialog
from ij.plugin import ZProjector
from ij.plugin import ContrastEnhancer
from ij.plugin import ChannelSplitter
from ij.plugin import GaussianBlur3D as gb
from ij.plugin.frame import RoiManager
from ij.plugin import RGBStackMerge

from ij.gui import WaitForUserDialog

from ij import IJ
from ij import ImagePlus

def create_composite(imp, channel_indeces = [2,3]):
    """
    channels is an array of ints representing channels to merge
    imp is image plus object representing main image
    """
#    print('splitting channels up')
    channels = ChannelSplitter.split(imp)
    
    img_list = []
#    print('putting channels into list')
    for i in channel_indeces:
        img_list.append(channels[i])
    
#    print('now merging channels')
    new_img = RGBStackMerge.mergeChannels(img_list, False)
    
#    print('showing merged img')
    new_img.show()
    
    return new_img

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
    imp.close() # close orig img
    
    imp3 = create_composite(imp2, [2,3])

    imp2.close()

    IJ.run(imp3, "Enhance Contrast", "saturated=0.35")
    IJ.run(imp3, "Gaussian Blur...", "sigma=3")
    IJ.run(imp3, "Unsharp Mask...", "radius=4 mask=0.60")
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
    
    

# ------------- main ------------------------------

od = OpenDialog("Images", "")
fileName = od.getFileName()
directory = od.getDirectory()
IJ.run("Bio-Formats Importer", "open=[" + directory + fileName + "] autoscale color_mode=Default view=Hyperstack stack_order=XYCZT series_list=")
print(fileName)

#process_image()

process_composite_img()
# open image
# apply auto contrast and brightness
# apply smoothing and/or blur

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