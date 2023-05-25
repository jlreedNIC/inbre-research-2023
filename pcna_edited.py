from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.features.spot import SpotContrastAndSNRAnalyzerFactory
import fiji.plugin.trackmate.tracking.sparselap.SparseLAPTrackerFactory as SparseLAPTrackerFactory
from fiji.plugin.trackmate.tracking.sparselap import SparseLAPTrackerFactory
#from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO
from fiji.plugin.trackmate.tracking import LAPUtils
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter
import fiji.plugin.trackmate.features.track.TrackDurationAnalyzer as TrackDurationAnalyzer
import fiji.plugin.trackmate.features.track.TrackSpotQualityFeatureAnalyzer as TrackSpotQualityFeatureAnalyzer
import fiji.plugin.trackmate.SelectionModel as SelectionModel
import fiji.plugin.trackmate.Settings as Settings
import fiji.plugin.trackmate.Model as Model
import fiji.plugin.trackmate.TrackMate as TrackMate
import fiji.plugin.trackmate.Spot as Spot
import fiji.plugin.trackmate.TrackMate
from ij.plugin import ChannelSplitter
from ij.plugin import ImageCalculator
from ij.plugin import ZProjector
from net.imglib2.img.display.imagej import ImageJFunctions
from java.awt.event import TextListener
from ij import Menus
from ij.gui import GenericDialog
from ij.io import OpenDialog
from ij.measure import ResultsTable
from ij.plugin.frame import RoiManager
from ij.gui import WaitForUserDialog
import java.util.ArrayList as ArrayList
import csv
import os
import sys
from ij import IJ
from ij import ImagePlus

def PCNA_semi_automated():
    imp = IJ.getImage()
    title = imp.getTitle()
    IJ.run(imp, "Set Scale...", "distance=3.0769 known=1 unit=micron");
    imp2 = ZProjector.run(imp,"max")
    imp.close()
    imp2.show()
    dimentions = imp2.getDimensions()
    numZ, numChannels, numframes  = dimentions[3], dimentions[2], dimentions[4]
    channels = ChannelSplitter.split(imp2)
    imp2.close()
    imp3 = channels[1]
    imp3.show()
    IJ.run(imp3, "Enhance Contrast", "saturated=0.35")
    IJ.run(imp3, "Gaussian Blur...", "sigma=3")
    IJ.run(imp3, "Unsharp Mask...", "radius=4 mask=0.60")
    rm = RoiManager()
    rm = rm.getInstance()
    WaitForUserDialog("Create ROI, then click Add.").show()
    IJ.setAutoThreshold(imp3, "Default dark")
    IJ.run("Threshold...")
    #IJ.run(imp3, "Convert to Mask", "")

    # --------
	# JReed addition
    WaitForUserDialog("Set threshold, then click apply. If you have more than one ROI, select the one you're interested in. ").show()
	
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
    

od = OpenDialog("Images", "")
firstDir = od.getDirectory()
fileList = os.listdir(firstDir)

if "DisplaySettings.json" in fileList:
    fileList.remove("DisplaySettings.json")
if ".DS_Store" in fileList:  
    fileList.remove(".DS_Store")  
if "comments.txt" in fileList:
    fileList.remove("comments.txt")
#print(firstDir + fileList[0])
fileList.sort()
for fileName in fileList:
    IJ.run("Collect Garbage")
    currentFile = firstDir + fileName
    #print(firstDir)
    #print(currentFile)
    IJ.run("Bio-Formats Importer", "open=[" + currentFile + "] autoscale color_mode=Default view=Hyperstack stack_order=XYCZT series_list=")
    PCNA_semi_automated()