#!/bin/bash
# Not sure why there is remount at the beginning of each script, kept it as is 
adb remount

# Creating the folder with current date/time per each shot (folder is created on PC). Each time 2 subfolders will be copied from device - with RGB picture and with Tof raw data (bin files)
datasetfolder=`date +%Y.%m.%d.%T`
mkdir $datasetfolder

# Copy files from device to PC
adb pull /data/vendor/camera/img/ $datasetfolder
adb pull /sdcard/DCIM/Camera/ $datasetfolder
