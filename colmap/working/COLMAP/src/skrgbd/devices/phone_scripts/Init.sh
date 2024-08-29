#!/bin/bash

adb wait-for-device
adb shell mount -o rw,remount /odm
adb shell mount -o rw,remount /vendor
# preparing ToF sensor
adb shell setprop persist.vendor.camera.itof.mode 1
# setting dump preferences
adb shell setprop vendor.camera.itofraw.dump 1
adb shell setprop vendor.camera.itofraw.dump.count 1
adb shell setprop vendor.camera.itofraw.dump.time 0
adb shell setprop vendor.camera.itofraw.dump.temperature 1
adb shell setprop vendor.camera.itofresult.dump 1
adb shell setprop vendor.camera.itofresult.dump.count 1
adb shell setprop vendor.camera.itofresult.dump.time 0

# cleaning img folder
adb shell rm -r /data/vendor/camera/img
adb shell mkdir /data/vendor/camera/img

# additional ToF preferences
adb shell setprop vendor.disable.tof.check 1

adb shell setprop vendor.camera.moca.onivp true 

adb shell setprop vendor.camera.moca.depth2rgb 1

adb devices
