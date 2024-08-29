#!/bin/bash
# Not sure why there is remount at the beginning of each script, kept it as is 
# adb remount put /system partition in writable mode. By default /system is only readable. It could only be done on rooted device.
adb remount

# Cleaning folders on device
adb shell rm -rf data/vendor/camera/img/*
adb shell rm -rf data/log/android_logs/*
adb shell rm -rf data/vendor/log/isp-log/*
adb shell rm -rf /sdcard/DCIM/Camera/*

adb shell rm -r data/vendor/camera/img/
adb shell mkdir data/vendor/camera/img/