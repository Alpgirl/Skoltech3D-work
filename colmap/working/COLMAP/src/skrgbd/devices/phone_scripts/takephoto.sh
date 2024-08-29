#!/bin/sh
# Swipe gesture to wake up camera
input swipe 500 2000 500 1000

input keyevent 27  # shoot

mv /data/vendor/camera/img/depth_0.bin /data/vendor/camera/img/raw_depth_$1.bin
mv /data/vendor/camera/img/confidence_0.bin /data/vendor/camera/img/ir_$1.bin
