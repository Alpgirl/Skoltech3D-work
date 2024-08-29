#!/bin/sh

# Tap to take the photo and wait until the photo is taken.
# The idea behind this waiting trick is that the photo is started to be compressed into JPEG when it is taken,
# which produces the following message in dmesg.
# So we clear dmesg first, then shoot, then wait until the message appears.
dmesg -C && input tap $2 $3 && (dmesg -w | grep -q 'CAMERA]INFO: hjpeg_power_on enter')

mv /data/vendor/camera/img/depth_0.bin /data/vendor/camera/img/raw_depth_$1.bin
mv /data/vendor/camera/img/confidence_0.bin /data/vendor/camera/img/ir_$1.bin
