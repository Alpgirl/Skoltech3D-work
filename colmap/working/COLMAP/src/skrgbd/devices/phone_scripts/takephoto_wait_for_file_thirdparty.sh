#!/bin/sh
dmesg -C && input keyevent 27 && (dmesg -w | grep -q -o 'CAMERA]INFO: hjpeg_power_off jpeg power off success') &

mv /data/vendor/camera/img/depth_0.bin /data/vendor/camera/img/raw_depth_$1.bin
mv /data/vendor/camera/img/confidence_0.bin /data/vendor/camera/img/ir_$1.bin

wait
