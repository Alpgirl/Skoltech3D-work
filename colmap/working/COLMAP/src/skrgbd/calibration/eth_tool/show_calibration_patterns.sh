#!/usr/bin/env bash
PATTERNS_DIR=/home/universal/Downloads/dev.sk_robot_rgbd_data/experiments/calibration/patterns

make_fullscreen() {
  sleep 1
  for window_id in $(xdotool search --name feh); do
    echo $window_id
    xdotool key --window $window_id --delay 300 v
  done
}

# Uncomment for multiple displays
#make_fullscreen & \
feh -xg 500x500+5760+0 --zoom 100 $PATTERNS_DIR/*apriltag_0*fs.png
