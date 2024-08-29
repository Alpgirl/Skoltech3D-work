#!/bin/bash

give_hints() {
  sed -u \
    -e "s/\(RuntimeError: xioctl(VIDIOC_S_FMT) failed Last Error: Input\/output error.*\)/\1\n\n PROBLEM: RealSense could not init --> Restart scan_cameras, reconnect RealSense, start the warmup.\n/" \
    -e "s/\(RuntimeError: get_xu(id=.*) failed! Last Error: Device or resource busy.*\)/\1\n\n PROBLEM: RealSense could not init --> Restart scan_cameras, reconnect RealSense, start the warmup.\n/" \
    -e "s/\(RuntimeError: close() failed. UVC device was not opened!.*\)/\1\n\n PROBLEM: RealSense could not stop --> Restart scan_cameras, reconnect RealSense, start the warmup.\n/" \
    -e "s/\(AttributeError: 'NoneType' object has no attribute 'reset'.*\)/\1\n\n PROBLEM: A device could not init --> If everything works, ignore, otherwise: restart scan_cameras, reconnect the device, start the warmup.\n/" \
    -e "s/\(AttributeError: 'NoneType' object has no attribute 'stop'.*\)/\1\n\n PROBLEM: A device could not stop --> If everything works, ignore, otherwise: restart scan_cameras, reconnect the device, start the warmup.\n/" \
    -e 's/\(Robot not running:.*\)/\1\n\n PROBLEM: Cannot connect to the robot --> If you are going to work with the robot check the control panel.\n/' \
    -e 's/\(RuntimeError: real_sense not responding.*\)/\1\n\n PROBLEM: RealSense froze --> Restart scan_cameras, reconnect RealSense, start the warmup.\n/' \
    -e 's/\(RuntimeError: kinect_v2 not responding.*\)/\1\n\n PROBLEM: Kinect froze --> Restart scan_cameras, start the warmup.\n/' \
    -e 's/\(RuntimeError: tis_left not responding.*\)/\1\n\n PROBLEM: Left TIS froze --> Restart scan_cameras, start the warmup.\n/' \
    -e 's/\(RuntimeError: tis_right not responding.*\)/\1\n\n PROBLEM: Right TIS froze --> Restart scan_cameras, start the warmup.\n/' \
    -e 's/\(RuntimeError: phone_left not responding.*\)/\1\n\n PROBLEM: Left Phone froze --> Restart scan_cameras, start the warmup.\n/' \
    -e 's/\(RuntimeError: phone_right not responding.*\)/\1\n\n PROBLEM: Right Phone froze --> Restart scan_cameras, start the warmup.\n/' \
    -e 's/\(File "\/home\/universal\/anaconda3\/envs\/py38_dev\/lib\/python3.8\/site-packages\/ppadb\/connection.py", line 63, in _recv.*\)/\1\n\n PROBLEM: Phone connectivity problem --> React depending on the situation.\n/' \
    -e 's/\(ValueError: zero-size array to reduction operation minimum which has no identity.*\)/\1\n\n PROBLEM: Most likely you have messed up the mask during the picking of the camera settings.\n/'
}

colorize() {
  colout '(DEBUG)|(INFO)|(STDERR|ERROR|Error)|(WARNING|Warning)|(PROBLEM)' "blue,green,red,yellow,cyan"
}

DATE=${2:-$(date +%y_%m_%d)}
cameras_log=~/Downloads/dev.sk_robot_rgbd_data/experiments/logs/scanning/${DATE}_cameras.log
stl_log=~/Downloads/dev.sk_robot_rgbd_data/experiments/logs/scanning/${DATE}_stl.log
phone_data_log=~/Downloads/dev.sk_robot_rgbd_data/experiments/logs/scanning/${DATE}_phone_data.log
calib_log=~/Downloads/dev.sk_robot_rgbd_data/experiments/calibration/logs/${DATE}.log


if [ -z ${2+x} ]; then
  trap 'kill $(jobs -p)' EXIT

  case "$1" in
   "all" )
   tail -F $cameras_log | colorize &
   tail -F $stl_log | colorize &
   tail -F $phone_data_log | colorize &
   tail -F $calib_log | colorize &
   ;;

   "info" )
   tail -F $cameras_log | grep --line-buffered INFO | colorize &
   tail -F $stl_log | grep --line-buffered INFO | colorize &
   tail -F $phone_data_log | grep --line-buffered INFO | colorize &
   ;;

   "err" )
   pattern="STDERR\|ERROR\|WARNING"
   tail -F $cameras_log | grep --line-buffered "$pattern" | give_hints | colorize &
   tail -F $stl_log | grep --line-buffered "$pattern" | give_hints  | colorize &
   tail -F $phone_data_log | grep --line-buffered "$pattern" | give_hints  | colorize &
   ;;
  esac

  wait
else
  case "$1" in
   "all" )
   cat $cameras_log $stl_log $phone_data_log $calib_log | sort | colorize
   ;;

   "info" )
   cat $cameras_log $stl_log $phone_data_log | grep INFO | sort | colorize
   ;;

   "err" )
   pattern="STDERR\|ERROR\|WARNING"
   cat $cameras_log $stl_log $phone_data_log | grep "$pattern" | sort | give_hints  | colorize
   ;;
  esac
fi
