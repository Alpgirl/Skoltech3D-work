for scan_dir in $@; do
  printf '*** Check %s\n' $scan_dir
  cd $scan_dir
  if [[ $(find . -path './tis_left/*.png' | wc -l) != 1400 ]]; then echo 'Lost tis_left RGB'; fi
  if [[ $(find . -path './tis_right/*.png' | wc -l) != 1400 ]]; then echo 'Lost tis_right RGB'; fi
  if [[ $(find . -path './phone_left/*.jpg' | wc -l) != 1400 ]]; then echo 'Lost phone_left RGB'; fi
  if [[ $(find . -path './phone_left/*_ir.png' | wc -l) != 100 ]]; then echo 'Lost phone_left IR'; fi
  if [[ $(find . -path './phone_left/*_depth.png' | wc -l) != 100 ]]; then echo 'Lost phone_left depth'; fi
  if [[ $(find . -path './phone_right/*.jpg' | wc -l) != 1400 ]]; then echo 'Lost phone_right RGB'; fi
  if [[ $(find . -path './phone_right/*_ir.png' | wc -l) != 100 ]]; then echo 'Lost phone_right IR'; fi
  if [[ $(find . -path './phone_right/*_depth.png' | wc -l) != 100 ]]; then echo 'Lost phone_right depth'; fi
  if [[ $(find . -path './real_sense/*.png' \! -path './real_sense/*_depth.png' \! -path './real_sense/*_ir.png' \! -path './real_sense/*_irr.png' | wc -l) != 1400 ]]; then echo 'Lost real_sense RGB'; fi
  if [[ $(find . -path './real_sense/*_ir.png' | wc -l) != 1400 ]]; then echo 'Lost real_sense IR'; fi
  if [[ $(find . -path './real_sense/*_irr.png' | wc -l) != 1400 ]]; then echo 'Lost real_sense IR right'; fi
  if [[ $(find . -path './real_sense/*_depth.png' | wc -l) != 1400 ]]; then echo 'Lost real_sense depth'; fi
  if [[ $(find . -path './kinect_v2/*.png' \! -path './kinect_v2/*_depth.png' \! -path './kinect_v2/*_ir.png' | wc -l) != 1300 ]]; then echo 'Lost kinect_v2 RGB'; fi
  if [[ $(find . -path './kinect_v2/*_ir.png' | wc -l) != 100 ]]; then echo 'Lost kinect_v2 IR'; fi
  if [[ $(find . -path './kinect_v2/*_depth.png' | wc -l) != 100 ]]; then echo 'Lost kinect_v2 depth'; fi
  cd ..
done
