#!/usr/bin/env bash
set -ex

source /gpfs/data/gpfs0/o.voinov/sk3d/sk3d.experiments/COLMAP/env/utils.sh
run_singularity << EOF
source /experiments_code/COLMAP/scripts/test_colmap/fuse/fuse.sh
EOF
