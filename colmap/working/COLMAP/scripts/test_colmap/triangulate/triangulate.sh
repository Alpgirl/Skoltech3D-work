#!/usr/bin/env bash
set -ex

# : ${SLURM_ARRAY_TASK_ID:?Define SLURM_ARRAY_TASK_ID}
: ${CONFIG_VERSION:?Define CONFIG_VERSION}
: ${SK3D_CAM:?Define SK3D_CAM}
: ${SK3D_LIGHT:?Define SK3D_LIGHT}

# source /experiments_code/COLMAP/env/utils.sh
# activate_env
# export PYTHONPATH=/code/src:/experiments_code/COLMAP/src
cd /app/working/COLMAP/src

# SCENE_NAME=$(python /code/src/skrgbd/data/processing/processing_pipeline/get_scene_name.py \
#     --scene-i $SLURM_ARRAY_TASK_ID)

python test_on_sk3d.py \
    --stage triangulate \
    --version "$CONFIG_VERSION" \
    --cam "$SK3D_CAM" \
    --light "$SK3D_LIGHT" \
    --data-dir "/app/sk3d/dataset" \
    --results-dir "/app/working/location" \
    --scene-name "$SCENE_NAME" \
    --colmap-bin $(which colmap) 
    # --threads-n $SLURM_CPUS_PER_TASK
