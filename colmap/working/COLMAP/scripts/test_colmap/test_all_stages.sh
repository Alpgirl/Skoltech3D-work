#!/usr/bin/env bash
set -e

# define env variables
export CONFIG_VERSION='v3.0.0'
export SK3D_CAM='tis_right'
export SK3D_LIGHT='ambient@best'
export SCENE_NAME='plush_bear'

# : ${SBATCH_ARRAY_INX:?Define job array ids}
: ${CONFIG_VERSION:?Define CONFIG_VERSION}
: ${SK3D_CAM:?Define SK3D_CAM}
: ${SK3D_LIGHT:?Define SK3D_LIGHT}

STAGE_JOB_ID=''
for stage in extract_feats match_feats triangulate; do # patch_match fuse; do
    # mkdir -p $stage
    # cd $stage
    cd /app/working/COLMAP/scripts/test_colmap/${stage}
    STAGE_JOB_ID=$(AFTERCORR=$STAGE_JOB_ID bash ${stage}.sh)
    STAGE_JOB_ID=${STAGE_JOB_ID##* }
    echo $stage $STAGE_JOB_ID
    cd /app/working/COLMAP
    sleep 1
done

chmod 777 /app/working/location/colmap/${CONFIG_VERSION}/${SK3D_CAM}/${SK3D_LIGHT}/${SCENE_NAME}/subsystem/sparse
chmod 777 /app/working/location/colmap/${CONFIG_VERSION}/${SK3D_CAM}/${SK3D_LIGHT}/${SCENE_NAME}/subsystem/sparse/0
