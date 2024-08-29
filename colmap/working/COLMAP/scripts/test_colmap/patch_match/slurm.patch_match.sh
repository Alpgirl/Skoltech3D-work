#!/usr/bin/env bash
set -e

: ${SBATCH_ARRAY_INX:?Define job array ids}
: ${CONFIG_VERSION:?Define CONFIG_VERSION}
: ${SK3D_CAM:?Define SK3D_CAM}
: ${SK3D_LIGHT:?Define SK3D_LIGHT}

SBATCH_JOB_NAME="3ddl_sk3d.experiments.test_colmap.patch_match"
SBATCH_OUTPUT='%x@%A_%a.out'
SBATCH_ERROR='%x@%A_%a.err'
SBATCH_PARTITION='ais-gpu,gpu'
SLURM_NTASKS=1
SBATCH_GPUS_PER_TASK=1
SBATCH_CPUS_PER_TASK=1
SBATCH_MEM_PER_NODE='20G'
SBATCH_TIMELIMIT='03:00:00'

sbatch \
    --job-name="$SBATCH_JOB_NAME" \
    --output="$SBATCH_OUTPUT" \
    --error="$SBATCH_ERROR" \
    --partition="$SBATCH_PARTITION" \
    --ntasks="$SLURM_NTASKS" \
    --gpus-per-task="$SBATCH_GPUS_PER_TASK" \
    --cpus-per-task="$SBATCH_CPUS_PER_TASK" \
    --mem="$SBATCH_MEM_PER_NODE" \
    --time="$SBATCH_TIMELIMIT" \
    --array="$SBATCH_ARRAY_INX" \
    ${AFTERCORR:+"--depend=aftercorr:${AFTERCORR}"} \
    /gpfs/data/gpfs0/o.voinov/sk3d/sk3d.experiments/COLMAP/scripts/test_colmap/patch_match/sing.patch_match.sh
