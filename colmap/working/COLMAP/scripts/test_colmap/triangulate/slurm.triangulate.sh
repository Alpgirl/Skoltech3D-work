#!/usr/bin/env bash
set -e

: ${SBATCH_ARRAY_INX:?Define job array ids}
: ${CONFIG_VERSION:?Define CONFIG_VERSION}
: ${SK3D_CAM:?Define SK3D_CAM}
: ${SK3D_LIGHT:?Define SK3D_LIGHT}

SBATCH_JOB_NAME="3ddl_sk3d.experiments.test_colmap.triangulate"
SBATCH_OUTPUT='%x@%A_%a.out'
SBATCH_ERROR='%x@%A_%a.err'
SBATCH_PARTITION='ais-htc,htc'
SLURM_NTASKS=1
SBATCH_CPUS_PER_TASK=4
SBATCH_MEM_PER_NODE='5G'
SBATCH_TIMELIMIT='00:10:00'

sbatch \
    --job-name="$SBATCH_JOB_NAME" \
    --output="$SBATCH_OUTPUT" \
    --error="$SBATCH_ERROR" \
    --partition="$SBATCH_PARTITION" \
    --ntasks="$SLURM_NTASKS" \
    --cpus-per-task="$SBATCH_CPUS_PER_TASK" \
    --mem="$SBATCH_MEM_PER_NODE" \
    --time="$SBATCH_TIMELIMIT" \
    --array="$SBATCH_ARRAY_INX" \
    ${AFTERCORR:+"--depend=aftercorr:${AFTERCORR}"} \
    /gpfs/data/gpfs0/o.voinov/sk3d/sk3d.experiments/COLMAP/scripts/test_colmap/triangulate/sing.triangulate.sh
