#!/usr/bin/env bash

CONDA_ROOT='/opt/miniconda3'
ENV_ROOT="${CONDA_ROOT}/envs/py38"
CODE_ROOT='/code'
EXP_CODE_ROOT='/experiments_code'
DATA_ROOT='/data'
RESULTS_ROOT='/results/experiments/colmap'

SING_IMAGE="/gpfs/data/gpfs0/3ddl/singularity-images/artonson_colmap_singularity.sif"
SING_CONDA_ROOT='/gpfs/data/gpfs0/o.voinov/sk3d/env/miniconda3'
SING_CODE_ROOT='/gpfs/data/gpfs0/o.voinov/sk3d/dev.sk_robot_rgbd_data'
SING_EXP_CODE_ROOT='/gpfs/data/gpfs0/o.voinov/sk3d/sk3d.experiments'
SING_DATA_ROOT='/gpfs/data/gpfs0/3ddl/datasets/sk3d'
SING_RESULTS_ROOT='/gpfs/data/gpfs0/3ddl/projects/sk3d/experiments/colmap'

DOCK_IMAGE='sk3ddl/sk3d_colmap:v1.0.0'
DOCK_CONDA_ROOT='/home/ovoinov/work/sk3d/env/miniconda3'
DOCK_CODE_ROOT='/home/ovoinov/work/sk3d/dev.sk_robot_rgbd_data'
DOCK_EXP_CODE_ROOT='/home/ovoinov/work/sk3d/sk3d.experiments'
DOCK_DATA_ROOT='/mnt/datasets/sk3d'
DOCK_RESULTS_ROOT='/home/ovoinov/work/sk3d/experiments/colmap'


run_singularity(){
    SINGULARITY_SHELL=/bin/bash \
    singularity shell --nv \
        --bind $SING_CONDA_ROOT:$CONDA_ROOT \
        --bind $SING_DATA_ROOT:$DATA_ROOT \
        --bind $SING_RESULTS_ROOT:$RESULTS_ROOT \
        --bind $SING_CODE_ROOT:$CODE_ROOT \
        --bind $SING_EXP_CODE_ROOT:$EXP_CODE_ROOT \
        --bind $(pwd):$(pwd) \
        --pwd $(pwd) \
        $SING_IMAGE
}


run_docker(){
    THREADS_N=$(nproc --all)
    docker run --rm -it --gpus all \
        -e SLURM_CPUS_PER_TASK=$THREADS_N \
        --mount type=bind,source=${DOCK_CONDA_ROOT},target=${CONDA_ROOT} \
        --mount type=bind,source=${DOCK_DATA_ROOT},target=${DATA_ROOT} \
        --mount type=bind,source=${DOCK_RESULTS_ROOT},target=${RESULTS_ROOT} \
        --mount type=bind,source=${DOCK_CODE_ROOT},target=${CODE_ROOT} \
        --mount type=bind,source=${DOCK_EXP_CODE_ROOT},target=${EXP_CODE_ROOT} \
        --entrypoint bash \
        ${DOCK_IMAGE} -c "exec bash --init-file <(echo 'source ~/.bashrc && source ${EXP_CODE_ROOT}/COLMAP/env/utils.sh')"
}


activate_env(){
    source ${CONDA_ROOT}/bin/activate ${ENV_ROOT}
}