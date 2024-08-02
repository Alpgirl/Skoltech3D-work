#!/bin/bash
# clean current location folder
if [ -d "location" ] ; then
    echo "Delete location folder"
    cd location/images
    chmod 777 *
    cd ../
    chmod 777 *
    rm -r images
    cd ../
    rm -r location
fi

if [ ! -d "location" ] ; then
    mkdir location
    chmod 777 location
fi

# define environmental variables for scene
export CONFIG_VERSION='v3.0.0'
export SK3D_CAM='tis_right'
export SK3D_LIGHT='ambient@best'
export SCENE_NAME='blue_shampoo_bottle'

# copy colmap sfm data (sparse/0) from colmap docker container 
cd location
docker cp beautiful_mccarthy:/app/working/location/colmap/${CONFIG_VERSION}/${SK3D_CAM}/${SK3D_LIGHT}/${SCENE_NAME}/subsystem/sparse/ .

# copy undistorted images from initial dataset
cp -r /mnt/datasets/sk3d/dataset/${SCENE_NAME}/${SK3D_CAM}/rgb/undistorted/${SK3D_LIGHT} .

# rename undistortde folder to images
mv ${SK3D_LIGHT} images
cd ..

# copy filled data template to docker container with 2DGS
docker cp location larina_2dgs_2:/app/2d-gaussian-splatting/