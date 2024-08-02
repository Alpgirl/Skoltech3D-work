#!/bin/bash

for scene in blue_shampoo_bottle
do
    docker cp larina_2dgs_2:/app/2d-gaussian-splatting/output/custom/${scene}/train/ours_30000/fuse_post.ply .
    mv fuse_post.ply ${scene}.ply
done