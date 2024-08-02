# FROM ubuntu:latest

# install apt dependencies
# RUN apt-get update -y && apt-get install -y python3.8 python3-pip pytorch2.0.0 torchaudio2.0.0 torchvision0.15.0 typing_extensions4.9.0
#RUN pip install -r requirements.txt
# FROM continuumio/miniconda3
FROM mambaorg/micromamba:0.15.3

# Set the working directory in the container
WORKDIR /app

# Install Micromamba
# RUN conda install -c conda-forge micromamba

# Copy the environment yml file to the working directory
COPY env.yml .

# Create Conda environment from the yml file
# RUN micromamba create -f env.yml -y
# Create the Micromamba environment from the environment.yml file
RUN micromamba create -f env.yml -y 

# Set the environment path
# ENV PATH /opt/conda/envs/surfel_splatting/bin:$PATH

