# Start from a clean Ubuntu base
# FROM ubuntu:22.04 
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
# Set noninteractive mode to prevent interactive prompts
# ENV DEBIAN_FRONTEND=noninteractive
# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    software-properties-common \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# Add NVIDIA package repositories
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update
# Install CUDA Toolkit (Replace with desired version)
RUN apt-get install -y cuda-toolkit-12-2
# Set environment variables
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN nvcc --version

EXPOSE 30501
WORKDIR /app
COPY . .

# Install Python and pip
RUN apt update && apt install -y python3 python3-venv python3-pip
RUN apt-get update && apt-get install -y libgl1
RUN python3 -m venv /opt/venv
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install opencv-python-headless
RUN pip install jupyter

# Set the entry point
# ENTRYPOINT ["python3", "/app/YOLO_model/main.py"]


# FROM python:3.12.2
# # FROM nvidia/cuda
# WORKDIR /app
# COPY . .
# RUN apt-get update && apt-get install -y libgl1
# RUN pip install -r requirements.txt
# RUN pip install opencv-python-headless
# EXPOSE 30501

# # Set the entry point
# ENTRYPOINT ["python", "/app/YOLO_model/main.py"]