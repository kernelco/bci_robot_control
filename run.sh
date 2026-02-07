#!/bin/bash

# Configuration
IMAGE_NAME="holoscan-robot-control"
TAG="latest"
DOCKER_IMAGE="$IMAGE_NAME:$TAG"

# 1. Build the Docker image
echo "--- Building Holoscan Image: $DOCKER_IMAGE ---"
docker build -t "$DOCKER_IMAGE" .

if [ $? -ne 0 ]; then
    echo "Error: Docker build failed."
    exit 1
fi

# 2. Run the container with required hardware access
echo "--- Launching Container ---"

# --runtime=nvidia: Required to access the GPU
# --ulimit stack=...: Fixes the 32MB warning/segfault risk
# --ipc=host: Recommended for Holoscan shared memory transport
docker run --rm -it \
    --runtime=nvidia \
    --ulimit stack=33554432:33554432 \
    --ipc=host \
    --net=host \
    --device /dev/nvidia0 \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    -v "$(pwd):/app" \
    -v "/etc/kernel.com/kortex.json:/etc/kernel.com/kortex.json" \
    "$DOCKER_IMAGE"
