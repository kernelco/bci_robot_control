# Use the official Holoscan SDK base (Ubuntu 22.04 + CUDA 13.0)
FROM nvcr.io/nvidia/clara-holoscan/holoscan:v3.10.0-cuda13

# Set required ulimit for GXF (must also be passed at 'docker run')
# Note: Dockerfile ENV doesn't set kernel ulimits, but we can set defaults
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

# Install application dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY . .

# Recommended entrypoint
CMD ["python3", "-m", "applications.robot_control"]
