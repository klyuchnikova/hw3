FROM nvcr.io/nvidia/tritonserver:25.01-py3

USER root

# Update package lists and install libstdc++6
RUN apt-get update && \
    apt-get install -y --no-install-recommends libstdc++6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER triton