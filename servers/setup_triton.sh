#!/bin/bash
apt update
apt install -y libre2-5 libre2-dev libb64-dev
# Add Triton Server libraries and binaries to the environment variables
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/servers/tritonserver/lib"
export PATH="${PATH}:/servers/tritonserver/bin"
