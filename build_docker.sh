#!/bin/bash
set -x
# build docker image
docker build --tag mcgpu:latest .
# do the advanced (CUDA) compiling
container_id=$(docker run -it --detach mcgpu:latest)
docker exec $container_id /bin/bash -c "cd /mcgpu && chmod +x ./compile.sh && ./compile.sh"
docker stop $container_id
docker commit $container_id mcgpu:latest
