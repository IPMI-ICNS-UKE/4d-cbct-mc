#!/bin/bash
set -x

# build docker image
docker build --tag cbct-mc:latest .
# do the advanced (CUDA) compiling
container_id=$(docker run -it --detach cbct-mc:latest)
docker exec $container_id /bin/bash -c "cd / && chmod +x ./compile.sh && ./compile.sh"
docker stop $container_id
docker commit $container_id cbct-mc:latest
