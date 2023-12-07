#!/bin/bash
if [ ! -d "cuda-samples" ]; then
    git clone https://github.com/NVIDIA/cuda-samples.git
    cd cuda-samples && git reset --hard 03309a2d4275a9186b748e033ee5f90a11492a2f
fi

# compile command without MPI
#cd /mcgpu && nvcc MC-GPU_v1.3.cu -o MC-GPU_v1.3.x -m64 -O3 -use_fast_math -DUSING_CUDA -I. -I/usr/local/cuda/include -I./cuda-samples/Common -L/usr/lib/ -lz --ptxas-options=-v -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86

# compile command with MPI
cd /mcgpu && nvcc MC-GPU_v1.3.cu -o MC-GPU_v1.3.x -m64 -O3 -use_fast_math -DUSING_CUDA -DUSING_MPI -I. -I/usr/local/cuda/include -I/cuda-samples/Common -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/ -lz -lmpi --ptxas-options=-v -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86

# make everything executable and link everything
chmod -R 755 /mcgpu
ln -s /mcgpu/MC-GPU_v1.3.x /usr/local/bin

# NVIDIA Compute Capability
# GeForce GTX 1080 Ti	6.1
# Geforce RTX 2080 Ti	7.5
# GeForce RTX 3090 Ti	8.6
# A100		            8.0
# A40		              8.6


# build ITK, VariationalRegistration and RTK (using CUDA)
mkdir /ITK_build && cd /ITK_build &&
  cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DBUILD_TESTING:BOOL=OFF \
    -DBUILD_EXAMPLES:BOOL=OFF \
    -DModule_RTK:BOOL=ON \
    -DRTK_BUILD_APPLICATIONS:BOOL=ON \
    -DRTK_USE_CUDA:BOOL=ON \
    -DRTK_CUDA_PROJECTIONS_SLAB_SIZE:STRING=16 \
    ../ITK && make -j 32

# make everything executable and link everything
chmod -R 755 /ITK_build/bin
ln -s /ITK_build/bin/* /usr/local/bin
