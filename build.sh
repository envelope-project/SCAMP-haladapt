cd build
cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.1/ \
      -D CMAKE_CUDA_COMPILER=/usr/local/cuda-10.1/bin/nvcc \
      -D CMAKE_CXX_COMPILER=gcc ..
make -j64
