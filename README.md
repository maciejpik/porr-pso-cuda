# porr-pso-cuda
This repository is prepared for VS 2019 project. You can also compile these files using the following CMakeLists.txt:

```
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(PORR_CUDA LANGUAGES CUDA CXX)

add_executable(PORR_CUDA main.cu)

set_property(TARGET PORR_CUDA PROPERTY CUDA_ARCHITECTURES 75)
set_property(TARGET PORR_CUDA PROPERTY CUDA_SEPARABLE_COMPILATION ON)
```
