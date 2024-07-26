
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>



extern "C" void checkCudaErrors(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA Error: %f - %f", msg, cudaGetErrorString(err));
        exit(0);
    }
}

