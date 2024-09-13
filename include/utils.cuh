#ifndef UTILS
#define UTILS

extern "C" void checkCudaErrors(cudaError_t err, const char* msg, const char* file, int line);

extern "C" void checkCublasErrors(cublasStatus_t status, const char* msg, const char* file, int line);

#endif
