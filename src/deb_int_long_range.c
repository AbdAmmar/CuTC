#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


extern void int_long_range(int nBlocks, int blockSize,
                           int n_grid2, int n_ao, double *wr2, double* aos_data2,
                           double *int_fct_long_range);


int deb_int_long_range(int nBlocks, int blockSize,
                       int n_grid2, int n_ao, double *h_wr2, double *h_aos_data2,
                       double *h_int_fct_long_range) {

    size_t size_wr2;
    size_t size_aos_r2;

    double *d_wr2;
    double *d_aos_data2;
    double *d_int_fct_long_range;


    printf(" DEBUG int_long_range_kernel\n");


    size_wr2 = n_grid2 * sizeof(double);
    size_aos_r2 = 4 * n_grid2 * n_ao * sizeof(double);


    cudaMalloc((void**)&d_wr2, size_wr2);
    cudaMalloc((void**)&d_aos_data2, size_aos_r2);
    cudaMalloc((void**)&d_int_fct_long_range, n_grid2 * n_ao * n_ao * sizeof(double));

    cudaMemcpy(d_wr2, h_wr2, size_wr2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aos_data2, h_aos_data2, size_aos_r2, cudaMemcpyHostToDevice);


    int_long_range(nBlocks, blockSize, n_grid2, n_ao, d_wr2, d_aos_data2, d_int_fct_long_range);
    cudaDeviceSynchronize();


    cudaMemcpy(h_int_fct_long_range, d_int_fct_long_range, n_grid2 * n_ao * n_ao * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_wr2);
    cudaFree(d_aos_data2);
    cudaFree(d_int_fct_long_range);

    return 0;
}


