#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


extern void tc_int_bh(int nBlocks, int blockSize,
                      int ii0, int n_grid1, int n_grid2, int n_nuc, int size_bh,
                      double *r1, double *r2, double *rn,
                      double *c_bh, int *m_bh, int *n_bh, int *o_bh,
                      double *grad1_u12);


int deb_int_bh_kernel(int nBlocks, int blockSize,
                      int n_grid1, int n_grid2, int n_ao, int n_nuc, int size_bh,
                      double *h_r1, double *h_r2, double *h_rn,
                      double *h_c_bh, int *h_m_bh, int *h_n_bh, int *h_o_bh,
                      double *h_grad1_u12) {

    int ii;

    double *d_r1, *d_wr1;
    double *d_r2, *d_wr2;
    double *d_rn;

    double *d_c_bh;
    int *d_m_bh, *d_n_bh, *d_o_bh;
    double *d_grad1_u12;

    size_t size_r1, size_r2, size_rn;
    size_t size_jbh_d, size_jbh_i;

    printf(" DEBUG tc_int_bh_kernel\n");

    size_jbh_d = size_bh * n_nuc * sizeof(double);
    size_jbh_i = size_bh * n_nuc * sizeof(int);

    size_r1 = 3 * n_grid1 * sizeof(double);
    size_r2 = 3 * n_grid2 * sizeof(double);
    size_rn = 3 * n_nuc * sizeof(double);

    cudaMalloc((void**)&d_c_bh, size_jbh_d);
    cudaMalloc((void**)&d_m_bh, size_jbh_i);
    cudaMalloc((void**)&d_n_bh, size_jbh_i);
    cudaMalloc((void**)&d_o_bh, size_jbh_i);

    cudaMalloc((void**)&d_r1, size_r1);
    cudaMalloc((void**)&d_r2, size_r2);
    cudaMalloc((void**)&d_rn, size_rn);

    cudaMemcpy(d_c_bh, h_c_bh, size_jbh_d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_bh, h_m_bh, size_jbh_i, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_bh, h_n_bh, size_jbh_i, cudaMemcpyHostToDevice);
    cudaMemcpy(d_o_bh, h_o_bh, size_jbh_i, cudaMemcpyHostToDevice);

    cudaMemcpy(d_r1, h_r1, size_r1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r2, h_r2, size_r2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rn, h_rn, size_rn, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_grad1_u12, 4 * n_grid1 * n_grid2 * sizeof(double));

    ii = 0;
    tc_int_bh(nBlocks, blockSize, 
              ii, n_grid1, n_grid2, n_nuc, size_bh,
              d_r1, d_r2, d_rn,
              d_c_bh, d_m_bh, d_n_bh, d_o_bh,
              d_grad1_u12);

    cudaDeviceSynchronize();


    cudaMemcpy(h_grad1_u12, d_grad1_u12, 4 * n_grid1 * n_grid2 * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_c_bh);
    cudaFree(d_m_bh);
    cudaFree(d_n_bh);
    cudaFree(d_o_bh);

    cudaFree(d_r1);
    cudaFree(d_r2);
    cudaFree(d_rn);

    cudaFree(d_grad1_u12);

    return 0;
}


