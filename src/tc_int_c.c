#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>




extern void get_int2_grad1_u12_ao(int nBlocks, int blockSize,
                                  cublasHandle_t handle,
                                  int n_grid1, int n_grid2, int n_ao, int n_nuc, int size_bh,
                                  int n_grid1_pass, int n_grid1_rest, int n_pass,
                                  double *r1, double *r2, double *wr2, double *rn, double *aos_data2,
                                  double *c_bh, int *m_bh, int *n_bh, int *o_bh,
                                  double *int2_grad1_u12_ao);

extern void get_int_2e_ao(int nBlocks, int blockSize,
                          cublasHandle_t handle,
                          int n_grid1, int n_ao, double *wr1, double *aos_data1,
                          double *int2_grad1_u12_ao, double *tc_int_2e_ao);



int tc_int_c(int nBlocks, int blockSize,
             int n_grid1, int n_grid2, int n_ao, int n_nuc, int size_bh,
             double *h_r1, double *h_wr1, double *h_r2, double *h_wr2, double *h_rn,
             double *h_aos_data1, double *h_aos_data2,
             double *h_c_bh, int *h_m_bh, int *h_n_bh, int *h_o_bh, 
             double *h_int2_grad1_u12_ao, double *h_int_2e_ao) {

    int n_grid1_pass, n_grid1_rest, n_pass;
    double n_tmp;

    double *d_r1, *d_wr1;
    double *d_r2, *d_wr2;
    double *d_rn;

    double *d_aos_data1, *d_aos_data2;

    double *d_c_bh; 
    int *d_m_bh, *d_n_bh, *d_o_bh;

    double *d_int2_grad1_u12_ao;
    double *d_int_2e_ao;

    size_t size_r1, size_wr1, size_r2, size_wr2, size_rn;
    size_t size_aos_r1, size_aos_r2;
    size_t size_int1, size_int2;
    size_t size_jbh_d, size_jbh_i;

    cublasHandle_t handle;

    cublasCreate(&handle);



    size_jbh_d = size_bh * n_nuc * sizeof(double);
    size_jbh_i = size_bh * n_nuc * sizeof(int);

    size_r1 = 3 * n_grid1 * sizeof(double);
    size_r2 = 3 * n_grid2 * sizeof(double);
    size_wr2 = n_grid2 * sizeof(double);
    size_rn = 3 * n_nuc * sizeof(double);

    size_aos_r2 = 4 * n_grid2 * n_ao * sizeof(double);

    size_int1 = 4 * n_grid1 * n_ao * n_ao * sizeof(double);


    cudaMalloc((void**)&d_c_bh, size_jbh_d);
    cudaMalloc((void**)&d_m_bh, size_jbh_i);
    cudaMalloc((void**)&d_n_bh, size_jbh_i);
    cudaMalloc((void**)&d_o_bh, size_jbh_i);


    cudaMalloc((void**)&d_r1, size_r1);

    cudaMalloc((void**)&d_r2, size_r2);
    cudaMalloc((void**)&d_wr2, size_wr2);

    cudaMalloc((void**)&d_rn, size_rn);

    cudaMalloc((void**)&d_aos_data2, size_aos_r2);

    cudaMalloc((void**)&d_int2_grad1_u12_ao, size_int1);


    cudaMemcpy(d_c_bh, h_c_bh, size_jbh_d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_bh, h_m_bh, size_jbh_i, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_bh, h_n_bh, size_jbh_i, cudaMemcpyHostToDevice);
    cudaMemcpy(d_o_bh, h_o_bh, size_jbh_i, cudaMemcpyHostToDevice);

    cudaMemcpy(d_r1, h_r1, size_r1, cudaMemcpyHostToDevice);

    cudaMemcpy(d_r2, h_r2, size_r2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wr2, h_wr2, size_wr2, cudaMemcpyHostToDevice);

    cudaMemcpy(d_rn, h_rn, size_rn, cudaMemcpyHostToDevice);

    cudaMemcpy(d_aos_data2, h_aos_data2, size_aos_r2, cudaMemcpyHostToDevice);


    //         1 GB
    n_tmp = (1.0e9 / 8.0) / (4.0 * (double) n_grid2);
    if(n_tmp < 1.0*n_grid1) {
        n_grid1_pass = (int) n_tmp;
    } else {
        n_grid1_pass = n_grid1;
    }

    n_grid1_rest = (int) fmod(1.0 * n_grid1, 1.0 * n_grid1_pass);
    n_pass = (int) ((n_grid1 - n_grid1_rest) / n_grid1_pass);

    printf("n_grid1_pass = %d\n", n_grid1_pass);
    printf("n_grid1_rest = %d\n", n_grid1_rest);
    printf("n_pass = %d\n", n_pass);

    get_int2_grad1_u12_ao(nBlocks, blockSize, 
                          handle,
                          n_grid1, n_grid2, n_ao, n_nuc, size_bh,
                          n_grid1_pass, n_grid1_rest, n_pass,
                          d_r1, d_r2, d_wr2, d_rn, d_aos_data2,
                          d_c_bh, d_m_bh, d_n_bh, d_o_bh,
                          d_int2_grad1_u12_ao);


    cudaFree(d_c_bh);
    cudaFree(d_m_bh);
    cudaFree(d_n_bh);
    cudaFree(d_o_bh);

    cudaFree(d_r1);

    cudaFree(d_r2);
    cudaFree(d_wr2);

    cudaFree(d_rn);

    cudaFree(d_aos_data2);

    // // //



    // 2-e integral

    size_wr1 = n_grid1 * sizeof(double);
    size_aos_r1 = 4 * n_grid1 * n_ao * sizeof(double);
    size_int2 = n_ao * n_ao * n_ao * n_ao * sizeof(double);

    cudaMalloc((void**)&d_wr1, size_wr1);
    cudaMalloc((void**)&d_aos_data1, size_aos_r1);
    cudaMalloc((void**)&d_int_2e_ao, size_int2);

    cudaMemcpy(d_wr1, h_wr1, size_wr1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aos_data1, h_aos_data1, size_aos_r1, cudaMemcpyHostToDevice);

    get_int_2e_ao(nBlocks, blockSize, 
                  handle,
                  n_grid1, n_ao, d_wr1, d_aos_data1, d_int2_grad1_u12_ao, d_int_2e_ao);

    cudaFree(d_wr1);
    cudaFree(d_aos_data1);

    // // //



    // transfer data to Host

    cudaMemcpy(h_int2_grad1_u12_ao, d_int2_grad1_u12_ao, size_int1, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_int_2e_ao, d_int_2e_ao, size_int2, cudaMemcpyDeviceToHost);

    cudaFree(d_int2_grad1_u12_ao);
    cudaFree(d_int_2e_ao);

    // // //

    cublasDestroy(handle);

    return 0;
}


