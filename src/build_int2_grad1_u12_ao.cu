
#include <stdio.h>
#include <cublas_v2.h>

#include "jast_bh.cuh"
#include "long_range_integ.cuh"
#include "utils.cuh"

extern "C" void get_int2_grad1_u12_ao(dim3 dimGrid, dim3 dimBlock,
                                      int n_grid1, int n_grid2, int n_ao, int n_nuc, int size_bh,
                                      double *r1, double *r2, double *wr2, double *rn, double *aos_data2,
                                      double *c_bh, int *m_bh, int *n_bh, int *o_bh, 
                                      double *int2_grad1_u12_ao) {


    int i_pass;
    int ii, jj, kk;
    int jj0, kk0;
    int n_grid1_pass, n_grid1_rest, n_pass;

    double *int_fct_long_range;
    double *grad1_u12;

    int m;
    double alpha, beta;

    size_t size_sh_mem;
    size_t free_mem, total_mem;

    double n_tmp;

    int blockSize = 32;
    int nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching int_long_range_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    size_sh_mem = n_nuc * size_bh * (sizeof(double) + 3 * sizeof(int)) 
                + 3 * n_nuc * sizeof(double);

    cublasHandle_t myhandle;

    checkCublasErrors(cublasCreate(&myhandle), "cublasCreate", __FILE__, __LINE__);

    alpha = 1.0;
    beta = 0.0;

    jj0 = n_ao * n_ao;

    checkCudaErrors(cudaMalloc((void**)&int_fct_long_range, n_grid2 * n_ao * n_ao * sizeof(double)), "cudaMalloc", __FILE__, __LINE__);

    int_long_range_kernel<<<nBlocks, blockSize>>>(0, n_grid2, n_grid2,
                                                  n_grid2, n_ao, wr2, aos_data2, int_fct_long_range);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);



    checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem), "cudaMemGetInfo", __FILE__, __LINE__);

    n_tmp = (((double)free_mem - 0.5e9) / 8.0) / (4.0 * (double) n_grid2);
    if(n_tmp < 1.0*n_grid1) {
        if(n_tmp > 1.0) {
            n_grid1_pass = (int) n_tmp;
        } else {
            n_grid1_pass = 1;
        }
    } else {
        n_grid1_pass = n_grid1;
    }

    n_grid1_rest = (int) fmod(1.0 * n_grid1, 1.0 * n_grid1_pass);
    n_pass = (int) ((n_grid1 - n_grid1_rest) / n_grid1_pass);

    printf("n_grid1_pass = %d\n", n_grid1_pass);
    printf("n_grid1_rest = %d\n", n_grid1_rest);
    printf("n_pass = %d\n", n_pass);



    checkCudaErrors(cudaMalloc((void**)&grad1_u12, 4 * n_grid1_pass * n_grid2 * sizeof(double)), "cudaMalloc", __FILE__, __LINE__);

    kk0 = n_grid1_pass * n_grid2;

    for (i_pass = 0; i_pass < n_pass; i_pass++) {

        ii = i_pass * n_grid1_pass;

        // TODO
        tc_int_bh_kernel<<<dimGrid, dimBlock, size_sh_mem>>>(ii, n_grid1_pass, n_grid1_pass,
                                                             0, n_grid2, n_grid2,
                                                             n_nuc, size_bh,
                                                             r1, r2, rn,
                                                             c_bh, m_bh, n_bh, o_bh,
                                                             grad1_u12);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
    
        for (m = 0; m < 4; m++) {
            jj = jj0 * (ii + m * n_grid1);
            kk = kk0 * m;
            checkCublasErrors( cublasDgemm( myhandle
                                          , CUBLAS_OP_T, CUBLAS_OP_N
                                          , n_ao*n_ao, n_grid1_pass, n_grid2
                                          , &alpha
                                          , &int_fct_long_range[0], n_grid2
                                          , &grad1_u12[kk], n_grid2
                                          , &beta
                                          , &int2_grad1_u12_ao[jj], n_ao*n_ao )
                             , "cublasDgemm", __FILE__, __LINE__);
        }

    }
    
    if(n_grid1_rest > 0) {

        ii = n_pass * n_grid1_pass;

        // TODO
        tc_int_bh_kernel<<<dimGrid, dimBlock, size_sh_mem>>>(ii, n_grid1_rest, n_grid1_pass,
                                                             0, n_grid2, n_grid2,
                                                             n_nuc, size_bh,
                                                             r1, r2, rn,
                                                             c_bh, m_bh, n_bh, o_bh,
                                                             grad1_u12);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
    
        for (m = 0; m < 4; m++) {
            jj = jj0 * (ii + m * n_grid1);
            kk = kk0 * m;
            checkCublasErrors( cublasDgemm( myhandle
                                          , CUBLAS_OP_T, CUBLAS_OP_N
                                          , n_ao*n_ao, n_grid1_rest, n_grid2
                                          , &alpha
                                          , &int_fct_long_range[0], n_grid2
                                          , &grad1_u12[kk], n_grid2
                                          , &beta
                                          , &int2_grad1_u12_ao[jj], n_ao*n_ao )
                             , "cublasDgemm", __FILE__, __LINE__);
        }

    }


    checkCublasErrors(cublasDestroy(myhandle), "cublasDestroy", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(int_fct_long_range), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(grad1_u12), "cudaFree", __FILE__, __LINE__);

}



