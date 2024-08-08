
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


    int n_ao2;
    int kk0, kk;
    int ii, jj;
    int m, mm;
    int i_pass, n_grid1_pass, n_grid1_rest, n1_pass;
    int j_pass, n_grid2_pass, n_grid2_rest, n2_pass;

    double n_tmp;

    double *int_fct_long_range;
    double *grad1_u12;

    double alpha, beta;

    size_t size_sh_mem;
    size_t free_mem, total_mem;

    size_t size_1, size_2;

    int blockSize;
    int nBlocks_pass;
    int nBlocks_rest;

    cublasHandle_t myhandle;

    checkCublasErrors(cublasCreate(&myhandle), "cublasCreate", __FILE__, __LINE__);

    n_ao2 = n_ao * n_ao;

    size_sh_mem = n_nuc * size_bh * (sizeof(double) + 3 * sizeof(int)) 
                + 3 * n_nuc * sizeof(double);


    checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem), "cudaMemGetInfo", __FILE__, __LINE__);
    printf(" free  memory = %.3f GB\n", (double)free_mem / 1073741824.0);
    printf(" total memory = %.3f GB\n", (double)total_mem/ 1073741824.0);

    /*
        chose n_grid1_past and  n_grid2_past suche that
            4 * n_grid1_past * n_grid2_past + n_ao * n_ao * n_grid2_past = (free memory) / sizeof(double)
        and
            n_grid2_past = 5 * n_grid1_past
    */


    n_tmp = 0.125 * (sqrt((double)(n_ao2*n_ao2) + 8.0*(double)free_mem) - (double)n_ao2);
    if(n_tmp < 1.0*n_grid1) {
        n_grid1_pass = (int) n_tmp;
    } else {
        n_grid1_pass = n_grid1;
    }

    n_grid1_rest = (int) fmod(1.0 * n_grid1, 1.0 * n_grid1_pass);
    n1_pass = (int) ((n_grid1 - n_grid1_rest) / n_grid1_pass);

    printf("n_grid1_pass = %d\n", n_grid1_pass);
    printf("n_grid1_rest = %d\n", n_grid1_rest);
    printf("n1_pass = %d\n", n1_pass);

    n_grid2_pass = 5 * n_grid1_pass;
    if(n_grid2_pass > n_grid2) {
        n_grid2_pass = n_grid2;
    }

    n_grid2_rest = (int) fmod(1.0 * n_grid2, 1.0 * n_grid2_pass);
    n2_pass = (int) ((n_grid2 - n_grid2_rest) / n_grid2_pass);

    printf("n_grid2_pass = %d\n", n_grid2_pass);
    printf("n_grid2_rest = %d\n", n_grid2_rest);
    printf("n2_pass = %d\n", n2_pass);

    size_1 = n_grid2_pass * n_ao2 * sizeof(double);
    size_2 = 4 * n_grid1_pass * n_grid2_pass * sizeof(double);
    kk0 = n_grid2_pass * n_grid1_pass;

    blockSize = 32;
    nBlocks_pass = (n_grid2_pass + blockSize - 1) / blockSize;
    nBlocks_rest = (n_grid2_pass + blockSize - 1) / blockSize;

    checkCudaErrors(cudaMalloc((void**)&int_fct_long_range, size_1), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&grad1_u12, size_2), "cudaMalloc", __FILE__, __LINE__);

    // // //

    alpha = 1.0;
    beta = 0.0;

    for (i_pass = 0; i_pass < n1_pass; i_pass++) {

        ii = i_pass * n_grid1_pass;

        for (j_pass = 0; j_pass < n2_pass; j_pass++) {

            jj = j_pass * n_grid2_pass;

            int_long_range_kernel<<<nBlocks_pass, blockSize>>>(jj, n_grid2_pass, n_grid2_pass,
                                                               n_grid2, n_ao, wr2, aos_data2, int_fct_long_range);
            checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
            checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

            cutc_int_bh_kernel<<<dimGrid, dimBlock, size_sh_mem>>>(ii, n_grid1_pass, n_grid1_pass,
                                                                   jj, n_grid2_pass, n_grid2_pass,
                                                                   n_nuc, size_bh,
                                                                   r1, r2, rn,
                                                                   c_bh, m_bh, n_bh, o_bh,
                                                                   grad1_u12);
            checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
            checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

            for (m = 0; m < 4; m++) {
                mm = n_ao2 * (ii + m * n_grid1);
                kk = kk0 * m;
                checkCublasErrors(cublasDgemm(myhandle,
                                              CUBLAS_OP_T, CUBLAS_OP_N,
                                              n_ao2, n_grid1_pass, n_grid2_pass,
                                              &alpha,
                                              &int_fct_long_range[0], n_grid2_pass,
                                              &grad1_u12[kk], n_grid2_pass,
                                              &beta,
                                              &int2_grad1_u12_ao[mm], n_ao2), "cublasDgemm", __FILE__, __LINE__);
            }

            beta = 1.0;

        }

        if(n_grid2_rest > 0) {

            jj = n2_pass * n_grid2_pass;

            int_long_range_kernel<<<nBlocks_rest, blockSize>>>(jj, n_grid2_rest, n_grid2_pass,
                                                               n_grid2, n_ao, wr2, aos_data2, int_fct_long_range);
            checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
            checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

            cutc_int_bh_kernel<<<dimGrid, dimBlock, size_sh_mem>>>(ii, n_grid1_pass, n_grid1_pass,
                                                                   jj, n_grid2_rest, n_grid2_pass,
                                                                   n_nuc, size_bh,
                                                                   r1, r2, rn,
                                                                   c_bh, m_bh, n_bh, o_bh,
                                                                   grad1_u12);
            checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
            checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

            for (m = 0; m < 4; m++) {
                mm = n_ao2 * (ii + m * n_grid1);
                kk = kk0 * m;
                checkCublasErrors(cublasDgemm(myhandle,
                                              CUBLAS_OP_T, CUBLAS_OP_N,
                                              n_ao2, n_grid1_pass, n_grid2_rest,
                                              &alpha,
                                              &int_fct_long_range[0], n_grid2_pass,
                                              &grad1_u12[kk], n_grid2_pass,
                                              &beta,
                                              &int2_grad1_u12_ao[mm], n_ao2), "cublasDgemm", __FILE__, __LINE__);
            }

        }

    }



    if(n_grid1_rest > 0) {

        ii = n1_pass * n_grid1_pass;

        for (j_pass = 0; j_pass < n2_pass; j_pass++) {

            jj = j_pass * n_grid2_pass;

            int_long_range_kernel<<<nBlocks_pass, blockSize>>>(jj, n_grid2_pass, n_grid2_pass,
                                                               n_grid2, n_ao, wr2, aos_data2, int_fct_long_range);
            checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
            checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

            cutc_int_bh_kernel<<<dimGrid, dimBlock, size_sh_mem>>>(ii, n_grid1_rest, n_grid1_pass,
                                                                   jj, n_grid2_pass, n_grid2_pass,
                                                                   n_nuc, size_bh,
                                                                   r1, r2, rn,
                                                                   c_bh, m_bh, n_bh, o_bh,
                                                                   grad1_u12);
            checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
            checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

            for (m = 0; m < 4; m++) {
                mm = n_ao2 * (ii + m * n_grid1);
                kk = kk0 * m;
                checkCublasErrors(cublasDgemm(myhandle,
                                              CUBLAS_OP_T, CUBLAS_OP_N,
                                              n_ao2, n_grid1_rest, n_grid2_pass,
                                              &alpha,
                                              &int_fct_long_range[0], n_grid2_pass,
                                              &grad1_u12[kk], n_grid2_pass,
                                              &beta,
                                              &int2_grad1_u12_ao[mm], n_ao2), "cublasDgemm", __FILE__, __LINE__);
            }

        }

        if(n_grid2_rest > 0) {

            jj = n2_pass * n_grid2_pass;

            int_long_range_kernel<<<nBlocks_rest, blockSize>>>(jj, n_grid2_rest, n_grid2_pass,
                                                               n_grid2, n_ao, wr2, aos_data2, int_fct_long_range);
            checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
            checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

            cutc_int_bh_kernel<<<dimGrid, dimBlock, size_sh_mem>>>(ii, n_grid1_rest, n_grid1_pass,
                                                                   jj, n_grid2_rest, n_grid2_pass,
                                                                   n_nuc, size_bh,
                                                                   r1, r2, rn,
                                                                   c_bh, m_bh, n_bh, o_bh,
                                                                   grad1_u12);
            checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
            checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

            for (m = 0; m < 4; m++) {
                mm = n_ao2 * (ii + m * n_grid1);
                kk = kk0 * m;
                checkCublasErrors(cublasDgemm(myhandle,
                                              CUBLAS_OP_T, CUBLAS_OP_N,
                                              n_ao2, n_grid1_rest, n_grid2_rest,
                                              &alpha,
                                              &int_fct_long_range[0], n_grid2_pass,
                                              &grad1_u12[kk], n_grid2_pass,
                                              &beta,
                                              &int2_grad1_u12_ao[mm], n_ao2), "cublasDgemm", __FILE__, __LINE__);
            }

        }

    }


    checkCublasErrors(cublasDestroy(myhandle), "cublasDestroy", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(int_fct_long_range), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(grad1_u12), "cudaFree", __FILE__, __LINE__);

}

