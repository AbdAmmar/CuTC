
#include <stdio.h>
#include <cublas_v2.h>

#include "short_range_integ_herm.cuh"
#include "short_range_integ_nonherm.cuh"
#include "utils.cuh"
#include "add_trans_inplace.cuh"


extern "C" void get_int_2e_ao(int n_grid1, int n_ao, double *wr1, double *aos_data1,
                              double *int2_grad1_u12, double *int_2e_ao) {


    double *int_fct_short_range_herm;
    double *int_fct_short_range_nonherm;

    double alpha, beta;

    int blockSize = 32;
    int nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching int_short_range_herm_kernel & int_short_range_nonherm_kernel and with %d blocks and %d threads/block\n", 
            nBlocks, blockSize);


    cublasHandle_t handle;

    checkCublasErrors(cublasCreate(&handle), "cublasCreate", __FILE__, __LINE__);


    // Hermitian part

    checkCudaErrors(cudaMalloc((void**)&int_fct_short_range_herm, n_grid1 * n_ao * n_ao * sizeof(double)), "cudaMalloc", __FILE__, __LINE__);

    int_short_range_herm_kernel<<<nBlocks, blockSize>>>(n_grid1, n_ao, wr1, aos_data1, int_fct_short_range_herm);

    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

    alpha = 1.0;
    beta = 0.0;

    checkCublasErrors( cublasDgemm( handle
                                  , CUBLAS_OP_N, CUBLAS_OP_N
                                  , n_ao*n_ao, n_ao*n_ao, n_grid1
                                  , &alpha
                                  , &int2_grad1_u12[n_ao*n_ao*n_grid1*3], n_ao*n_ao
                                  , &int_fct_short_range_herm[0], n_grid1
                                  , &beta
                                  , &int_2e_ao[0], n_ao*n_ao )
                     , "cublasDgemm", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(int_fct_short_range_herm), "cudaFree", __FILE__, __LINE__);

    // // //



    // non-Hermitian part

    checkCudaErrors(cudaMalloc((void**)&int_fct_short_range_nonherm, 3*n_grid1*n_ao*n_ao*sizeof(double)), "cudaMalloc", __FILE__, __LINE__);

    int_short_range_nonherm_kernel<<<nBlocks, blockSize>>>(n_grid1, n_ao, wr1, aos_data1, int_fct_short_range_nonherm);

    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

    alpha = -0.5;
    beta = 1.0;
    checkCublasErrors( cublasDgemm( handle
                                  , CUBLAS_OP_N, CUBLAS_OP_N
                                  , n_ao*n_ao, n_ao*n_ao, 3*n_grid1
                                  , &alpha
                                  , &int2_grad1_u12[0], n_ao*n_ao
                                  , &int_fct_short_range_nonherm[0], 3*n_grid1
                                  , &beta
                                  , &int_2e_ao[0], n_ao*n_ao )
                     , "cublasDgemm", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(int_fct_short_range_nonherm), "cudaFree", __FILE__, __LINE__);

    // // //


    // int_2e_ao <-- int_2e_ao + int_2e_ao.T

    int sBlocks = 32;
    int nbBlocks = (n_ao*n_ao + sBlocks - 1) / sBlocks;

    dim3 dimGrid(nbBlocks, nbBlocks, 1);
    dim3 dimBlock(sBlocks, sBlocks, 1);

    trans_inplace_kernel<<<dimGrid, dimBlock>>>(int_2e_ao, n_ao*n_ao);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

    // // //

    checkCublasErrors(cublasDestroy(handle), "cublasDestroy", __FILE__, __LINE__);

}


