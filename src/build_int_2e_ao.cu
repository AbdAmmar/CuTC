
#include <cublas_v2.h>

#include "short_range_integ_herm.cuh"
#include "short_range_integ_nonherm.cuh"


extern "C" void get_int_2e_ao(int nBlocks, int blockSize,
                              int n_grid1, int n_ao, double *wr1, double *aos_data1,
                              double *int2_grad1_u12, double *int_2e_ao) {


    double *int_fct_short_range_herm;
    double *int_fct_short_range_nonherm;

    double alpha, beta;

    cublasHandle_t handle;


    // Hermitian part

    cudaMalloc((void**)&int_fct_short_range_herm, n_grid1 * n_ao * n_ao * sizeof(double));

    int_short_range_herm_kernel<<<nBlocks, blockSize>>>(n_grid1, n_ao, wr1, aos_data1, int_fct_short_range_herm);
    cudaDeviceSynchronize();

    cublasCreate(&handle);

    cublasDgemm( handle
               , CUBLAS_OP_N, CUBLAS_OP_N
               , n_ao*n_ao, n_ao*n_ao, n_grid1
               , &alpha
               , &int2_grad1_u12[n_ao*n_ao*n_grid1*3], n_ao*n_ao
               , &int_fct_short_range_herm[0], n_grid1
               , &beta
               , &int_2e_ao[0], n_ao*n_ao );

    cublasDestroy(handle);

    cudaFree(int_fct_short_range_herm);

    // // //



    // non-Hermitian part

    cudaMalloc((void**)&int_fct_short_range_nonherm, 3*n_grid1*n_ao*n_ao*sizeof(double));
    int_short_range_nonherm_kernel<<<nBlocks, blockSize>>>(n_grid1, n_ao, wr1, aos_data1, int_fct_short_range_nonherm);

    cublasCreate(&handle);

    alpha = -1.0;
    beta = 1.0;
    cublasDgemm( handle
               , CUBLAS_OP_N, CUBLAS_OP_N
               , n_ao*n_ao, n_ao*n_ao, 3*n_grid1
               , &alpha
               , &int2_grad1_u12[0], n_ao*n_ao
               , &int_fct_short_range_nonherm[0], 3*n_grid1
               , &beta
               , &int_2e_ao[0], n_ao*n_ao );

    cublasDestroy(handle);

    cudaFree(int_fct_short_range_nonherm);

    // // //


    // int_2e_ao <-- int_2e_ao + int_2e_ao.T

    cublasCreate(&handle);

    alpha = 1.0;
    beta = 1.0;
    cublasDgeam( handle
               , CUBLAS_OP_T, CUBLAS_OP_N
               , n_ao*n_ao, n_ao*n_ao
               , &alpha
               , &int_2e_ao[0], n_ao*n_ao
               , &beta
               , &int_2e_ao[0], n_ao*n_ao
               , &int_2e_ao[0], n_ao*n_ao );

    cublasDestroy(handle);

    // // //

}


