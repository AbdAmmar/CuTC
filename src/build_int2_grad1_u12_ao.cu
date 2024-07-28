
#include <cublas_v2.h>

#include "jast_bh.cuh"
#include "long_range_integ.cuh"

extern "C" void get_int2_grad1_u12_ao(int nBlocks, int blockSize,
                                      int n_grid1, int n_grid2, int n_ao, int n_nuc, int size_bh,
                                      double *r1, double *r2, double *wr2, double *rn, double *aos_data2,
                                      double *c_bh, int *m_bh, int *n_bh, int *o_bh, 
                                      double *int2_grad1_u12_ao) {


    double *int_fct_long_range;
    double *grad1_u12;

    int m;
    double alpha, beta;


    cublasHandle_t handle;



    // calculate grad1_u12

    cudaMalloc((void**)&grad1_u12, 4 * n_grid1 * n_grid2 * sizeof(double));

    tc_int_bh_kernel<<<nBlocks, blockSize>>>(n_grid1, n_grid2, n_nuc, size_bh,
                                             r1, r2, rn,
                                             c_bh, m_bh, n_bh, o_bh,
                                             grad1_u12);

    // // //



    // calculate int_fct_long_range

    cudaMalloc((void**)&int_fct_long_range, n_grid2 * n_ao * n_ao * sizeof(double));

    int_long_range_kernel<<<nBlocks, blockSize>>>(n_grid2, n_ao, wr2, aos_data2, int_fct_long_range);

    // // //


    cudaDeviceSynchronize();

    cublasCreate(&handle);

    alpha = 1.0;
    beta = 0.0;
    for (m = 0; m < 4; m++) {
        cublasDgemm( handle
                   , CUBLAS_OP_T, CUBLAS_OP_N
                   , n_ao*n_ao, n_grid1, n_grid2
                   , &alpha
                   , &int_fct_long_range[0], n_grid2
                   , &grad1_u12[n_grid1*n_grid2*m], n_grid2
                   , &beta
                   , &int2_grad1_u12_ao[n_ao*n_ao*n_grid1*m], n_ao*n_ao );
    }

    cublasDestroy(handle);

    cudaFree(int_fct_long_range);
    cudaFree(grad1_u12);

}



