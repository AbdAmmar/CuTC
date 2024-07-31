
#include <cublas_v2.h>

#include "jast_bh.cuh"
#include "long_range_integ.cuh"

extern "C" void get_int2_grad1_u12_ao(int nBlocks, int blockSize,
                                      cublasHandle_t handle,
                                      int n_grid1, int n_grid2, int n_ao, int n_nuc, int size_bh,
                                      int n_grid1_pass, int n_grid1_rest, int n_pass,
                                      double *r1, double *r2, double *wr2, double *rn, double *aos_data2,
                                      double *c_bh, int *m_bh, int *n_bh, int *o_bh, 
                                      double *int2_grad1_u12_ao) {


    int i_pass;
    int ii, jj, kk;
    int jj0, kk0;

    double *int_fct_long_range;
    double *grad1_u12;

    int m;
    double alpha, beta;

    cublasHandle_t myhandle;

    cublasCreate(&myhandle);

    alpha = 1.0;
    beta = 0.0;

    jj0 = n_ao * n_ao;
    kk0 = n_grid1_pass * n_grid2;

    cudaMalloc((void**)&int_fct_long_range, n_grid2 * n_ao * n_ao * sizeof(double));

    int_long_range_kernel<<<nBlocks, blockSize>>>(n_grid2, n_ao, wr2, aos_data2, int_fct_long_range);


    cudaMalloc((void**)&grad1_u12, 4 * n_grid1_pass * n_grid2 * sizeof(double));

    for (i_pass = 0; i_pass < n_pass; i_pass++) {

        ii = i_pass * n_grid1_pass;
  
        tc_int_bh_kernel<<<nBlocks, blockSize>>>(ii, n_grid1, n_grid2, n_nuc, size_bh,
                                                 r1, r2, rn,
                                                 c_bh, m_bh, n_bh, o_bh,
                                                 grad1_u12);
    
        cudaDeviceSynchronize();
    
        for (m = 0; m < 4; m++) {
            jj = jj0 * (ii + m * n_grid1);
            kk = kk0 * m;
            cublasDgemm( myhandle
                       , CUBLAS_OP_T, CUBLAS_OP_N
                       , n_ao*n_ao, n_grid1_pass, n_grid2
                       , &alpha
                       , &int_fct_long_range[0], n_grid2
                       , &grad1_u12[kk], n_grid2
                       , &beta
                       , &int2_grad1_u12_ao[jj], n_ao*n_ao );
        }

    }
    
    if(n_grid1_rest > 0) {

        ii = n_pass * n_grid1_pass;
     
        tc_int_bh_kernel<<<nBlocks, blockSize>>>(ii, n_grid1, n_grid2, n_nuc, size_bh,
                                                 r1, r2, rn,
                                                 c_bh, m_bh, n_bh, o_bh,
                                                 grad1_u12);
    
        cudaDeviceSynchronize();
    
        for (m = 0; m < 4; m++) {
            jj = jj0 * (ii + m * n_grid1);
            kk = kk0 * m;
            cublasDgemm( myhandle
                       , CUBLAS_OP_T, CUBLAS_OP_N
                       , n_ao*n_ao, n_grid1_rest, n_grid2
                       , &alpha
                       , &int_fct_long_range[0], n_grid2
                       , &grad1_u12[kk], n_grid2
                       , &beta
                       , &int2_grad1_u12_ao[jj], n_ao*n_ao );
        }

    }


    cublasDestroy(myhandle);

    cudaFree(int_fct_long_range);
    cudaFree(grad1_u12);

}



