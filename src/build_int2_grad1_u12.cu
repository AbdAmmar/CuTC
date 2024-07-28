
#include <cublas_v2.h>

#include "jast_bh.cuh"
#include "long_range_integ.cuh"

extern "C" void get_int2_grad1_u12(int nBlocks, int blockSize,
                                   int n_grid1, int n_grid2, int ao_num, int n_nuc, int size_bh,
                                   double *r1, double *r2, double *wr2, double *rn, double *aos_data2,
                                   double *c_bh, int *m_bh, int *n_bh, int *o_bh, 
                                   double *int2_grad1_u12) {


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

    cudaMalloc((void**)&int_fct_long_range, n_grid2 * ao_num * ao_num * sizeof(double));

    int_long_range_kernel<<<nBlocks, blockSize>>>(n_grid2, ao_num, wr2, aos_data2, int_fct_long_range);

    // // //


    cudaDeviceSynchronize();

    cublasCreate(&handle);

    alpha = 1.0;
    beta = 0.0;
    for (m = 0; m < 4; m++) {
        cublasDgemm( handle
                   , CUBLAS_OP_T, CUBLAS_OP_N
                   , ao_num*ao_num, n_grid1, n_grid2
                   , &alpha
                   , &int_fct_long_range[0], n_grid2
                   , &grad1_u12[n_grid1*n_grid2*m], n_grid2
                   , &beta
                   , &int2_grad1_u12[ao_num*ao_num*n_grid1*m], ao_num*ao_num );
    }

    cublasDestroy(handle);

    cudaFree(int_fct_long_range);
    cudaFree(grad1_u12);

}



