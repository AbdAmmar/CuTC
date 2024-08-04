
#include <stdio.h>

__global__ void int_long_range_kernel(int n_grid2, int n_ao, double *wr2, double* aos_data2, double *int_fct_long_range) {


    int i_grid2;
    int i_ao, j_ao;
    int ii_ao, jj_ao;
    int ll0, ll1, ll2;

    double wr2_tmp;
    double ao_val_i, ao_val_j;


    i_grid2 = blockIdx.x * blockDim.x + threadIdx.x;

    ll0 = n_ao * n_grid2;

    while(i_grid2 < n_grid2) {

        wr2_tmp = wr2[i_grid2];

        for(i_ao = 0; i_ao < n_ao; i_ao++) {

            ll1 = i_grid2 + i_ao * ll0;

            ii_ao = i_grid2 + n_grid2 * i_ao;
            ao_val_i = aos_data2[ii_ao];

            for(j_ao = 0; j_ao < n_ao; j_ao++) {

                ll2 = ll1 + j_ao * n_grid2;

                jj_ao = i_grid2 + n_grid2 * j_ao;
                ao_val_j = aos_data2[jj_ao];

                int_fct_long_range[ll2] = wr2_tmp * ao_val_i * ao_val_j;

            } // j_ao
        } // i_ao

        i_grid2 += blockDim.x * gridDim.x;

    } // i_grid2

}



extern "C" void int_long_range(int n_grid2, int n_ao, double *wr2, double* aos_data2,
                               double *int_fct_long_range) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid2 + blockSize - 1) / blockSize;

    printf("lunching int_long_range_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    int_long_range_kernel<<<nBlocks, blockSize>>>(n_grid2, n_ao, wr2, aos_data2, int_fct_long_range);

}


