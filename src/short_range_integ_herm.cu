
#include <stdio.h>

__global__ void int_short_range_herm_kernel(int n_grid1, int n_ao, double *wr1, double* aos_data1, double *int_fct_short_range_herm) {


    int i_grid1;
    int i_ao, j_ao;
    int ii_ao, jj_ao;
    int ll0, ll1, ll2;

    double wr1_tmp;
    double ao_val_i, ao_val_j;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    ll0 = n_ao * n_grid1;

    while(i_grid1 < n_grid1) {

        wr1_tmp = wr1[i_grid1];

        for(i_ao = 0; i_ao < n_ao; i_ao++) {

            ll1 = i_grid1 + i_ao * ll0;

            ii_ao = i_grid1 + n_grid1 * i_ao;
            ao_val_i = aos_data1[ii_ao];

            for(j_ao = 0; j_ao < n_ao; j_ao++) {

                ll2 = ll1 + j_ao * n_grid1;

                jj_ao = i_grid1 + n_grid1 * j_ao;
                ao_val_j = aos_data1[jj_ao];

                int_fct_short_range_herm[ll2] = wr1_tmp * ao_val_i * ao_val_j;

            } // j_ao
        } // i_ao

        i_grid1 += blockDim.x * gridDim.x;

    } // i_grid1

}



extern "C" void int_short_range_herm(int n_grid1, int n_ao, double *wr1, double* aos_data1,
                                     double *int_fct_short_range_herm) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching int_short_range_herm_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    int_short_range_herm_kernel<<<nBlocks, blockSize>>>(n_grid1, n_ao, wr1, aos_data1, int_fct_short_range_herm);

}


