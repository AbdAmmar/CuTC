
#include <stdio.h>

__global__ void int_short_range_nonherm_kernel(int n_grid1, int n_ao, double *wr1, double* aos_data1, double *int_fct_short_range_nonherm) {


    int i_grid1;
    int m, mm;
    int i_ao, j_ao;
    int ii0_ao, ii1_ao;
    int jj0_ao, jj1_ao;
    int ll0, ll1, ll2, ll3;

    double wr1_tmp;
    double ao_val_i, ao_val_j;
    double ao_der_i, ao_der_j;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    ll0 = n_ao * n_grid1;

    while(i_grid1 < n_grid1) {

        wr1_tmp = wr1[i_grid1];

        for(i_ao = 0; i_ao < n_ao; i_ao++) {

            ll1 = 3 * i_ao * ll0 + i_grid1;

            ii0_ao = i_grid1 + n_grid1 * i_ao;

            ao_val_i = aos_data1[ii0_ao];

            for(j_ao = 0; j_ao < n_ao; j_ao++) {

                ll2 = ll1 + 3 * j_ao * n_grid1;

                jj0_ao = i_grid1 + n_grid1 * j_ao;

                ao_val_j = aos_data1[jj0_ao];

                for(m = 0; m < 3; m++) {

                    mm = (m + 1) * n_ao * n_grid1;

                    ii1_ao = ii0_ao + mm;
                    jj1_ao = jj0_ao + mm;
                    
                    ao_der_i = aos_data1[ii1_ao];
                    ao_der_j = aos_data1[jj1_ao];

                    ll3 = m * n_grid1 + ll2;

                    int_fct_short_range_nonherm[ll3] = wr1_tmp * (ao_val_j * ao_der_i - ao_val_i * ao_der_j);

                } // m
            } // j_ao
        } // i_ao

        i_grid1 += blockDim.x * gridDim.x;

    } // i_grid1

}



extern "C" void int_short_range_nonherm(int n_grid1, int n_ao, double *wr1, double* aos_data1,
                                        double *int_fct_short_range_nonherm) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    //printf("lunching int_short_range_nonherm_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    int_short_range_nonherm_kernel<<<nBlocks, blockSize>>>(n_grid1, n_ao, wr1, aos_data1, int_fct_short_range_nonherm);

}


