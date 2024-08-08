
#include <stdio.h>

__global__ void no_2e_tmpO_cs_kernel(int n_grid1, int ne_b, 
                                     double * wr1, double * mos_l_in_r, double * mos_r_in_r,
                                     double * tmpO) {


    int i_grid1;
    int ie;
    int i_mo;

    double wr1_tmp;

    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    while(i_grid1 < n_grid1) {

        tmpO[i_grid1] = 0.0;

        wr1_tmp = wr1[i_grid1];

        for(ie = 0; ie < ne_b; ie++) {

            i_mo = i_grid1 + ie * n_grid1;

            tmpO[i_grid1] += wr1_tmp * mos_l_in_r[i_mo] * mos_r_in_r[i_mo];

        }

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_2e_tmpO_cs(int n_grid1, int ne_b,
                              double * wr1, double * mos_l_in_r, double * mos_r_in_r,
                              double * tmpO) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_2e_tmpO_cs_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_2e_tmpO_cs_kernel<<<nBlocks, blockSize>>>(n_grid1, ne_b,
                                                 wr1, mos_l_in_r, mos_r_in_r,
                                                 tmpO);

}

