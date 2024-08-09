
#include <stdio.h>

__global__ void no_1e_tmpJ_cs_kernel(int n_grid1, int n_mo, int ne_b, 
                                     double * int2_grad1_u12,
                                     double * tmpJ) {


    int i_grid1;
    int ie;

    int ii_grid1;

    int nn;

    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    nn = 3 * n_grid1 * (1 + n_mo);

    while(i_grid1 < n_grid1) {

        tmpJ[i_grid1            ] = 0.0;
        tmpJ[i_grid1 +   n_grid1] = 0.0;
        tmpJ[i_grid1 + 2*n_grid1] = 0.0;

        for(ie = 0; ie < ne_b; ie++) {

            ii_grid1 = i_grid1 + ie * nn;

            tmpJ[i_grid1            ] += int2_grad1_u12[ii_grid1            ];
            tmpJ[i_grid1 +   n_grid1] += int2_grad1_u12[ii_grid1 +   n_grid1];
            tmpJ[i_grid1 + 2*n_grid1] += int2_grad1_u12[ii_grid1 + 2*n_grid1];

        }

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_1e_tmpJ_cs(int n_grid1, int n_mo, int ne_b,
                              double * int2_grad1_u12,
                              double * tmpJ) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_1e_tmpJ_cs_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_1e_tmpJ_cs_kernel<<<nBlocks, blockSize>>>(n_grid1, n_mo, ne_b,
                                                 int2_grad1_u12,
                                                 tmpJ);

}


