
#include <stdio.h>

__global__ void no_2e_tmpJ_os_kernel(int n_grid1, int n_mo, int ne_b, int ne_a,
                                     double * wr1, double * int2_grad1_u12,
                                     double * tmpJ) {


    int i_grid1;
    int ie;

    int ii_grid1;

    int nn;

    double wr1_tmp;

    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    nn = 3 * n_grid1 * (1 + n_mo);

    while(i_grid1 < n_grid1) {

        tmpJ[i_grid1            ] = 0.0;
        tmpJ[i_grid1 +   n_grid1] = 0.0;
        tmpJ[i_grid1 + 2*n_grid1] = 0.0;

        wr1_tmp = wr1[i_grid1];

        for(ie = 0; ie < ne_b; ie++) {

            ii_grid1 = i_grid1 + ie * nn;

            tmpJ[i_grid1            ] += wr1_tmp * int2_grad1_u12[ii_grid1            ];
            tmpJ[i_grid1 +   n_grid1] += wr1_tmp * int2_grad1_u12[ii_grid1 +   n_grid1];
            tmpJ[i_grid1 + 2*n_grid1] += wr1_tmp * int2_grad1_u12[ii_grid1 + 2*n_grid1];

        }

        for(ie = ne_b; ie < ne_a; ie++) {

            ii_grid1 = i_grid1 + ie * nn;

            tmpJ[i_grid1            ] += 0.5 * wr1_tmp * int2_grad1_u12[ii_grid1            ];
            tmpJ[i_grid1 +   n_grid1] += 0.5 * wr1_tmp * int2_grad1_u12[ii_grid1 +   n_grid1];
            tmpJ[i_grid1 + 2*n_grid1] += 0.5 * wr1_tmp * int2_grad1_u12[ii_grid1 + 2*n_grid1];

        }

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_2e_tmpJ_os(int n_grid1, int n_mo, int ne_b, int ne_a,
                              double * wr1, double * int2_grad1_u12,
                              double * tmpJ) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_2e_tmpJ_os_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_2e_tmpJ_os_kernel<<<nBlocks, blockSize>>>(n_grid1, n_mo, ne_b, ne_a,
                                                 wr1, int2_grad1_u12,
                                                 tmpJ);

}


