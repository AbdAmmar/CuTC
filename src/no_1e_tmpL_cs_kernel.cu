
#include <stdio.h>

__global__ void no_1e_tmpL_cs_kernel(int n_grid1, int n_mo, int ne_b, 
                                     double * mos_l_in_r, double * int2_grad1_u12,
                                     double * tmpL) {


    int i_grid1;
    int ie;
    int p_mo;

    int ix, iy, iz;
    int iix;

    int n1;
    int n2;

    double mol_tmp;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    n1 = 3 * n_grid1;
    n2 = n1 * n_mo;

    while(i_grid1 < n_grid1) {

        for(p_mo = 0; p_mo < n_mo; p_mo++) {

            ix = i_grid1 + p_mo * n1;
            iy = ix + n_grid1;
            iz = iy + n_grid1;

            tmpL[ix] = 0.0;
            tmpL[iy] = 0.0;
            tmpL[iz] = 0.0;

            for(ie = 0; ie < ne_b; ie++) {

                mol_tmp = mos_l_in_r[i_grid1 + ie * n_grid1];

                iix = ix + ie * n2;

                tmpL[ix] += mol_tmp * int2_grad1_u12[iix            ];
                tmpL[iy] += mol_tmp * int2_grad1_u12[iix +   n_grid1];
                tmpL[iz] += mol_tmp * int2_grad1_u12[iix + 2*n_grid1];

            }
        }

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_1e_tmpL_cs(int n_grid1, int n_mo, int ne_b,
                              double * mos_l_in_r, double * int2_grad1_u12,
                              double * tmpL) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_1e_tmpL_cs_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_1e_tmpL_cs_kernel<<<nBlocks, blockSize>>>(n_grid1, n_mo, ne_b,
                                                 mos_l_in_r, int2_grad1_u12,
                                                 tmpL);

}


