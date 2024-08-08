
#include <stdio.h>

__global__ void no_2e_tmpB_cs_kernel(int n_grid1, int n_mo, int ne_b, 
                                     double * wr1, double * mos_l_in_r, double * int2_grad1_u12,
                                     double * tmpB) {


    int i_grid1;
    int ie;
    int p_mo;

    int ix, iy, iz;
    int jx, jjx;

    int n1;
    int n2;

    double wr1_tmp;
    double mol_tmp;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    n1 = 3 * n_grid1;
    n2 = n1 * n_mo;

    while(i_grid1 < n_grid1) {

        wr1_tmp = wr1[i_grid1];

        for(p_mo = 0; p_mo < n_mo; p_mo++) {

            ix = i_grid1 + p_mo * n1;
            iy = ix + n_grid1;
            iz = iy + n_grid1;

            tmpB[ix] = 0.0;
            tmpB[iy] = 0.0;
            tmpB[iz] = 0.0;

            jx = i_grid1 + p_mo * n2;

            for(ie = 0; ie < ne_b; ie++) {

                mol_tmp = mos_l_in_r[i_grid1 + ie * n_grid1];

                jjx = jx + ie * n1;

                tmpB[ix] += wr1_tmp * mol_tmp * int2_grad1_u12[jjx              ];
                tmpB[iy] += wr1_tmp * mol_tmp * int2_grad1_u12[jjx +     n_grid1];
                tmpB[iz] += wr1_tmp * mol_tmp * int2_grad1_u12[jjx + 2 * n_grid1];

            }
        }

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_2e_tmpB_cs(int n_grid1, int n_mo, int ne_b,
                              double * wr1, double * mos_l_in_r, double * int2_grad1_u12,
                              double * tmpB) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_2e_tmpB_cs_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_2e_tmpB_cs_kernel<<<nBlocks, blockSize>>>(n_grid1, n_mo, ne_b,
                                                 wr1, mos_l_in_r, int2_grad1_u12,
                                                 tmpB);

}


