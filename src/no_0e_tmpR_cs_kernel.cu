
#include <stdio.h>

__global__ void no_0e_tmpR_cs_kernel(int n_grid1, int n_mo, int ne_b, 
                                     double * mos_r_in_r, double * int2_grad1_u12,
                                     double * tmpR) {


    int i_grid1;
    int ie;
    int je;

    int ix, iy, iz;
    int jx, jjx;

    int n1;
    int n2;

    double mor_tmp;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    n1 = 3 * n_grid1;
    n2 = n1 * n_mo;

    while(i_grid1 < n_grid1) {

        for(je = 0; je < ne_b; je++) {

            ix = i_grid1 + je * n1;
            iy = ix + n_grid1;
            iz = iy + n_grid1;

            tmpR[ix] = 0.0;
            tmpR[iy] = 0.0;
            tmpR[iz] = 0.0;

            jx = i_grid1 + je * n2;

            for(ie = 0; ie < ne_b; ie++) {

                mor_tmp = mos_r_in_r[i_grid1 + ie * n_grid1];

                jjx = jx + ie * n1;

                tmpR[ix] += mor_tmp * int2_grad1_u12[jjx            ];
                tmpR[iy] += mor_tmp * int2_grad1_u12[jjx +   n_grid1];
                tmpR[iz] += mor_tmp * int2_grad1_u12[jjx + 2*n_grid1];

            }

        }

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_0e_tmpR_cs(int n_grid1, int n_mo, int ne_b,
                              double * mos_r_in_r, double * int2_grad1_u12,
                              double * tmpR) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_0e_tmpR_cs_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_0e_tmpR_cs_kernel<<<nBlocks, blockSize>>>(n_grid1, n_mo, ne_b,
                                                 mos_r_in_r, int2_grad1_u12,
                                                 tmpR);

}


