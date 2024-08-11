
#include <stdio.h>

__global__ void no_0e_tmpG_os_kernel(int n_grid1, int n_mo, int ne_b, int ne_a,
                                     double * wr1, double * mos_l_in_r, double * int2_grad1_u12,
                                     double * tmpG) {


    int i_grid1;
    int ie;
    int je;

    int ix, iy, iz;
    int iix;

    int n1;
    int n2;

    double wr1_tmp;
    double mol_tmp;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    n1 = 3 * n_grid1;
    n2 = n1 * n_mo;

    while(i_grid1 < n_grid1) {

        wr1_tmp = wr1[i_grid1];

        for(je = 0; je < ne_b; je++) {

            ix = i_grid1 + je * n1;
            iy = ix + n_grid1;
            iz = iy + n_grid1;

            tmpG[ix] = 0.0;
            tmpG[iy] = 0.0;
            tmpG[iz] = 0.0;

            for(ie = ne_b; ie < ne_a; ie++) {

                mol_tmp = mos_l_in_r[i_grid1 + ie * n_grid1];

                iix = ix + ie * n2;

                tmpG[ix] += 0.5 * mol_tmp * wr1_tmp * int2_grad1_u12[iix            ];
                tmpG[iy] += 0.5 * mol_tmp * wr1_tmp * int2_grad1_u12[iix +   n_grid1];
                tmpG[iz] += 0.5 * mol_tmp * wr1_tmp * int2_grad1_u12[iix + 2*n_grid1];

            }
        }

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_0e_tmpG_os(int n_grid1, int n_mo, int ne_b, int ne_a,
                              double * wr1, double * mos_l_in_r, double * int2_grad1_u12,
                              double * tmpG) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_0e_tmpG_os_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_0e_tmpG_os_kernel<<<nBlocks, blockSize>>>(n_grid1, n_mo, ne_b, ne_a,
                                                 wr1, mos_l_in_r, int2_grad1_u12,
                                                 tmpG);

}


