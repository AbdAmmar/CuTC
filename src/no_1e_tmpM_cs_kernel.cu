
#include <stdio.h>

__global__ void no_1e_tmpM_cs_kernel(int n_grid1, int n_mo, int ne_b, 
                                     double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                                     double * tmpM) {


    int i_grid1;
    int ie;
    int je;

    int ix;
    int iix;

    int n1;
    int n2;

    double mol_tmp;
    double mor_tmp;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    n1 = 3 * n_grid1;
    n2 = n1 * n_mo;

    while(i_grid1 < n_grid1) {

        tmpM[i_grid1            ] = 0.0;
        tmpM[i_grid1 +   n_grid1] = 0.0;
        tmpM[i_grid1 + 2*n_grid1] = 0.0;

        for(ie = 0; ie < ne_b; ie++) {

            mol_tmp = mos_l_in_r[i_grid1 + ie * n_grid1];

            ix = i_grid1 + ie * n2;

            for(je = 0; je < ne_b; je++) {

                iix = ix + je * n1;

                mor_tmp = mos_r_in_r[i_grid1 + je * n_grid1];

                tmpM[i_grid1            ] += mol_tmp * mor_tmp * int2_grad1_u12[iix            ];
                tmpM[i_grid1 +   n_grid1] += mol_tmp * mor_tmp * int2_grad1_u12[iix +   n_grid1];
                tmpM[i_grid1 + 2*n_grid1] += mol_tmp * mor_tmp * int2_grad1_u12[iix + 2*n_grid1];

            }

        }

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_1e_tmpM_cs(int n_grid1, int n_mo, int ne_b,
                              double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                              double * tmpM) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_1e_tmpM_cs_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_1e_tmpM_cs_kernel<<<nBlocks, blockSize>>>(n_grid1, n_mo, ne_b,
                                                 mos_l_in_r, mos_r_in_r, int2_grad1_u12,
                                                 tmpM);

}


