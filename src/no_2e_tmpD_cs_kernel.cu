
#include <stdio.h>

__global__ void no_2e_tmpD_cs_kernel(int n_grid1, int n_mo,
                                     double * wr1, double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                                     double * tmpD) {


    int i_grid1;
    int p_mo;
    int s_mo;

    int ix, iix;
    int jx, jjx;

    int n1, n2;
    int m1, m2;

    double wr1_tmp;
    double mol_tmp;
    double mor_tmp;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    n1 = 3 * n_grid1;
    n2 = n1 * n_mo;

    m1 = n1 + n_grid1;
    m2 = m1 * n_mo;

    while(i_grid1 < n_grid1) {

        wr1_tmp = wr1[i_grid1];

        for(p_mo = 0; p_mo < n_mo; p_mo++) {

            mol_tmp = mos_l_in_r[i_grid1 + p_mo*n_grid1];

            ix = i_grid1 + p_mo * m1;
            jx = i_grid1 + p_mo * n1;

            for(s_mo = 0; s_mo < n_mo; s_mo++) {

                iix = ix + s_mo * m2;

                jjx = jx + s_mo * n2;

                mor_tmp = mos_r_in_r[i_grid1 + s_mo*n_grid1];

                tmpD[iix            ] = int2_grad1_u12[jjx            ];
                tmpD[iix +   n_grid1] = int2_grad1_u12[jjx +   n_grid1];
                tmpD[iix + 2*n_grid1] = int2_grad1_u12[jjx + 2*n_grid1];
                tmpD[iix + 3*n_grid1] = wr1_tmp * mol_tmp * mor_tmp;

            } // s_mo

        } // p_mo

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_2e_tmpD_cs(int n_grid1, int n_mo,
                              double * wr1, double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                              double * tmpD) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_2e_tmpD_cs_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_2e_tmpD_cs_kernel<<<nBlocks, blockSize>>>(n_grid1, n_mo,
                                                 wr1, mos_l_in_r, mos_r_in_r, int2_grad1_u12,
                                                 tmpD);

}


