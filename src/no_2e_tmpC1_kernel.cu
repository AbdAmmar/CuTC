
#include <stdio.h>

__global__ void no_2e_tmpC1_kernel(int n_grid1, int n_mo,
                                   double * wr1, double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                                   double * tmpJ, double * tmpO, double * tmpA, double * tmpB,
                                   double * tmpC1) {


    int i_grid1;
    int p_mo;
    int s_mo;

    int ix, iix;

    int iA, iB;

    int n1, n2;

    double wr1_tmp;
    double mol_tmp;
    double mor_tmp;
    double O;
    double Jx, Jy, Jz;
    double Ax, Ay, Az;
    double Bx, By, Bz;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    n1 = 3 * n_grid1;
    n2 = n1 * n_mo;

    while(i_grid1 < n_grid1) {

        wr1_tmp = wr1[i_grid1];

        O = wr1_tmp * tmpO[i_grid1];

        Jx = wr1_tmp * tmpJ[i_grid1            ];
        Jy = wr1_tmp * tmpJ[i_grid1 +   n_grid1];
        Jz = wr1_tmp * tmpJ[i_grid1 + 2*n_grid1];

        for(p_mo = 0; p_mo < n_mo; p_mo++) {

            mol_tmp = mos_l_in_r[i_grid1 + p_mo*n_grid1];

            iA = i_grid1 + p_mo * n1;

            Ax = tmpA[iA            ];
            Ay = tmpA[iA +   n_grid1];
            Az = tmpA[iA + 2*n_grid1];

            ix = i_grid1 + p_mo * n1;

            for(s_mo = 0; s_mo < n_mo; s_mo++) {

                iix = ix + s_mo * n2;

                mor_tmp = mos_r_in_r[i_grid1 + s_mo*n_grid1];

                iB = i_grid1 + s_mo * n1;
                Bx = tmpB[iB            ];
                By = tmpB[iB +   n_grid1];
                Bz = tmpB[iB + 2*n_grid1];

                tmpC1[iix            ] = mor_tmp * Ax + mol_tmp * Bx - O * int2_grad1_u12[iix            ] - 2.0 * mol_tmp * mor_tmp * Jx;
                tmpC1[iix +   n_grid1] = mor_tmp * Ay + mol_tmp * By - O * int2_grad1_u12[iix +   n_grid1] - 2.0 * mol_tmp * mor_tmp * Jy;
                tmpC1[iix + 2*n_grid1] = mor_tmp * Az + mol_tmp * Bz - O * int2_grad1_u12[iix + 2*n_grid1] - 2.0 * mol_tmp * mor_tmp * Jz;

            } // s_mo

        } // p_mo

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_2e_tmpC1(int n_grid1, int n_mo,
                            double * wr1, double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                            double * tmpJ, double * tmpO, double * tmpA, double * tmpB,
                            double * tmpC1) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_2e_tmpC1_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_2e_tmpC1_kernel<<<nBlocks, blockSize>>>(n_grid1, n_mo,
                                               wr1, mos_l_in_r, mos_r_in_r, int2_grad1_u12,
                                               tmpJ, tmpO, tmpA, tmpB,
                                               tmpC1);

}


