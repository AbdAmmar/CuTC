
#include <stdio.h>

__global__ void no_2e_tmpC_cs_kernel(int n_grid1, int n_mo, int ne_b, 
                                     double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                                     double * tmpJ, double * tmpO, double * tmpA, double * tmpB,
                                     double * tmpC) {


    int i_grid1;
    int ie;
    int p_mo;
    int s_mo;

    int ix, iix;
    int jx, jjx;
    int kx, kkx;

    int n1, n2;
    int m1, m2;

    double mol_tmp;
    double mor_tmp;
    double O;
    double Jx, Jy, Jz;
    double Ax, Ay, Az;
    double Bx, By, Bz;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    n1 = 3 * n_grid1;
    n2 = n1 * n_mo;

    m1 = n1 + n_grid1;
    m2 = m1 * n_mo;

    while(i_grid1 < n_grid1) {

        O = tmpO[i_grid1];

        Jx = tmpJ[i_grid1            ];
        Jy = tmpJ[i_grid1 +   n_grid1];
        Jz = tmpJ[i_grid1 + 2*n_grid1];

        for(p_mo = 0; p_mo < n_mo; p_mo++) {

            mol_tmp = mos_l_in_r[i_grid1 + p_mo*n_grid1];

            Ax = tmpA[i_grid1 + p_mo * n1            ];
            Ay = tmpA[i_grid1 + p_mo * n1 +   n_grid1];
            Az = tmpA[i_grid1 + p_mo * n1 + 2*n_grid1];

            ix = i_grid1 + p_mo * m1;

            jx = i_grid1 + p_mo * n1;

            for(s_mo = 0; s_mo < n_mo; s_mo++) {

                iix = ix + s_mo * m2;

                jjx = jx + s_mo * n2;

                mor_tmp = mos_r_in_r[i_grid1 + s_mo*n_grid1];

                Bx = tmpB[i_grid1 + s_mo * n1            ];
                By = tmpB[i_grid1 + s_mo * n1 +   n_grid1];
                Bz = tmpB[i_grid1 + s_mo * n1 + 2*n_grid1];

                tmpC[iix            ] = mor_tmp * Ax + mol_tmp * Bx - O * int2_grad1_u12[jjx            ] - 2.0 * mol_tmp * mor_tmp * Jx;
                tmpC[iix +   n_grid1] = mor_tmp * Ay + mol_tmp * By - O * int2_grad1_u12[jjx +   n_grid1] - 2.0 * mol_tmp * mor_tmp * Jy;
                tmpC[iix + 2*n_grid1] = mor_tmp * Az + mol_tmp * Bz - O * int2_grad1_u12[jjx + 2*n_grid1] - 2.0 * mol_tmp * mor_tmp * Jz;

                kx = i_grid1 + s_mo * n2;

                tmpC[iix + 3*n_grid1] = 0.0;

                for(ie = 0; ie < ne_b; ie++) {

                    jjx = jx + ie * n2;

                    kkx = kx + ie * n1;

                    tmpC[iix + 3*n_grid1] += int2_grad1_u12[jjx            ] * int2_grad1_u12[kkx            ];
                    tmpC[iix + 3*n_grid1] += int2_grad1_u12[jjx +   n_grid1] * int2_grad1_u12[kkx +   n_grid1];
                    tmpC[iix + 3*n_grid1] += int2_grad1_u12[jjx + 2*n_grid1] * int2_grad1_u12[kkx + 2*n_grid1];

                } // ie
    
            } // s_mo

        } // p_mo

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_2e_tmpC_cs(int n_grid1, int n_mo, int ne_b,
                              double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                              double * tmpJ, double * tmpO, double * tmpA, double * tmpB,
                              double * tmpC) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_2e_tmpC_cs_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_2e_tmpC_cs_kernel<<<nBlocks, blockSize>>>(n_grid1, n_mo, ne_b,
                                                 mos_l_in_r, mos_r_in_r, int2_grad1_u12,
                                                 tmpJ, tmpO, tmpA, tmpB,
                                                 tmpC);

}


