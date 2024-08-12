
#include <stdio.h>

__global__ void no_1e_tmpE_os_kernel(int n_grid1, int n_mo, int ne_b, int ne_a,
                                     double * wr1, double * mos_l_in_r, double * int2_grad1_u12,
                                     double * tmpJ, double * tmpL,
                                     double * tmpE) {


    int i_grid1;
    int ie;
    int je;
    int p_mo;

    int jjx;
    int kx, kkx;
    int llx;
    int mx, mmx;

    int iL, iE;
    int iiE;

    int n1, n2;
    int m1;

    double wr1_tmp;
    double mol_tmp, mol_i, mol_j;
    double Jx, Jy, Jz;
    double Lx, Ly, Lz;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    n1 = 3 * n_grid1;
    n2 = n1 * n_mo;

    m1 = 5 * n_grid1;

    while(i_grid1 < n_grid1) {

        wr1_tmp = wr1[i_grid1];

        Jx = tmpJ[i_grid1            ];
        Jy = tmpJ[i_grid1 +   n_grid1];
        Jz = tmpJ[i_grid1 + 2*n_grid1];

        for(p_mo = 0; p_mo < n_mo; p_mo++) {

            mol_tmp = mos_l_in_r[i_grid1 + p_mo*n_grid1];

            iL = i_grid1 + p_mo * n1;
            iE = i_grid1 + p_mo * m1;

            iiE = iE + n_grid1;

            Lx = tmpL[iL            ];
            Ly = tmpL[iL +   n_grid1];
            Lz = tmpL[iL + 2*n_grid1];

            tmpE[iE            ] = wr1_tmp * mol_tmp;
            tmpE[iE +   n_grid1] = -2.0 * (Lx * Jx + Ly * Jy + Lz * Jz);
            tmpE[iE + 2*n_grid1] = Lx;
            tmpE[iE + 3*n_grid1] = Ly;
            tmpE[iE + 4*n_grid1] = Lz;

            for(je = 0; je < ne_b; je++) {

                mol_j = wr1_tmp * mos_l_in_r[i_grid1 + je*n_grid1];

                kx = i_grid1 + je * n2;

                for(ie = 0; ie < ne_b; ie++) {

                    jjx = iL + ie * n2;
                    kkx = kx + ie * n1;

                    tmpE[iiE] += mol_j * int2_grad1_u12[jjx            ] * int2_grad1_u12[kkx            ];
                    tmpE[iiE] += mol_j * int2_grad1_u12[jjx +   n_grid1] * int2_grad1_u12[kkx +   n_grid1];
                    tmpE[iiE] += mol_j * int2_grad1_u12[jjx + 2*n_grid1] * int2_grad1_u12[kkx + 2*n_grid1];

                } // ie

            } // je

            for(je = 0; je < ne_b; je++) {

                mol_j = 0.5 * wr1_tmp *  mos_l_in_r[i_grid1 + je*n_grid1];

                kx = i_grid1 + je * n2;
                mx = i_grid1 + je * n1;

                llx = iL + je * n2;

                for(ie = ne_b; ie < ne_a; ie++) {

                    mol_i = 0.5 * wr1_tmp * mos_l_in_r[i_grid1 + ie*n_grid1];

                    jjx = iL + ie * n2;
                    kkx = kx + ie * n1;
                    mmx = mx + ie * n2;

                    tmpE[iiE] += mol_j * int2_grad1_u12[jjx            ] * int2_grad1_u12[kkx            ] + mol_i * int2_grad1_u12[llx            ] * int2_grad1_u12[mmx            ];
                    tmpE[iiE] += mol_j * int2_grad1_u12[jjx +   n_grid1] * int2_grad1_u12[kkx +   n_grid1] + mol_i * int2_grad1_u12[llx +   n_grid1] * int2_grad1_u12[mmx +   n_grid1];
                    tmpE[iiE] += mol_j * int2_grad1_u12[jjx + 2*n_grid1] * int2_grad1_u12[kkx + 2*n_grid1] + mol_i * int2_grad1_u12[llx + 2*n_grid1] * int2_grad1_u12[mmx + 2*n_grid1];

                } // ie

            } // je

            for(je = ne_b; je < ne_a; je++) {

                mol_j = 0.5 * wr1_tmp * mos_l_in_r[i_grid1 + je*n_grid1];

                kx = i_grid1 + je * n2;

                for(ie = ne_b; ie < ne_a; ie++) {

                    jjx = iL + ie * n2;
                    kkx = kx + ie * n1;

                    tmpE[iiE] += mol_j * int2_grad1_u12[jjx            ] * int2_grad1_u12[kkx            ];
                    tmpE[iiE] += mol_j * int2_grad1_u12[jjx +   n_grid1] * int2_grad1_u12[kkx +   n_grid1];
                    tmpE[iiE] += mol_j * int2_grad1_u12[jjx + 2*n_grid1] * int2_grad1_u12[kkx + 2*n_grid1];

                } // ie

            } // je
    
        } // p_mo

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_1e_tmpE_os(int n_grid1, int n_mo, int ne_b, int ne_a,
                              double * wr1, double * mos_l_in_r, double * int2_grad1_u12,
                              double * tmpJ, double * tmpL,
                              double * tmpE) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_1e_tmpE_os_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_1e_tmpE_os_kernel<<<nBlocks, blockSize>>>(n_grid1, n_mo, ne_b, ne_a,
                                                 wr1, mos_l_in_r, int2_grad1_u12,
                                                 tmpJ, tmpL,
                                                 tmpE);

}


