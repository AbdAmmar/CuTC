
#include <stdio.h>

__global__ void no_1e_tmpF_os_kernel(int n_grid1, int n_mo, int ne_b, int ne_a,
                                     double * wr1, double * mos_r_in_r, double * int2_grad1_u12,
                                     double * tmpS, double * tmpJ, double * tmpR,
                                     double * tmpF) {


    int i_grid1;
    int ie;
    int je;
    int p_mo;

    int iix;
    int jx, jjx;
    int kx, kkx;
    int lx, llx;

    int iR, iF;

    int n1, n2;
    int m1;

    double wr1_tmp;
    double mor_tmp, mor_i, mor_j;
    double S;
    double Jx, Jy, Jz;
    double Rx, Ry, Rz;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    n1 = 3 * n_grid1;
    n2 = n1 * n_mo;

    m1 = 5 * n_grid1;

    while(i_grid1 < n_grid1) {

        wr1_tmp = wr1[i_grid1];

        S = tmpS[i_grid1];

        Jx = tmpJ[i_grid1            ];
        Jy = tmpJ[i_grid1 +   n_grid1];
        Jz = tmpJ[i_grid1 + 2*n_grid1];

        for(p_mo = 0; p_mo < n_mo; p_mo++) {

            mor_tmp = mos_r_in_r[i_grid1 + p_mo*n_grid1];

            iR = i_grid1 + p_mo * n1;
            iF = i_grid1 + p_mo * m1;

            Rx = tmpR[iR            ];
            Ry = tmpR[iR +   n_grid1];
            Rz = tmpR[iR + 2*n_grid1];

            tmpF[iF            ] = -2.0 * (Rx * Jx + Ry * Jy + Rz * Jz) + mor_tmp * S;
            tmpF[iF +   n_grid1] = wr1_tmp * mor_tmp;
            tmpF[iF + 2*n_grid1] = Rx;
            tmpF[iF + 3*n_grid1] = Ry;
            tmpF[iF + 4*n_grid1] = Rz;

            jx = i_grid1 + p_mo * n2;

            for(ie = 0; ie < ne_b; ie++) {

                mor_i = mos_r_in_r[i_grid1 + ie*n_grid1];

                kx = i_grid1 + ie * n1;

                for(je = 0; je < ne_b; je++) {

                    jjx = jx + je * n1;
                    kkx = kx + je * n2;

                    tmpF[iF] += mor_i * int2_grad1_u12[jjx            ] * int2_grad1_u12[kkx            ];
                    tmpF[iF] += mor_i * int2_grad1_u12[jjx +   n_grid1] * int2_grad1_u12[kkx +   n_grid1];
                    tmpF[iF] += mor_i * int2_grad1_u12[jjx + 2*n_grid1] * int2_grad1_u12[kkx + 2*n_grid1];

                } // ie

            } // je

            for(ie = ne_b; ie < ne_a; ie++) {

                mor_i = mos_r_in_r[i_grid1 + ie*n_grid1];

                kx = i_grid1 + ie * n1;
                lx = i_grid1 + ie * n2;

                iix = jx + ie * n1;

                for(je = 0; je < ne_b; je++) {

                    mor_j = mos_r_in_r[i_grid1 + je*n_grid1];

                    jjx = jx + je * n1;
                    kkx = kx + je * n2;
                    llx = lx + je * n1;

                    tmpF[iF] += 0.5 * (mor_i * int2_grad1_u12[jjx            ] * int2_grad1_u12[kkx            ] + mor_j * int2_grad1_u12[iix            ] * int2_grad1_u12[llx            ]);
                    tmpF[iF] += 0.5 * (mor_i * int2_grad1_u12[jjx +   n_grid1] * int2_grad1_u12[kkx +   n_grid1] + mor_j * int2_grad1_u12[iix +   n_grid1] * int2_grad1_u12[llx +   n_grid1]);
                    tmpF[iF] += 0.5 * (mor_i * int2_grad1_u12[jjx + 2*n_grid1] * int2_grad1_u12[kkx + 2*n_grid1] + mor_j * int2_grad1_u12[iix + 2*n_grid1] * int2_grad1_u12[llx + 2*n_grid1]);

                } // ie

            } // je

            for(ie = ne_b; ie < ne_a; ie++) {

                mor_i = mos_r_in_r[i_grid1 + ie*n_grid1];

                kx = i_grid1 + ie * n1;

                for(je = ne_b; je < ne_a; je++) {

                    jjx = jx + je * n1;
                    kkx = kx + je * n2;

                    tmpF[iF] += 0.5 * mor_i * int2_grad1_u12[jjx            ] * int2_grad1_u12[kkx            ];
                    tmpF[iF] += 0.5 * mor_i * int2_grad1_u12[jjx +   n_grid1] * int2_grad1_u12[kkx +   n_grid1];
                    tmpF[iF] += 0.5 * mor_i * int2_grad1_u12[jjx + 2*n_grid1] * int2_grad1_u12[kkx + 2*n_grid1];

                } // ie

            } // je
    
        } // p_mo

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_1e_tmpF_os(int n_grid1, int n_mo, int ne_b, int ne_a,
                              double * wr1, double * mos_r_in_r, double * int2_grad1_u12,
                              double * tmpS, double * tmpJ, double * tmpR,
                              double * tmpF) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_1e_tmpF_os_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_1e_tmpF_os_kernel<<<nBlocks, blockSize>>>(n_grid1, n_mo, ne_b, ne_a,
                                                 wr1, mos_r_in_r, int2_grad1_u12,
                                                 tmpS, tmpJ, tmpR,
                                                 tmpF);

}


