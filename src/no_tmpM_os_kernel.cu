
#include <stdio.h>

__global__ void no_tmpM_os_kernel(int n_grid1, int n_mo, int ne_b, int ne_a,
                                  double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                                  double * tmpM) {


    int i_grid1;
    int ie;
    int je;

    int ix, iix;
    int jx, jjx;

    int n1;
    int n2;

    double mol_i, mol_j;
    double mor_i, mor_j;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    n1 = 3 * n_grid1;
    n2 = n1 * n_mo;

    while(i_grid1 < n_grid1) {

        tmpM[i_grid1            ] = 0.0;
        tmpM[i_grid1 +   n_grid1] = 0.0;
        tmpM[i_grid1 + 2*n_grid1] = 0.0;

        for(ie = 0; ie < ne_b; ie++) {

            mol_i = mos_l_in_r[i_grid1 + ie * n_grid1];

            ix = i_grid1 + ie * n2;

            for(je = 0; je < ne_b; je++) {

                iix = ix + je * n1;

                mor_j = mos_r_in_r[i_grid1 + je * n_grid1];

                tmpM[i_grid1            ] += mol_i * mor_j * int2_grad1_u12[iix            ];
                tmpM[i_grid1 +   n_grid1] += mol_i * mor_j * int2_grad1_u12[iix +   n_grid1];
                tmpM[i_grid1 + 2*n_grid1] += mol_i * mor_j * int2_grad1_u12[iix + 2*n_grid1];

            }

        }

        for(ie = ne_b; ie < ne_a; ie++) {

            mol_i = mos_l_in_r[i_grid1 + ie * n_grid1];
            mor_i = mos_r_in_r[i_grid1 + ie * n_grid1];

            ix = i_grid1 + ie * n2;
            jx = i_grid1 + ie * n1;

            for(je = 0; je < ne_b; je++) {

                iix = ix + je * n1;
                jjx = jx + je * n2;

                mol_j = mos_l_in_r[i_grid1 + je * n_grid1];
                mor_j = mos_r_in_r[i_grid1 + je * n_grid1];

                tmpM[i_grid1            ] += 0.5 * (mol_i * mor_j * int2_grad1_u12[iix            ] + mol_j * mor_i * int2_grad1_u12[jjx            ]);
                tmpM[i_grid1 +   n_grid1] += 0.5 * (mol_i * mor_j * int2_grad1_u12[iix +   n_grid1] + mol_j * mor_i * int2_grad1_u12[jjx +   n_grid1]);
                tmpM[i_grid1 + 2*n_grid1] += 0.5 * (mol_i * mor_j * int2_grad1_u12[iix + 2*n_grid1] + mol_j * mor_i * int2_grad1_u12[jjx + 2*n_grid1]);

            }

        }

        for(ie = ne_b; ie < ne_a; ie++) {

            mol_i = mos_l_in_r[i_grid1 + ie * n_grid1];

            ix = i_grid1 + ie * n2;

            for(je = ne_b; je < ne_a; je++) {

                iix = ix + je * n1;

                mor_j = mos_r_in_r[i_grid1 + je * n_grid1];

                tmpM[i_grid1            ] += 0.5 * mol_i * mor_j * int2_grad1_u12[iix            ];
                tmpM[i_grid1 +   n_grid1] += 0.5 * mol_i * mor_j * int2_grad1_u12[iix +   n_grid1];
                tmpM[i_grid1 + 2*n_grid1] += 0.5 * mol_i * mor_j * int2_grad1_u12[iix + 2*n_grid1];

            }

        }


        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_tmpM_os(int n_grid1, int n_mo, int ne_b, int ne_a,
                           double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                           double * tmpM) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_tmpM_os_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_tmpM_os_kernel<<<nBlocks, blockSize>>>(n_grid1, n_mo, ne_b, ne_a,
                                              mos_l_in_r, mos_r_in_r, int2_grad1_u12,
                                              tmpM);

}


