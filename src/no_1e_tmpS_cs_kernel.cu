
#include <stdio.h>

__global__ void no_1e_tmpS_cs_kernel(int n_grid1, int n_mo, int ne_b, 
                                     double * int2_grad1_u12, double * tmpJ,
                                     double * tmpS) {


    int i_grid1;
    int ie;
    int je;

    int ix;
    int iix;

    int jx;
    int jjx;

    int n1;
    int n2;

    double Jx, Jy, Jz;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    n1 = 3 * n_grid1;
    n2 = n1 * n_mo;

    while(i_grid1 < n_grid1) {

        Jx = tmpJ[i_grid1            ];
        Jy = tmpJ[i_grid1 +   n_grid1];
        Jz = tmpJ[i_grid1 + 2*n_grid1];

        tmpS[i_grid1] = 2.0 * (Jx * Jx + Jy * Jy + Jz * Jz);

        for(ie = 0; ie < ne_b; ie++) {

            ix = i_grid1 + ie * n2;

            jx = i_grid1 + ie * n1;

            for(je = 0; je < ne_b; je++) {

                iix = ix + je * n1;

                jjx = jx + je * n2;

                tmpS[i_grid1] -= int2_grad1_u12[iix            ] * int2_grad1_u12[jjx            ];
                tmpS[i_grid1] -= int2_grad1_u12[iix +   n_grid1] * int2_grad1_u12[jjx +   n_grid1];
                tmpS[i_grid1] -= int2_grad1_u12[iix + 2*n_grid1] * int2_grad1_u12[jjx + 2*n_grid1];

            }

        }

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_1e_tmpS_cs(int n_grid1, int n_mo, int ne_b,
                              double * int2_grad1_u12, double * tmpJ,
                              double * tmpS) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_1e_tmpS_cs_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_1e_tmpS_cs_kernel<<<nBlocks, blockSize>>>(n_grid1, n_mo, ne_b,
                                                 int2_grad1_u12, tmpJ,
                                                 tmpS);

}


