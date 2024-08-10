
#include <stdio.h>

__global__ void no_1e_tmpC_cs_kernel(int n_grid1, int n_mo, int ne_b, 
                                     double * int2_grad1_u12,
                                     double * tmpC) {


    int i_grid1;
    int ie;
    int p_mo;
    int s_mo;

    int ix;
    int jx, jjx;
    int kx, kkx;
    int iC;

    int n1, n2;
    int m1, m2;

    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    n1 = 3 * n_grid1;
    n2 = n1 * n_mo;

    m1 = n1 + n_grid1;
    m2 = m1 * n_mo;

    while(i_grid1 < n_grid1) {

        for(p_mo = 0; p_mo < n_mo; p_mo++) {

            ix = i_grid1 + p_mo * m1;

            jx = i_grid1 + p_mo * n1;

            for(s_mo = 0; s_mo < n_mo; s_mo++) {

                iC = ix + s_mo * m2;

                jjx = jx + s_mo * n2;

                tmpC[iC            ] = int2_grad1_u12[jjx            ];
                tmpC[iC +   n_grid1] = int2_grad1_u12[jjx +   n_grid1];
                tmpC[iC + 2*n_grid1] = int2_grad1_u12[jjx + 2*n_grid1];

                kx = i_grid1 + s_mo * n2;

                tmpC[iC + 3*n_grid1] = 0.0;

                for(ie = 0; ie < ne_b; ie++) {

                    jjx = jx + ie * n2;

                    kkx = kx + ie * n1;

                    tmpC[iC + 3*n_grid1] += int2_grad1_u12[jjx            ] * int2_grad1_u12[kkx            ];
                    tmpC[iC + 3*n_grid1] += int2_grad1_u12[jjx +   n_grid1] * int2_grad1_u12[kkx +   n_grid1];
                    tmpC[iC + 3*n_grid1] += int2_grad1_u12[jjx + 2*n_grid1] * int2_grad1_u12[kkx + 2*n_grid1];

                } // ie
    
            } // s_mo

        } // p_mo

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_1e_tmpC_cs(int n_grid1, int n_mo, int ne_b,
                              double * int2_grad1_u12,
                              double * tmpC) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_tmpC_cs_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_1e_tmpC_cs_kernel<<<nBlocks, blockSize>>>(n_grid1, n_mo, ne_b,
                                                 int2_grad1_u12,
                                                 tmpC);

}


