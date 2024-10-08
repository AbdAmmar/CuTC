
#include <stdio.h>

__global__ void no_tmpC2_os_kernel(int n_grid1, int n_mo, int ne_b, int ne_a,
                                   double * int2_grad1_u12,
                                   double * tmpC2) {


    int i_grid1;
    int ie;
    int p_mo;
    int s_mo;

    int ix, iix;
    int jx, jjx;
    int kx, kkx;

    int n1, n2;
    int m2;

    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    n1 = 3 * n_grid1;
    n2 = n1 * n_mo;

    m2 = n_grid1 * n_mo;

    while(i_grid1 < n_grid1) {

        p_mo = blockIdx.y * blockDim.y + threadIdx.y;
        while(p_mo < n_mo) {

            ix = i_grid1 + p_mo * n_grid1;

            jx = i_grid1 + p_mo * n1;

            for(s_mo = 0; s_mo < n_mo; s_mo++) {

                iix = ix + s_mo * m2;

                jjx = jx + s_mo * n2;

                kx = i_grid1 + s_mo * n2;

                tmpC2[iix] = 0.0;

                for(ie = 0; ie < ne_b; ie++) {

                    jjx = jx + ie * n2;

                    kkx = kx + ie * n1;

                    tmpC2[iix] += int2_grad1_u12[jjx            ] * int2_grad1_u12[kkx            ];
                    tmpC2[iix] += int2_grad1_u12[jjx +   n_grid1] * int2_grad1_u12[kkx +   n_grid1];
                    tmpC2[iix] += int2_grad1_u12[jjx + 2*n_grid1] * int2_grad1_u12[kkx + 2*n_grid1];

                } // ie

                for(ie = ne_b; ie < ne_a; ie++) {

                    jjx = jx + ie * n2;

                    kkx = kx + ie * n1;

                    tmpC2[iix] += 0.5 * int2_grad1_u12[jjx            ] * int2_grad1_u12[kkx            ];
                    tmpC2[iix] += 0.5 * int2_grad1_u12[jjx +   n_grid1] * int2_grad1_u12[kkx +   n_grid1];
                    tmpC2[iix] += 0.5 * int2_grad1_u12[jjx + 2*n_grid1] * int2_grad1_u12[kkx + 2*n_grid1];

                } // ie
    
            } // s_mo

            p_mo += blockDim.y * gridDim.y;

        } // p_mo

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_tmpC2_os(int n_grid1, int n_mo, int ne_b, int ne_a,
                            double * int2_grad1_u12,
                            double * tmpC2) {

    int nxBlocks, blockxSize;
    int nyBlocks, blockySize;

    blockxSize = 32;
    nxBlocks = (n_grid1 + blockxSize - 1) / blockxSize;

    blockySize = 32;
    nyBlocks = (n_mo + blockySize - 1) / blockySize;

    dim3 dimGrid(nxBlocks, nyBlocks, 1);
    dim3 dimBlock(blockxSize, blockySize, 1);

    printf("lunching no_tmpC_os_kernel with %dx%d blocks and %dx%d threads/block\n", nxBlocks, nyBlocks, blockxSize, blockySize);

    no_tmpC2_os_kernel<<<dimGrid, dimBlock>>>(n_grid1, n_mo, ne_b, ne_a,
                                              int2_grad1_u12,
                                              tmpC2);


}


