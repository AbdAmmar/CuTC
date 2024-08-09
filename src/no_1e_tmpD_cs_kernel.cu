
#include <stdio.h>

__global__ void no_1e_tmpD_cs_kernel(int n_grid1,
                                     double * wr1, double * tmpO, double * tmpJ, double * tmpM,
                                     double * tmpD) {


    int i_grid1;

    double wr1_tmp;
    double O;
    double Jx, Jy, Jz;
    double Mx, My, Mz;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    while(i_grid1 < n_grid1) {

        wr1_tmp = wr1[i_grid1];

        O = tmpO[i_grid1];

        Jx = tmpJ[i_grid1            ];
        Jy = tmpJ[i_grid1 +   n_grid1];
        Jz = tmpJ[i_grid1 + 2*n_grid1];

        Mx = tmpM[i_grid1            ];
        My = tmpM[i_grid1 +   n_grid1];
        Mz = tmpM[i_grid1 + 2*n_grid1];

        tmpD[i_grid1            ] = wr1_tmp * (2.0 * O * Jx - Mx);
        tmpD[i_grid1 +   n_grid1] = wr1_tmp * (2.0 * O * Jy - My);
        tmpD[i_grid1 + 2*n_grid1] = wr1_tmp * (2.0 * O * Jz - Mz);
        tmpD[i_grid1 + 3*n_grid1] = -wr1_tmp * O;

        i_grid1 += blockDim.x * gridDim.x;

    }

}



extern "C" void no_1e_tmpD_cs(int n_grid1,
                              double * wr1, double * tmpO, double * tmpJ, double * tmpM,
                              double * tmpD) {

    int nBlocks, blockSize;

    blockSize = 32;
    nBlocks = (n_grid1 + blockSize - 1) / blockSize;

    printf("lunching no_1e_tmpD_cs_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_1e_tmpD_cs_kernel<<<nBlocks, blockSize>>>(n_grid1,
                                                 wr1, tmpO, tmpJ, tmpM,
                                                 tmpD);

}


