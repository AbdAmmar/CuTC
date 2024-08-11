
#include <stdio.h>

__global__ void no_0e_tmpE_kernel(int n_grid1, int nBlocks, int blockSize,
                                  double * wr1, double * tmpO, double * tmpS, double * tmpJ, double * tmpM,
                                  double * tmpE) {


    extern __shared__ double cache[];


    int i_grid1;

    int i, cacheIndex;

    double tmpE_loc;
    double wr1_tmp;
    double O, S;
    double Jx, Jy, Jz;
    double Mx, My, Mz;

    tmpE_loc = 0.0;

    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    while(i_grid1 < n_grid1) {

        wr1_tmp = wr1[i_grid1];

        O = tmpO[i_grid1];
        S = tmpS[i_grid1];

        Jx = tmpJ[i_grid1            ];
        Jy = tmpJ[i_grid1 +   n_grid1];
        Jz = tmpJ[i_grid1 + 2*n_grid1];

        Mx = tmpM[i_grid1            ];
        My = tmpM[i_grid1 +   n_grid1];
        Mz = tmpM[i_grid1 + 2*n_grid1];

        tmpE_loc = wr1_tmp * (O * S - 2.0 * (Jx*Mx + Jy*My + Jz*Mz));

    }

    cacheIndex = threadIdx.x;

    cache[cacheIndex] = tmpE_loc;
    __syncthreads();

    i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += tmpE_loc;
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        tmpE[blockIdx.x] = cache[0];
    }


}



extern "C" void no_0e_tmpE(int n_grid1, int nBlocks, int blockSize,
                           double * wr1, double * tmpO, double * tmpS, double * tmpJ, double * tmpM,
                           double * tmpE) {

    printf("lunching no_0e_tmpE_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_0e_tmpE_kernel<<<nBlocks, blockSize>>>(n_grid1, nBlocks, blockSize,
                                              wr1, tmpO, tmpS, tmpJ, tmpM,
                                              tmpE);

}


