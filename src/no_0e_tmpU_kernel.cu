
#include <stdio.h>

__global__ void no_0e_tmpU_kernel(int n_grid1,
                                  double * wr1, double * tmpO, double * tmpS, double * tmpJ, double * tmpM,
                                  double * tmpU) {


    extern __shared__ double cache[];


    int i_grid1;

    int i, cacheIndex;

    double tmpU_loc;
    double wr1_tmp;
    double O, S;
    double Jx, Jy, Jz;
    double Mx, My, Mz;

    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;
    cacheIndex = threadIdx.x;
    tmpU_loc = 0.0;


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

        tmpU_loc += wr1_tmp * (O * S - 2.0 * (Jx*Mx + Jy*My + Jz*Mz));

        i_grid1 += blockDim.x * gridDim.x;

    }

    cache[cacheIndex] = tmpU_loc;
    __syncthreads();

    i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        tmpU[blockIdx.x] = cache[0];
    }


}



extern "C" void no_0e_tmpU(int n_grid1, int nBlocks, int blockSize,
                           double * wr1, double * tmpO, double * tmpS, double * tmpJ, double * tmpM,
                           double * tmpU) {

    printf("lunching no_0e_tmpU_kernel with %d blocks and %d threads/block\n", nBlocks, blockSize);

    no_0e_tmpU_kernel<<<nBlocks, blockSize, blockSize*sizeof(double)>>>(n_grid1,
                                                                        wr1, tmpO, tmpS, tmpJ, tmpM,
                                                                        tmpU);

}


