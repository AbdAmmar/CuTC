
#include <stdio.h>

__global__ void trans_pqst_psqt_inplace_kernel(int size, double * data) {

    int p, q, s, t;
    int i_new, i_old;
    int size2, size3;
    int ii, jj, kk;

    double tmp;

    p = blockIdx.x * blockDim.x + threadIdx.x;

    size2 = size * size;
    size3 = size2 * size;

    while(p < size) {

        t = blockIdx.y * blockDim.y + threadIdx.y;
        while(t < size) {

            ii = p + t * size3;

            for(s = 0; s < size; s++) {

                jj = ii + s * size2;
                kk = ii + s * size;

                for(q = 0; q < s; q++) {

                    i_new = jj + q * size;
                    i_old = kk + q * size2;

                    tmp = data[i_new];
                    data[i_new] = data[i_old];
                    data[i_old] = tmp;
                }
            }

            t += blockDim.y * gridDim.y;
        }

        p += blockDim.x * gridDim.x;
    }

}


extern "C" void trans_pqst_psqt_inplace(int size, double * data) {

    int sBlocks = 32;
    int nBlocks = (size + sBlocks - 1) / sBlocks;

    dim3 dimGrid(nBlocks, nBlocks, 1);
    dim3 dimBlock(sBlocks, sBlocks, 1);

    printf("lunching trans_pqst_psqt_inplace_kernel with %dx%d blocks and %dx%d threads/block\n", nBlocks, nBlocks, sBlocks, sBlocks);

    trans_pqst_psqt_inplace_kernel<<<dimGrid, dimBlock>>>(size, data);

}

