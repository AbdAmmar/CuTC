
#include <stdio.h>

__global__ void trans_pqkm_pqmk_copy_kernel(int size, int n_grid1, double * data_old, double * data_new) {

    int p, q;
    int i_grid1, m;
    int i_new, i_old;
    int size2;
    int dim1, dim2;
    int ii, jj, kk;

    p = blockIdx.x * blockDim.x + threadIdx.x;

    size2 = size * size;

    dim1 = 4 * size2;
    dim2 = n_grid1 * size2;

    while(p < size) {

        q = blockIdx.y * blockDim.y + threadIdx.y;
        while(q < size) {

            ii = p + q * size;

            for(i_grid1 = 0; i_grid1 < n_grid1; i_grid1++) {

                jj = ii + i_grid1 * size2;
                kk = ii + i_grid1 * dim1;

                for(m = 0; m < 4; m++) {

                    i_new = jj + m * dim2;
                    i_old = kk + m * size2;

                    data_new[i_new] = data_old[i_old];
                }
            }

            q += blockDim.y * gridDim.y;
        }

        p += blockDim.x * gridDim.x;
    }

}


extern "C" void trans_pqkm_pqmk_copy(int size, int n_grid1, double * data_old, double * data_new) {

    int sBlocks = 32;
    int nBlocks = (size + sBlocks - 1) / sBlocks;

    dim3 dimGrid(nBlocks, nBlocks, 1);
    dim3 dimBlock(sBlocks, sBlocks, 1);

    printf("lunching trans_pqkm_pqmk_copy_kernel with %dx%d blocks and %dx%d threads/block\n", nBlocks, nBlocks, sBlocks, sBlocks);

    trans_pqkm_pqmk_copy_kernel<<<dimGrid, dimBlock>>>(size, n_grid1, data_old, data_new);

}

