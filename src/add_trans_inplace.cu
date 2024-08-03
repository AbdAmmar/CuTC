
__global__ void trans_inplace_kernel(double *data, int size) {

    int i, j;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    while(i < size) {

        j = blockIdx.y * blockDim.y + threadIdx.y;
        while(j <= i) {

            data[i + size*j] += data[j + i*size];
            data[j + i*size] = data[i + size*j];

            j += blockDim.y * gridDim.y;
        }

        i += blockDim.x * gridDim.x;
    }

}


extern "C" void trans_inplace(double *data, int size) {

    int sBlocks = 32;
    int nBlocks = (size + sBlocks - 1) / sBlocks;

    dim3 dimGrid(nBlocks, nBlocks, 1);
    dim3 dimBlock(sBlocks, sBlocks, 1);

    trans_inplace_kernel<<<dimGrid, dimBlock>>>(data, size);

}

