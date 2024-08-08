#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>


extern void checkCudaErrors(cudaError_t err, const char* msg, const char* file, int line);
extern void checkCublasErrors(cublasStatus_t status, const char* msg, const char* file, int line);



extern void get_int2_grad1_u12_ao(dim3 dimGrid, dim3 dimBlock,
                                  int n_grid1, int n_grid2, int n_ao, int n_nuc, int size_bh,
                                  double *r1, double *r2, double *wr2, double *rn, double *aos_data2,
                                  double *c_bh, int *m_bh, int *n_bh, int *o_bh,
                                  double *int2_grad1_u12_ao);

extern void get_int_2e_ao(int n_grid1, int n_ao, double *wr1, double *aos_data1,
                          double *int2_grad1_u12_ao, double *tc_int_2e_ao);



int tc_int_c(int nxBlocks, int nyBlocks, int nzBlocks, int blockxSize, int blockySize, int blockzSize,
             int n_grid1, int n_grid2, int n_ao, int n_nuc, int size_bh,
             double * h_r1, double * h_wr1, double * h_r2, double * h_wr2, double * h_rn,
             double * h_aos_data1, double * h_aos_data2,
             double * h_c_bh, int * h_m_bh, int * h_n_bh, int * h_o_bh, 
             double * h_int2_grad1_u12_ao, double * h_int_2e_ao) {


    double *d_r1, *d_wr1;
    double *d_r2, *d_wr2;
    double *d_rn;

    double *d_aos_data1, *d_aos_data2;

    double *d_c_bh; 
    int *d_m_bh, *d_n_bh, *d_o_bh;

    double *d_int2_grad1_u12_ao;
    double *d_int_2e_ao;

    size_t size_r1, size_wr1, size_r2, size_wr2, size_rn;
    size_t size_aos_r1, size_aos_r2;
    size_t size_int1, size_int2, size_int1_send;
    size_t size_jbh_d, size_jbh_i;

    struct cudaDeviceProp deviceProp;

    dim3 dimGrid;
    dim3 dimBlock;


    printf(" Computing TC-Integrals With CuTC\n");


    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max block dimensions: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("Max grid dimensions: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

    dimGrid.x = nxBlocks;
    dimGrid.y = nyBlocks;
    dimGrid.z = nzBlocks;
    dimBlock.x = blockxSize;
    dimBlock.y = blockySize;
    dimBlock.z = blockzSize;

    if(dimBlock.x * dimBlock.y * dimBlock.z > deviceProp.maxThreadsPerBlock) {
        printf("Error: Too many threads per block!\n");
        return -1;
    }
    if(dimGrid.x > deviceProp.maxGridSize[0] || dimGrid.y > deviceProp.maxGridSize[1] || dimGrid.z > deviceProp.maxGridSize[2]) {
        printf("Error: Grid dimensions exceed device capabilities!\n");
        return -1;
    }
    //printf("Grid Size: (%u, %u, %u)\n", dimGrid.x, dimGrid.y, dimGrid.z);
    //printf("Block Size: (%u, %u, %u)\n", dimBlock.x, dimBlock.y, dimBlock.z);





    size_r1 = 3 * n_grid1 * sizeof(double);
    size_r2 = 3 * n_grid2 * sizeof(double);
    size_wr2 = n_grid2 * sizeof(double);
    size_rn = 3 * n_nuc * sizeof(double);

    size_aos_r2 = 4 * n_grid2 * n_ao * sizeof(double);

    size_jbh_d = size_bh * n_nuc * sizeof(double);
    size_jbh_i = size_bh * n_nuc * sizeof(int);

    size_int1 = 4 * n_grid1 * n_ao * n_ao * sizeof(double);
    size_int1_send = 3 * n_grid1 * n_ao * n_ao * sizeof(double);



    checkCudaErrors(cudaMalloc((void**)&d_r1, size_r1), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_r2, size_r2), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_wr2, size_wr2), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_rn, size_rn), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_aos_data2, size_aos_r2), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_c_bh, size_jbh_d), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_m_bh, size_jbh_i), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_n_bh, size_jbh_i), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_o_bh, size_jbh_i), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_int2_grad1_u12_ao, size_int1), "cudaMalloc", __FILE__, __LINE__);




    checkCudaErrors(cudaMemcpy(d_r1, h_r1, size_r1, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_r2, h_r2, size_r2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_wr2, h_wr2, size_wr2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_rn, h_rn, size_rn, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);

    checkCudaErrors(cudaMemcpy(d_aos_data2, h_aos_data2, size_aos_r2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);

    checkCudaErrors(cudaMemcpy(d_c_bh, h_c_bh, size_jbh_d, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_m_bh, h_m_bh, size_jbh_i, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_n_bh, h_n_bh, size_jbh_i, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_o_bh, h_o_bh, size_jbh_i, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);


    get_int2_grad1_u12_ao(dimGrid, dimBlock, 
                          n_grid1, n_grid2, n_ao, n_nuc, size_bh,
                          d_r1, d_r2, d_wr2, d_rn, d_aos_data2,
                          d_c_bh, d_m_bh, d_n_bh, d_o_bh,
                          d_int2_grad1_u12_ao);


    checkCudaErrors(cudaFree(d_r1), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_r2), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_wr2), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_rn), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_aos_data2), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_c_bh), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_m_bh), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_n_bh), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_o_bh), "cudaFree", __FILE__, __LINE__);

    // // //



    // 2-e integral

    size_wr1 = n_grid1 * sizeof(double);
    size_aos_r1 = 4 * n_grid1 * n_ao * sizeof(double);
    size_int2 = n_ao * n_ao * n_ao * n_ao * sizeof(double);

    checkCudaErrors(cudaMalloc((void**)&d_wr1, size_wr1), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_aos_data1, size_aos_r1), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_int_2e_ao, size_int2), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMemcpy(d_wr1, h_wr1, size_wr1, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_aos_data1, h_aos_data1, size_aos_r1, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);

    get_int_2e_ao(n_grid1, n_ao, d_wr1, d_aos_data1, d_int2_grad1_u12_ao, d_int_2e_ao);

    checkCudaErrors(cudaFree(d_wr1), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_aos_data1), "cudaFree", __FILE__, __LINE__);

    // // //



    // transfer data to Host

    checkCudaErrors(cudaMemcpy(h_int2_grad1_u12_ao, d_int2_grad1_u12_ao, size_int1_send, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_int_2e_ao, d_int_2e_ao, size_int2, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_int2_grad1_u12_ao), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_int_2e_ao), "cudaFree", __FILE__, __LINE__);

    // // //

    printf(" Done ;)\n");
    return 0;
}


