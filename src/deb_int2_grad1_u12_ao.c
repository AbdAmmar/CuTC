#include <cuda.h>
#include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>

//typedef struct {
//    unsigned int x;
//    unsigned int y;
//    unsigned int z;
//} dim3;


extern void checkCudaErrors(cudaError_t err, const char* msg, const char* file, int line);
extern void checkCublasErrors(cublasStatus_t status, const char* msg, const char* file, int line);


extern void int_long_range(int nBlocks, int blockSize,
                           int n_grid2, int n_ao, double *wr2, double* aos_data2,
                           double *int_fct_long_range);


extern void tc_int_bh(dim3 dimGrid, dim3 dimBlock,
                      int ii0, int n_grid1_eff, int n_grid1_tot,
                      int n_grid1, int n_grid2, int n_nuc, int size_bh,
                      double *r1, double *r2, double *rn,
                      double *c_bh, int *m_bh, int *n_bh, int *o_bh,
                      double *grad1_u12);


int deb_int2_grad1_u12_ao(int nxBlocks, int nyBlocks, int nzBlocks, int blockxSize, int blockySize, int blockzSize,
                          int n_grid1, int n_grid2, int n_ao, int n_nuc, int size_bh,
                          double *h_r1, double *h_r2, double *h_wr2, double *h_rn, double *h_aos_data2,
                          double *h_c_bh, int *h_m_bh, int *h_n_bh, int *h_o_bh,
                          double *h_int2_grad1_u12_ao) {

    int ii;
    int jj0, jj;
    int kk0, kk;
    int i_pass;
    int n_grid1_pass, n_grid1_rest, n_pass;
    double n_tmp;

    int m;
    double alpha, beta;

    cublasHandle_t myhandle;

    checkCublasErrors(cublasCreate(&myhandle), "cublasCreate", __FILE__, __LINE__);


    size_t size_r1, size_r2, size_wr2, size_rn;
    size_t size_aos_r2;
    size_t size_jbh_d, size_jbh_i;
    size_t size_1, size_2, size_3;

    double *d_r1, *d_r2, *d_wr2, *d_rn;
    double *d_aos_data2;
    double *d_c_bh;
    int *d_m_bh, *d_n_bh, *d_o_bh;

    double *d_int_fct_long_range;
    double *d_grad1_u12;
    double *d_int2_grad1_u12_ao;

    dim3 dimGrid;
    dim3 dimBlock;


    cudaEvent_t start_loc, stop_loc;
    float time_loc;
    float tt0, tt1, tt2, tt3, tt4;

    struct cudaDeviceProp deviceProp;

    printf(" DEBUG int2_grad1_u12_ao\n");


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
    printf("Grid Size: (%u, %u, %u)\n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("Block Size: (%u, %u, %u)\n", dimBlock.x, dimBlock.y, dimBlock.z);


    // used for timing
    checkCudaErrors(cudaEventCreate(&start_loc), "cudaEventCreate", __FILE__, __LINE__);
    checkCudaErrors(cudaEventCreate(&stop_loc), "cudaEventCreate", __FILE__, __LINE__);




    size_r1 = 3 * n_grid1 * sizeof(double);
    size_r2 = 3 * n_grid2 * sizeof(double);
    size_wr2 = n_grid2 * sizeof(double);
    size_rn = 3 * n_nuc * sizeof(double);

    size_aos_r2 = 4 * n_grid2 * n_ao * sizeof(double);

    size_jbh_d = size_bh * n_nuc * sizeof(double);
    size_jbh_i = size_bh * n_nuc * sizeof(int);

    size_1 = n_grid2 * n_ao * n_ao * sizeof(double);
    size_3 = 4 * n_grid1 * n_ao * n_ao * sizeof(double);


    checkCudaErrors(cudaMalloc((void**)&d_r1, size_r1), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_r2, size_r2), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_wr2, size_wr2), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_rn, size_rn), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_aos_data2, size_aos_r2), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_c_bh, size_jbh_d), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_m_bh, size_jbh_i), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_n_bh, size_jbh_i), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_o_bh, size_jbh_i), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_int_fct_long_range, size_1), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_int2_grad1_u12_ao, size_3), "cudaMalloc", __FILE__, __LINE__);


    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_r1, h_r1, size_r1, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_r2, h_r2, size_r2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_wr2, h_wr2, size_wr2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_rn, h_rn, size_rn, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_aos_data2, h_aos_data2, size_aos_r2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_c_bh, h_c_bh, size_jbh_d, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_m_bh, h_m_bh, size_jbh_i, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_n_bh, h_n_bh, size_jbh_i, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_o_bh, h_o_bh, size_jbh_i, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tt0 = time_loc;


    // // //

    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);

    int_long_range(nxBlocks, blockxSize, n_grid2, n_ao, d_wr2, d_aos_data2, d_int_fct_long_range);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

    checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tt1 = time_loc;

    // // //





    // // //

    jj0 = n_ao * n_ao;

    alpha = 1.0;
    beta = 0.0;

    //       amount in GB
    // TODO get available memory
    // cudaMemGetInfo
    n_tmp = (5.0e9 / 8.0) / (4.0 * (double) n_grid2);
    if(n_tmp < 1.0*n_grid1) {
        if(n_tmp > 1.0) {
            n_grid1_pass = (int) n_tmp;
        } else {
            n_grid1_pass = 1;
        }
    } else {
        n_grid1_pass = n_grid1;
    }

    n_grid1_rest = (int) fmod(1.0 * n_grid1, 1.0 * n_grid1_pass);
    n_pass = (int) ((n_grid1 - n_grid1_rest) / n_grid1_pass);

    printf("n_grid1_pass = %d\n", n_grid1_pass);
    printf("n_grid1_rest = %d\n", n_grid1_rest);
    printf("n_pass = %d\n", n_pass);
    
    size_2 = 4 * n_grid1_pass * n_grid2 * sizeof(double);
    kk0 = n_grid1_pass * n_grid2;

    checkCudaErrors(cudaMalloc((void**)&d_grad1_u12, size_2), "cudaMalloc", __FILE__, __LINE__);


    tt2 = 0.0f;
    tt3 = 0.0f;
    for (i_pass = 0; i_pass < n_pass; i_pass++) {

        ii = i_pass * n_grid1_pass;

        checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        tc_int_bh(dimGrid, dimBlock,
                  ii, n_grid1_pass, n_grid1_pass,
                  n_grid1, n_grid2, n_nuc, size_bh,
                  d_r1, d_r2, d_rn,
                  d_c_bh, d_m_bh, d_n_bh, d_o_bh,
                  d_grad1_u12);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tt2 += time_loc;


        checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        for (m = 0; m < 4; m++) {
            jj = jj0 * (ii + m * n_grid1);
            kk = kk0 * m;
            checkCublasErrors( cublasDgemm( myhandle
                                          , CUBLAS_OP_T, CUBLAS_OP_N
                                          , n_ao*n_ao, n_grid1_pass, n_grid2
                                          , &alpha
                                          , &d_int_fct_long_range[0], n_grid2
                                          , &d_grad1_u12[kk], n_grid2
                                          , &beta
                                          , &d_int2_grad1_u12_ao[jj], n_ao*n_ao )
                             , "cublasDgemm", __FILE__, __LINE__);
        }
        checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tt3 += time_loc;

    }

    if(n_grid1_rest > 0) {

        ii = n_pass * n_grid1_pass;

        checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        tc_int_bh(dimGrid, dimBlock,
                  ii, n_grid1_rest, n_grid1_pass,
                  n_grid1, n_grid2, n_nuc, size_bh,
                  d_r1, d_r2, d_rn,
                  d_c_bh, d_m_bh, d_n_bh, d_o_bh,
                  d_grad1_u12);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tt2 += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        for (m = 0; m < 4; m++) {
            jj = jj0 * (ii + m * n_grid1);
            kk = kk0 * m;
            checkCublasErrors( cublasDgemm( myhandle
                                          , CUBLAS_OP_T, CUBLAS_OP_N
                                          , n_ao*n_ao, n_grid1_rest, n_grid2
                                          , &alpha
                                          , &d_int_fct_long_range[0], n_grid2
                                          , &d_grad1_u12[kk], n_grid2
                                          , &beta
                                          , &d_int2_grad1_u12_ao[jj], n_ao*n_ao )
                             , "cublasDgemm", __FILE__, __LINE__);
        }
        checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tt3 += time_loc;

    }


    // // //



    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_int2_grad1_u12_ao, d_int2_grad1_u12_ao, size_3, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tt4 = time_loc;

    printf("Ellapsed time for transfer data (CPU -> GPU) = %.3f sec\n", tt0/1000.0f);
    printf("Ellapsed time for int_long_range kernel = %.3f sec\n", tt1/1000.0f);
    printf("Ellapsed time for tc_int_bh kernel = %.3f sec\n", tt2/1000.0f);
    printf("Ellapsed time for cublas DGEMM = %.3f sec\n", tt3/1000.0f);
    printf("Ellapsed time for transfer data (GPU -> CPU) = %.3f sec\n", tt4/1000.0f);
    printf("Ellapsed time on GPU = %.3f sec\n", (tt0 + tt1 + tt2 + tt3 + tt4)/1000.0f);

    checkCudaErrors(cudaFree(d_r1), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_r2), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_wr2), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_rn), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_aos_data2), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_c_bh), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_m_bh), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_n_bh), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_o_bh), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_int_fct_long_range), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_grad1_u12), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_int2_grad1_u12_ao), "cudaFree", __FILE__, __LINE__);

    checkCublasErrors(cublasDestroy(myhandle), "cublasDestroy", __FILE__, __LINE__);

    return 0;
}








