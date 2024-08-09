#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>






extern void checkCudaErrors(cudaError_t err, const char * msg, const char * file, int line);
extern void checkCublasErrors(cublasStatus_t status, const char * msg, const char * file, int line);


extern void no_2e_tmpO_cs(int n_grid1, int ne_b,
                          double * wr1, double * mos_l_in_r, double * mos_r_in_r,
                          double * tmpO);

extern void no_2e_tmpJ_cs(int n_grid1, int n_mo, int ne_b,
                          double * wr1, double * int2_grad1_u12,
                          double * tmpJ);


extern void no_2e_tmpA_cs(int n_grid1, int n_mo, int ne_b,
                          double * wr1, double * mos_l_in_r, double * int2_grad1_u12,
                          double * tmpA);

extern void no_2e_tmpB_cs(int n_grid1, int n_mo, int ne_b,
                          double * wr1, double * mos_r_in_r, double * int2_grad1_u12,
                          double * tmpB);

extern void no_2e_tmpC_cs(int n_grid1, int n_mo, int ne_b,
                          double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                          double * tmpJ, double * tmpO, double * tmpA, double * tmpB,
                          double * tmpC);

extern void no_2e_tmpD_cs(int n_grid1, int n_mo,
                          double * wr1, double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                          double * tmpD);

extern void trans_inplace(double * data, int size);

extern void trans_pqst_psqt_inplace(int size, double * data);






int cutc_no_2e(int n_grid1, int n_mo, int ne_a, int ne_b,
               double * h_wr1, 
               double * h_mos_l_in_r, double * h_mos_r_in_r, 
               double * h_int2_grad1_u12, 
               double * h_no_2e) {


    int n_mo2;

    size_t size_wr1;
    size_t size_mos_in_r;
    size_t size_int2;

    size_t sizeO, sizeJ;
    size_t sizeA, sizeB;
    size_t sizeC, sizeD;
    size_t sizeE;

    size_t size_2e;

    double * d_wr1;
    double * d_mos_l_in_r;
    double * d_mos_r_in_r;
    double * d_int2_grad1_u12;

    double * d_no_2e;

    double * d_tmpO;
    double * d_tmpJ;
    double * d_tmpA;
    double * d_tmpB;
    double * d_tmpC;
    double * d_tmpD;

    float time_loc;
    float ttO, ttJ;
    float ttA, ttB;
    float ttC, ttD;
    float ttE1, ttE2;
    float tt1;

    double alpha, beta;

    cudaEvent_t start_loc, stop_loc;

    checkCudaErrors(cudaEventCreate(&start_loc), "cudaEventCreate", __FILE__, __LINE__);
    checkCudaErrors(cudaEventCreate(&stop_loc), "cudaEventCreate", __FILE__, __LINE__);

    cublasHandle_t myhandle;

    checkCublasErrors(cublasCreate(&myhandle), "cublasCreate", __FILE__, __LINE__);




    n_mo2 = n_mo * n_mo;

    size_wr1 = n_grid1 * sizeof(double);
    size_mos_in_r = n_grid1 * n_mo * sizeof(double);
    size_int2 = 3 * n_grid1 * n_mo2 * sizeof(double);

    sizeO = n_grid1 * sizeof(double);
    sizeJ = 3 * n_grid1 * sizeof(double);
    sizeA = 3 * n_grid1 * n_mo * sizeof(double);
    sizeB = 3 * n_grid1 * n_mo * sizeof(double);
    sizeC = 4 * n_grid1 * n_mo2 * sizeof(double);
    sizeD = 4 * n_grid1 * n_mo2 * sizeof(double);

    size_2e = n_mo2 * n_mo2 * sizeof(double);


    checkCudaErrors(cudaMalloc((void**)&d_wr1, size_wr1), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_mos_l_in_r, size_mos_in_r), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_mos_r_in_r, size_mos_in_r), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_int2_grad1_u12, size_int2), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_tmpO, sizeO), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpJ, sizeJ), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpA, sizeA), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpB, sizeB), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpC, sizeC), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpD, sizeD), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_no_2e, size_2e), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMemcpy(d_wr1, h_wr1, size_wr1, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_mos_l_in_r, h_mos_l_in_r, size_mos_in_r, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_mos_r_in_r, h_mos_r_in_r, size_mos_in_r, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_int2_grad1_u12, h_int2_grad1_u12, size_int2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);


    // tmpO
    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    no_2e_tmpO_cs(n_grid1, ne_b,
                  d_wr1, d_mos_l_in_r, d_mos_r_in_r,
                  d_tmpO);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    ttO = time_loc;


    // tmpJ
    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    no_2e_tmpJ_cs(n_grid1, n_mo, ne_b,
                  d_wr1, d_int2_grad1_u12,
                  d_tmpJ);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    ttJ = time_loc;


    // tmpA
    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    no_2e_tmpA_cs(n_grid1, n_mo, ne_b,
                  d_wr1, d_mos_l_in_r, d_int2_grad1_u12,
                  d_tmpA);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    ttA = time_loc;


    // tmpB
    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    no_2e_tmpB_cs(n_grid1, n_mo, ne_b,
                  d_wr1, d_mos_r_in_r, d_int2_grad1_u12,
                  d_tmpB);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    ttB = time_loc;


    // tmpC
    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    no_2e_tmpC_cs(n_grid1, n_mo, ne_b,
                  d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12,
                  d_tmpJ, d_tmpO, d_tmpA, d_tmpB,
                  d_tmpC);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    ttC = time_loc;



    // tmpD
    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    no_2e_tmpD_cs(n_grid1, n_mo,
                  d_wr1, d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12,
                  d_tmpD);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    ttD = time_loc;

    checkCudaErrors(cudaFree(d_tmpO), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpJ), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpA), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpB), "cudaFree", __FILE__, __LINE__);


    // tmpE
    alpha = 0.5;
    beta = 0.0;
    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    checkCublasErrors(cublasDgemm(myhandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  n_mo2, n_mo2, 4*n_grid1,
                                  &alpha,
                                  &d_tmpC[0], 4*n_grid1,
                                  &d_tmpD[0], 4*n_grid1,
                                  &beta,
                                  &d_no_2e[0], n_mo2), "cublasDgemm", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    ttE1 = time_loc;


    // tmpE <-- tmpE + tmpE.T
    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    trans_inplace(d_no_2e, n_mo2);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    ttE2 = time_loc;


    checkCudaErrors(cudaFree(d_tmpC), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpD), "cudaFree", __FILE__, __LINE__);



    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    trans_pqst_psqt_inplace(n_mo, d_no_2e);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tt1 = time_loc;



    checkCudaErrors(cudaMemcpy(h_no_2e, d_no_2e, size_2e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);



    checkCudaErrors(cudaFree(d_no_2e), "cudaFree", __FILE__, __LINE__);

    checkCublasErrors(cublasDestroy(myhandle), "cublasDestroy", __FILE__, __LINE__);


    printf("Ellapsed time for tmpO kernel = %.3f sec\n", ttO/1000.0f);
    printf("Ellapsed time for tmpJ kernel = %.3f sec\n", ttJ/1000.0f);
    printf("Ellapsed time for tmpA kernel = %.3f sec\n", ttA/1000.0f);
    printf("Ellapsed time for tmpB kernel = %.3f sec\n", ttB/1000.0f);
    printf("Ellapsed time for tmpC kernel = %.3f sec\n", ttC/1000.0f);
    printf("Ellapsed time for tmpD kernel = %.3f sec\n", ttD/1000.0f);
    printf("Ellapsed time for tmpE DGEMM = %.3f sec\n", ttE1/1000.0f);
    printf("Ellapsed time for tmpE + tmpE.T = %.3f sec\n", ttE2/1000.0f);
    printf("Ellapsed time for pqst -> psqt = %.3f sec\n", tt1/1000.0f);



    return 0;

}
