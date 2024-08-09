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

extern void no_2e_tmpO_os(int n_grid1, int ne_b, int ne_a,
                          double * wr1, double * mos_l_in_r, double * mos_r_in_r,
                          double * tmpO);

extern void no_2e_tmpJ_cs(int n_grid1, int n_mo, int ne_b,
                          double * wr1, double * int2_grad1_u12,
                          double * tmpJ);

extern void no_2e_tmpJ_os(int n_grid1, int n_mo, int ne_b, int ne_a,
                          double * wr1, double * int2_grad1_u12,
                          double * tmpJ);

extern void no_2e_tmpA_cs(int n_grid1, int n_mo, int ne_b,
                          double * wr1, double * mos_l_in_r, double * int2_grad1_u12,
                          double * tmpA);

extern void no_2e_tmpA_os(int n_grid1, int n_mo, int ne_b, int ne_a,
                          double * wr1, double * mos_l_in_r, double * int2_grad1_u12,
                          double * tmpA);

extern void no_2e_tmpB_cs(int n_grid1, int n_mo, int ne_b,
                          double * wr1, double * mos_r_in_r, double * int2_grad1_u12,
                          double * tmpB);

extern void no_2e_tmpB_os(int n_grid1, int n_mo, int ne_b, int ne_a,
                          double * wr1, double * mos_r_in_r, double * int2_grad1_u12,
                          double * tmpB);

extern void no_2e_tmpC_cs(int n_grid1, int n_mo, int ne_b,
                          double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                          double * tmpJ, double * tmpO, double * tmpA, double * tmpB,
                          double * tmpC);

extern void no_2e_tmpC_os(int n_grid1, int n_mo, int ne_b, int ne_a,
                          double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                          double * tmpJ, double * tmpO, double * tmpA, double * tmpB,
                          double * tmpC);


extern void no_2e_tmpC1(int n_grid1, int n_mo,
                        double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                        double * tmpJ, double * tmpO, double * tmpA, double * tmpB,
                        double * tmpC1);


extern void no_tmpC2_cs(int n_grid1, int n_mo, int ne_b,
                        double * int2_grad1_u12,
                        double * tmpC2);

extern void no_tmpC2_os(int n_grid1, int n_mo, int ne_b, int ne_a,
                        double * int2_grad1_u12,
                        double * tmpC2);

extern void no_2e_tmpD(int n_grid1, int n_mo,
                       double * wr1, double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                       double * tmpD);

extern void no_2e_tmpD2(int n_grid1, int n_mo,
                        double * wr1, double * mos_l_in_r, double * mos_r_in_r,
                        double * tmpD2);

extern void trans_inplace(double * data, int size);

extern void trans_pqst_psqt_copy(int size, double * data_old, double * data_new);






int deb_no_2e(int n_grid1, int n_mo, int ne_a, int ne_b,
              double * h_wr1, double * h_mos_l_in_r, double * h_mos_r_in_r, double * h_int2_grad1_u12, 
              double * h_tmpO, double * h_tmpJ, double * h_tmpA, double * h_tmpB, double * h_tmpC, double * h_tmpD, double * h_tmpE,
              double * h_no_2e) {


    int n_mo2;

    size_t size_wr1;
    size_t size_mos_in_r;
    size_t size_int2;

    size_t sizeO, sizeJ;
    size_t sizeA, sizeB;
    size_t sizeD;
    size_t sizeD2;
    size_t sizeC;
    size_t sizeC1;
    size_t sizeC2;
    size_t sizeE;
    size_t size_2e;

    double * d_wr1;
    double * d_mos_l_in_r;
    double * d_mos_r_in_r;
    double * d_int2_grad1_u12;

    double * d_tmpO;
    double * d_tmpJ;
    double * d_tmpA;
    double * d_tmpB;
    double * d_tmpC1;
    double * d_tmpC2;
    double * d_tmpC;
    double * d_tmpD;
    double * d_tmpD2;
    double * d_tmpE;
    double * d_no_2e;

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
    sizeC1 = 3 * n_grid1 * n_mo2 * sizeof(double);
    sizeC2 = n_grid1 * n_mo2 * sizeof(double);
    sizeD = 4 * n_grid1 * n_mo2 * sizeof(double);
    sizeD2 = n_grid1 * n_mo2 * sizeof(double);
    sizeE = n_mo2 * n_mo2 * sizeof(double);
    size_2e = n_mo2 * n_mo2 * sizeof(double);



    checkCudaErrors(cudaMalloc((void**)&d_wr1, size_wr1), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_mos_l_in_r, size_mos_in_r), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_mos_r_in_r, size_mos_in_r), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_int2_grad1_u12, size_int2), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_tmpO, sizeO), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpJ, sizeJ), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpA, sizeA), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpB, sizeB), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpC1, sizeC1), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpC2, sizeC2), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpC, sizeC), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpD, sizeD), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpD2, sizeD2), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpE, sizeE), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_no_2e, size_2e), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMemcpy(d_wr1, h_wr1, size_wr1, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_mos_l_in_r, h_mos_l_in_r, size_mos_in_r, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_mos_r_in_r, h_mos_r_in_r, size_mos_in_r, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_int2_grad1_u12, h_int2_grad1_u12, size_int2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);


    // tmpO
    if(ne_a == ne_b) {
        checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        no_2e_tmpO_cs(n_grid1, ne_b,
                      d_wr1, d_mos_l_in_r, d_mos_r_in_r,
                      d_tmpO);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        ttO = time_loc;
    } else {
        checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        no_2e_tmpO_os(n_grid1, ne_b, ne_a,
                      d_wr1, d_mos_l_in_r, d_mos_r_in_r,
                      d_tmpO);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        ttO = time_loc;
    }


    // tmpJ
    if(ne_a == ne_b) {
        checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        no_2e_tmpJ_cs(n_grid1, n_mo, ne_b,
                      d_wr1, d_int2_grad1_u12,
                      d_tmpJ);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        ttJ = time_loc;
    } else {
        checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        no_2e_tmpJ_os(n_grid1, n_mo, ne_b, ne_a,
                      d_wr1, d_int2_grad1_u12,
                      d_tmpJ);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        ttJ = time_loc;
    }

    // tmpA
    if(ne_a == ne_b) {
        checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        no_2e_tmpA_cs(n_grid1, n_mo, ne_b,
                      d_wr1, d_mos_l_in_r, d_int2_grad1_u12,
                      d_tmpA);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        ttA = time_loc;
    } else {
        checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        no_2e_tmpA_os(n_grid1, n_mo, ne_b, ne_a,
                      d_wr1, d_mos_l_in_r, d_int2_grad1_u12,
                      d_tmpA);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        ttA = time_loc;
    }

    // tmpB
    if(ne_a == ne_b) {
        checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        no_2e_tmpB_cs(n_grid1, n_mo, ne_b,
                      d_wr1, d_mos_r_in_r, d_int2_grad1_u12,
                      d_tmpB);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        ttB = time_loc;
    } else {
        checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        no_2e_tmpB_os(n_grid1, n_mo, ne_b, ne_a,
                      d_wr1, d_mos_r_in_r, d_int2_grad1_u12,
                      d_tmpB);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        ttB = time_loc;
    }

    // tmpC
    if(ne_a == ne_b) {
        no_2e_tmpC_cs(n_grid1, n_mo, ne_b,
                      d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12,
                      d_tmpJ, d_tmpO, d_tmpA, d_tmpB,
                      d_tmpC);
        checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        no_2e_tmpC1(n_grid1, n_mo,
                    d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12,
                    d_tmpJ, d_tmpO, d_tmpA, d_tmpB,
                    d_tmpC1);
        no_tmpC2_cs(n_grid1, n_mo, ne_b, d_int2_grad1_u12, d_tmpC2);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        ttC = time_loc;
    } else {
        no_2e_tmpC_os(n_grid1, n_mo, ne_b, ne_a,
                      d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12,
                      d_tmpJ, d_tmpO, d_tmpA, d_tmpB,
                      d_tmpC);
        checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        no_2e_tmpC1(n_grid1, n_mo,
                    d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12,
                    d_tmpJ, d_tmpO, d_tmpA, d_tmpB,
                    d_tmpC1);
        no_tmpC2_os(n_grid1, n_mo, ne_b, ne_a, d_int2_grad1_u12, d_tmpC2);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        ttC = time_loc;
    }

    // tmpD
    no_2e_tmpD(n_grid1, n_mo,
               d_wr1, d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12,
               d_tmpD);
    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    no_2e_tmpD2(n_grid1, n_mo,
                d_wr1, d_mos_l_in_r, d_mos_r_in_r,
                d_tmpD2);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    ttD = time_loc;


    // tmpE
    alpha = 0.5;
    beta = 0.0;
    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    //checkCublasErrors(cublasDgemm(myhandle,
    //                              CUBLAS_OP_T, CUBLAS_OP_N,
    //                              n_mo2, n_mo2, 4*n_grid1,
    //                              &alpha,
    //                              &d_tmpC[0], 4*n_grid1,
    //                              &d_tmpD[0], 4*n_grid1,
    //                              &beta,
    //                              &d_tmpE[0], n_mo2), "cublasDgemm", __FILE__, __LINE__);
    checkCublasErrors(cublasDgemm(myhandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  n_mo2, n_mo2, 3*n_grid1,
                                  &alpha,
                                  &d_tmpC1[0], 3*n_grid1,
                                  &d_int2_grad1_u12[0], 3*n_grid1,
                                  &beta,
                                  &d_tmpE[0], n_mo2), "cublasDgemm", __FILE__, __LINE__);
    beta = 1.0;
    checkCublasErrors(cublasDgemm(myhandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  n_mo2, n_mo2, n_grid1,
                                  &alpha,
                                  &d_tmpC2[0], n_grid1,
                                  &d_tmpD2[0], n_grid1,
                                  &beta,
                                  &d_tmpE[0], n_mo2), "cublasDgemm", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    ttE1 = time_loc;


    // tmpE <-- tmpE + tmpE.T
    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    trans_inplace(d_tmpE, n_mo2);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    ttE2 = time_loc;





    checkCudaErrors(cudaEventRecord(start_loc, NULL), "cudaEventRecord", __FILE__, __LINE__);
    trans_pqst_psqt_copy(n_mo, d_tmpE, d_no_2e);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tt1 = time_loc;




    checkCudaErrors(cudaMemcpy(h_tmpO, d_tmpO, sizeO, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_tmpJ, d_tmpJ, sizeJ, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_tmpA, d_tmpA, sizeA, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_tmpB, d_tmpB, sizeB, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_tmpC, d_tmpC, sizeC, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_tmpD, d_tmpD, sizeD, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_tmpE, d_tmpE, sizeE, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_no_2e, d_no_2e, size_2e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);


    checkCudaErrors(cudaFree(d_tmpO), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpJ), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpA), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpB), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpC), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpC1), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpC2), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpD), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpD2), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpE), "cudaFree", __FILE__, __LINE__);
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
    printf("Ellapsed time on GPU = %.3f sec\n", (ttO + ttJ + ttA + ttB + ttC + ttD + ttE1 + ttE2 + tt1)/1000.0f);


    return 0;

}
