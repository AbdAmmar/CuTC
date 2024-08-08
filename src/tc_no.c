#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>






extern void checkCudaErrors(cudaError_t err, const char* msg, const char* file, int line);

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
                          double * wr1, double * mos_l_in_r, double * int2_grad1_u12,
                          double * tmpB);

extern void no_2e_tmpC_cs(int n_grid1, int n_mo, int ne_b,
                          double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                          double * tmpJ, double * tmpO, double * tmpA, double * tmpB,
                          double * tmpC);

extern void no_2e_tmpD_cs(int n_grid1, int n_mo,
                          double * wr1, double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                          double * tmpD);





int cutc_no_elements(int n_grid1, int n_mo, int ne_a, int ne_b,
                     double * h_wr1, 
                     double * h_mos_l_in_r, double * h_mos_r_in_r, 
                     double * h_int2_grad1_u12, 
                     double * h_no_2e, double * h_no_1e, double * h_no_0e) {


    int n_mo2;

    size_t size_wr1;
    size_t size_mos_in_r;
    size_t size_int2;

    size_t sizeO, sizeJ;
    size_t sizeA, sizeB;
    size_t sizeC, sizeD;
    size_t sizeE;

    size_t size_0e;
    size_t size_1e;
    size_t size_2e;

    double * d_wr1;
    double * d_mos_l_in_r;
    double * d_mos_r_in_r;
    double * d_int2_grad1_u12;

    double * d_no_0e;
    double * d_no_1e;
    double * d_no_2e;

    double * d_tmpO;
    double * d_tmpJ;
    double * d_tmpA;
    double * d_tmpB;
    double * d_tmpC;
    double * d_tmpD;
    double * d_tmpE;

    cudaEvent_t start_loc, stop_loc;
    float time_loc;
    float ttO, ttJ, ttA, ttB, ttC, ttD, ttE;







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
    sizeE = n_mo2 * n_mo2 * sizeof(double);

    size_0e = sizeof(double);
    size_1e = n_mo2 * sizeof(double);
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
                  d_wr1, d_mos_l_in_r, d_int2_grad1_u12,
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












    checkCudaErrors(cudaMemcpy(h_no_2e, d_no_2e, size_2e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_no_1e, d_no_1e, size_2e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_no_0e, d_no_0e, size_2e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);

    return 0;

}
