#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <time.h>



#include "cutc_no.h"



int cutc_no(int n_grid1, int n_mo, int ne_a, int ne_b,
            double * h_wr1, double * h_mos_l_in_r, double * h_mos_r_in_r, double * h_int2_grad1_u12, 
            double * h_no_2e, double * h_no_1e, double * h_no_0e) {


    int n_mo2;

    // input
    size_t size_wr1, size_mos_in_r, size_int2;
    double * d_wr1, * d_mos_l_in_r, * d_mos_r_in_r, * d_int2_grad1_u12;

    // output
    size_t size_2e, size_1e, size_0e;
    double * d_no_2e, * d_no_1e, * d_no_0e;

    // useful tensors
    size_t sizeO, sizeJ, sizeM, sizeS;
    double * d_tmpO, * d_tmpJ, * d_tmpM, * d_tmpS;

    // 2e-tensors
    size_t sizeA, sizeB, sizeC1, sizeC2, sizeD2;
    double * d_tmpA, * d_tmpB, * d_tmpC1, * d_tmpC2, * d_tmpD2;

    // 1e-tensors
    size_t sizeD, sizeL, sizeR, sizeE, sizeF, sizeG, sizeH;
    double * d_tmpD, * d_tmpL, * d_tmpR, * d_tmpE, * d_tmpF, * d_tmpG, * d_tmpH;

    // 0e-tensors
    int i_block, n_blocks, s_blocks;
    size_t sizeU, sizeX, sizeY;
    double * d_tmpU, * d_tmpX, * d_tmpY, * d_no_0e_os;
    double * h_tmpU, * h_no_0e_os;


    cublasHandle_t myhandle;

    double alpha, beta;


    cudaEvent_t start_loc, stop_loc;
    cudaEvent_t start_tot, stop_tot;

    float time_loc=0.0f;
    float time_tot=0.0f;
    float tHD=0.0f;
    float tO=0.0f, tJ=0.0f, tM=0.0f, tS=0.0f;
    float tA=0.0f, tB=0.0f, tC1=0.0f, tC2=0.0f, tD2=0.0f;
    float tD=0.0f, tL=0.0f, tR=0.0f, tE=0.0f, tF=0.0f, tG=0.0f, tH=0.0f;
    float tX=0.0f, tY=0.0f, tU=0.0f;
    float tDgemm=0.0f, tDgemv=0.0f, tDdot=0.0f; 
    float t1=0.0f, t2=0.0f;
    




    printf(" Computing 0e, 1e, and 2e-Elements for Normal-Ordering With CuTC\n");

    checkCudaErrors(cudaEventCreate(&start_tot), "cudaEventCreate", __FILE__, __LINE__);
    checkCudaErrors(cudaEventCreate(&stop_tot), "cudaEventCreate", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(start_tot, 0), "cudaEventRecord", __FILE__, __LINE__);



    checkCudaErrors(cudaEventCreate(&start_loc), "cudaEventCreate", __FILE__, __LINE__);
    checkCudaErrors(cudaEventCreate(&stop_loc), "cudaEventCreate", __FILE__, __LINE__);


    checkCublasErrors(cublasCreate(&myhandle), "cublasCreate", __FILE__, __LINE__);
    //checkCublasErrors(cublasSetPointerMode(myhandle, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode", __FILE__, __LINE__);




    n_mo2 = n_mo * n_mo;

    size_wr1 = n_grid1 * sizeof(double);
    size_mos_in_r = n_grid1 * n_mo * sizeof(double);
    size_int2 = 3 * n_grid1 * n_mo2 * sizeof(double);

    size_0e = sizeof(double);
    size_1e = n_mo2 * sizeof(double);
    size_2e = n_mo2 * n_mo2 * sizeof(double);

    sizeO = n_grid1 * sizeof(double);
    sizeJ = 3 * n_grid1 * sizeof(double);
    sizeM = 3 * n_grid1 * sizeof(double);

    sizeA = 3 * n_grid1 * n_mo * sizeof(double);
    sizeB = 3 * n_grid1 * n_mo * sizeof(double);
    sizeC1 = 3 * n_grid1 * n_mo2 * sizeof(double);
    sizeC2 = n_grid1 * n_mo2 * sizeof(double);
    sizeD2 = n_grid1 * n_mo2 * sizeof(double);

    sizeS = n_grid1 * sizeof(double);
    sizeD = 4 * n_grid1 * sizeof(double);
    sizeL = 3 * n_grid1 * n_mo * sizeof(double);
    sizeR = 3 * n_grid1 * n_mo * sizeof(double);
    sizeE = 5 * n_grid1 * n_mo * sizeof(double);
    sizeF = 5 * n_grid1 * n_mo * sizeof(double);

    s_blocks = 32;
    n_blocks = (n_grid1 + s_blocks - 1) / s_blocks;
    sizeU = n_blocks * sizeof(double);


    checkCudaErrors(cudaMalloc((void**)&d_wr1, size_wr1), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_mos_l_in_r, size_mos_in_r), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_mos_r_in_r, size_mos_in_r), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_int2_grad1_u12, size_int2), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_no_2e, size_2e), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_no_1e, size_1e), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_no_0e, size_0e), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_tmpJ, sizeJ), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpO, sizeO), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpM, sizeM), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpS, sizeS), "cudaMalloc", __FILE__, __LINE__);

    // 2e-tensors
    checkCudaErrors(cudaMalloc((void**)&d_tmpA, sizeA), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpB, sizeB), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpC1, sizeC1), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpC2, sizeC2), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpD2, sizeD2), "cudaMalloc", __FILE__, __LINE__);

    // 1e-tensors
    checkCudaErrors(cudaMalloc((void**)&d_tmpD, sizeD), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpL, sizeL), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpR, sizeR), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpE, sizeE), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpF, sizeF), "cudaMalloc", __FILE__, __LINE__);

    // 0e-tensors
    h_tmpU = (double*) malloc(sizeU);
    checkCudaErrors(cudaMalloc((void**)&d_tmpU, sizeU), "cudaMalloc", __FILE__, __LINE__);


    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_wr1, h_wr1, size_wr1, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_mos_l_in_r, h_mos_l_in_r, size_mos_in_r, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_mos_r_in_r, h_mos_r_in_r, size_mos_in_r, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_int2_grad1_u12, h_int2_grad1_u12, size_int2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tHD = time_loc;
    time_tot += time_loc;




    

    if(ne_a == ne_b) {

        // Closed-Shell

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_tmpJ_cs(n_grid1, n_mo, ne_b, d_int2_grad1_u12, d_tmpJ);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tJ = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_tmpO_cs(n_grid1, ne_b, d_mos_l_in_r, d_mos_r_in_r, d_tmpO);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tO = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_tmpM_cs(n_grid1, n_mo, ne_b, d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12, d_tmpM);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tM = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_2e_tmpA_cs(n_grid1, n_mo, ne_b, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpA);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tA = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_2e_tmpB_cs(n_grid1, n_mo, ne_b, d_wr1, d_mos_r_in_r, d_int2_grad1_u12, d_tmpB);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tB = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_tmpC2_cs(n_grid1, n_mo, ne_b, d_int2_grad1_u12, d_tmpC2);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tC2 = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_1e_tmpL_cs(n_grid1, n_mo, ne_b, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpL);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tL = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_1e_tmpR_cs(n_grid1, n_mo, ne_b, d_mos_r_in_r, d_int2_grad1_u12, d_tmpR);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tR = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_tmpS_cs(n_grid1, n_mo, ne_b, d_int2_grad1_u12, d_tmpJ, d_tmpS);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tS = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_1e_tmpE_cs(n_grid1, n_mo, ne_b, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpJ, d_tmpL, d_tmpE);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tE = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_1e_tmpF_cs(n_grid1, n_mo, ne_b, d_mos_r_in_r, d_int2_grad1_u12, d_tmpS, d_tmpJ, d_tmpR, d_tmpF);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tF = time_loc;
        time_tot += time_loc;

    } else {

        // Open-Shell

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_tmpJ_os(n_grid1, n_mo, ne_b, ne_a, d_int2_grad1_u12, d_tmpJ);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tJ = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_tmpO_os(n_grid1, ne_b, ne_a, d_mos_l_in_r, d_mos_r_in_r, d_tmpO);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tO = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_tmpM_os(n_grid1, n_mo, ne_b, ne_a, d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12, d_tmpM);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tM = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_2e_tmpA_os(n_grid1, n_mo, ne_b, ne_a, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpA);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tA = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_2e_tmpB_os(n_grid1, n_mo, ne_b, ne_a, d_wr1, d_mos_r_in_r, d_int2_grad1_u12, d_tmpB);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tB = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_tmpC2_os(n_grid1, n_mo, ne_b, ne_a, d_int2_grad1_u12, d_tmpC2);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tC2 = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_tmpS_os(n_grid1, n_mo, ne_b, ne_a, d_int2_grad1_u12, d_tmpJ, d_tmpS);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tS = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_1e_tmpL_os(n_grid1, n_mo, ne_b, ne_a, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpL);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tL = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_1e_tmpR_os(n_grid1, n_mo, ne_b, ne_a, d_mos_r_in_r, d_int2_grad1_u12, d_tmpR);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tR = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_1e_tmpE_os(n_grid1, n_mo, ne_b, ne_a, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpJ, d_tmpL, d_tmpE);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tE = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_1e_tmpF_os(n_grid1, n_mo, ne_b, ne_a, d_mos_r_in_r, d_int2_grad1_u12, d_tmpS, d_tmpJ, d_tmpR, d_tmpF);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tF = time_loc;
        time_tot += time_loc;

    }


    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    no_2e_tmpC1(n_grid1, n_mo, d_wr1, d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12,
                d_tmpJ, d_tmpO, d_tmpA, d_tmpB, d_tmpC1);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tC1 = time_loc;
    time_tot += time_loc;

    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    no_2e_tmpD2(n_grid1, n_mo, d_wr1, d_mos_l_in_r, d_mos_r_in_r, d_tmpD2);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tD2 = time_loc;
    time_tot += time_loc;

    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    no_1e_tmpD(n_grid1, d_wr1, d_tmpO, d_tmpJ, d_tmpM, d_tmpD);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tD = time_loc;
    time_tot += time_loc;

    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

    alpha = 0.5;
    beta = 0.0;
    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCublasErrors(cublasDgemm(myhandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  n_mo2, n_mo2, 3*n_grid1,
                                  &alpha,
                                  &d_tmpC1[0], 3*n_grid1,
                                  &d_int2_grad1_u12[0], 3*n_grid1,
                                  &beta,
                                  &d_no_2e[0], n_mo2), "cublasDgemm", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tDgemm = time_loc;
    time_tot += time_loc;

    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);


    alpha = 0.5;
    beta = 1.0;
    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCublasErrors(cublasDgemm(myhandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  n_mo2, n_mo2, n_grid1,
                                  &alpha,
                                  &d_tmpC2[0], n_grid1,
                                  &d_tmpD2[0], n_grid1,
                                  &beta,
                                  &d_no_2e[0], n_mo2), "cublasDgemm", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tDgemm += time_loc;
    time_tot += time_loc;

    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

    // d_no_2e <-- d_no_2e + d_no_2e.T
    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    trans_inplace(d_no_2e, n_mo2);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    t1 = time_loc;
    time_tot += time_loc;

    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    trans_pqst_psqt_inplace(n_mo, d_no_2e);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    t2 = time_loc;
    time_tot += time_loc;



    alpha = 2.0;
    beta = 0.0;
    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCublasErrors(cublasDgemv(myhandle, 
                                  CUBLAS_OP_T,
                                  3*n_grid1, n_mo2,
                                  &alpha,
                                  &d_int2_grad1_u12[0], 3*n_grid1,
                                  &d_tmpD[0], 1,
                                  &beta,
                                  &d_no_1e[0], 1), "cublasDgemv", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tDgemv = time_loc;
    time_tot += time_loc;

    alpha = 2.0;
    beta = 1.0;
    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCublasErrors(cublasDgemv(myhandle, 
                                  CUBLAS_OP_T,
                                  n_grid1, n_mo2,
                                  &alpha,
                                  &d_tmpC2[0], n_grid1,
                                  &d_tmpD[3*n_grid1], 1,
                                  &beta,
                                  &d_no_1e[0], 1), "cublasDgemv", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tDgemv += time_loc;
    time_tot += time_loc;


    alpha = 1.0;
    beta = 1.0;
    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCublasErrors(cublasDgemm(myhandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  n_mo, n_mo, 5*n_grid1,
                                  &alpha,
                                  &d_tmpE[0], 5*n_grid1,
                                  &d_tmpF[0], 5*n_grid1,
                                  &beta,
                                  &d_no_1e[0], n_mo), "cublasDgemm", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tDgemm += time_loc;
    time_tot += time_loc;


    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCublasErrors(cublasDdot(myhandle, 3*n_grid1*ne_b,
                                 &d_tmpL[0], 1,
                                 &d_tmpR[0], 1,
                                 &d_no_0e[0]), "cublasDdot", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tDdot = time_loc;
    time_tot += time_loc;

    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_no_0e, d_no_0e, size_0e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tHD += time_loc;
    time_tot += time_loc;

    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    no_0e_tmpU(n_grid1, n_blocks, s_blocks, d_wr1, d_tmpO, d_tmpS, d_tmpJ, d_tmpM, d_tmpU);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tU = time_loc;
    time_tot += time_loc;

    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_tmpU, d_tmpU, sizeU, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tU += time_loc;
    time_tot += time_loc;

    clock_t time_req;
    time_req = clock();
    for(i_block=0; i_block<n_blocks; i_block++) {
        h_no_0e[0] += h_tmpU[i_block];
    }
    h_no_0e[0] = -2.0 * h_no_0e[0];
    time_req = clock() - time_req;
    printf("CPU time : %f sec\n", (float)time_req / CLOCKS_PER_SEC);
    tU += 1000.f * (float)time_req / CLOCKS_PER_SEC;


    if(ne_a != ne_b) {

        sizeG = 3 * n_grid1 * n_mo * sizeof(double);
        sizeH = 3 * n_grid1 * n_mo * sizeof(double);

        checkCudaErrors(cudaMalloc((void**)&d_tmpG, sizeG), "cudaMalloc", __FILE__, __LINE__);
        checkCudaErrors(cudaMalloc((void**)&d_tmpH, sizeH), "cudaMalloc", __FILE__, __LINE__);

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_1e_tmpG_os(n_grid1, n_mo, ne_b, ne_a, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpG);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tG = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_1e_tmpH_os(n_grid1, n_mo, ne_b, ne_a, d_mos_r_in_r, d_int2_grad1_u12, d_tmpH);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tH = time_loc;
        time_tot += time_loc;

        alpha = 1.0;
        beta = 1.0;
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCublasErrors(cublasDgemm(myhandle,
                                      CUBLAS_OP_T, CUBLAS_OP_N,
                                      n_mo, n_mo, 3*n_grid1,
                                      &alpha,
                                      &d_tmpG[0], 3*n_grid1,
                                      &d_tmpH[0], 3*n_grid1,
                                      &beta,
                                      &d_no_1e[0], n_mo), "cublasDgemm", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tDgemm += time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaFree(d_tmpG), "cudaFree", __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_tmpH), "cudaFree", __FILE__, __LINE__);



        sizeX = 3 * n_grid1 * ne_a * sizeof(double);
        sizeY = 3 * n_grid1 * ne_a * sizeof(double);

        checkCudaErrors(cudaMalloc((void**)&d_tmpX, sizeX), "cudaMalloc", __FILE__, __LINE__);
        checkCudaErrors(cudaMalloc((void**)&d_tmpY, sizeY), "cudaMalloc", __FILE__, __LINE__);
        h_no_0e_os = (double*) malloc(size_0e);
        checkCudaErrors(cudaMalloc((void**)&d_no_0e_os, size_0e), "cudaMalloc", __FILE__, __LINE__);

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_0e_tmpX_os(n_grid1, n_mo, ne_b, ne_a, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpX);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tX = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        no_0e_tmpY_os(n_grid1, n_mo, ne_b, ne_a, d_mos_r_in_r, d_int2_grad1_u12, d_tmpY);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tY = time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCublasErrors(cublasDdot(myhandle, 3*n_grid1*ne_a,
                                     &d_tmpX[0], 1,
                                     &d_tmpY[0], 1,
                                     &d_no_0e_os[0]), "cublasDdot", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tDdot += time_loc;
        time_tot += time_loc;

        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

        checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaMemcpy(h_no_0e_os, d_no_0e_os, size_0e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
        checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
        checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
        checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
        tHD += time_loc;
        time_tot += time_loc;

        h_no_0e[0] -= 2.0 * h_no_0e_os[0];

        checkCudaErrors(cudaFree(d_no_0e_os), "cudaFree", __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_tmpX), "cudaFree", __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_tmpY), "cudaFree", __FILE__, __LINE__);
        free(h_no_0e_os);

        printf("Ellapsed time for tmpG  = %.3f sec\n", tG /1000.0f);
        printf("Ellapsed time for tmpH  = %.3f sec\n", tH /1000.0f);
        printf("Ellapsed time for tmpX  = %.3f sec\n", tX /1000.0f);
        printf("Ellapsed time for tmpY  = %.3f sec\n", tY /1000.0f);

    }




    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);


    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_no_2e, d_no_2e, size_2e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_no_1e, d_no_1e, size_1e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tHD += time_loc;
    time_tot += time_loc;


    checkCudaErrors(cudaFree(d_wr1), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_mos_l_in_r), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_mos_r_in_r), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_int2_grad1_u12), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_no_2e), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_no_1e), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_no_0e), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_tmpO), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpJ), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_tmpA), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpB), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpC1), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpC2), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpD2), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_tmpM), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpS), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpD), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpL), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpR), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpE), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpF), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_tmpU), "cudaFree", __FILE__, __LINE__);

    free(h_tmpU);

    checkCublasErrors(cublasDestroy(myhandle), "cublasDestroy", __FILE__, __LINE__);

    checkCudaErrors(cudaEventDestroy(start_loc), "cudaEventDestroy", __FILE__, __LINE__);
    checkCudaErrors(cudaEventDestroy(stop_loc), "cudaEventDestroy", __FILE__, __LINE__);

    printf("Ellapsed time for Device <-> Host transf = %.3f sec\n", tHD/1000.0f);
    printf("Ellapsed time for tmpO  = %.3f sec\n", tO /1000.0f);
    printf("Ellapsed time for tmpJ  = %.3f sec\n", tJ /1000.0f);
    printf("Ellapsed time for tmpM  = %.3f sec\n", tM /1000.0f);
    printf("Ellapsed time for tmpS  = %.3f sec\n", tS /1000.0f);
    printf("Ellapsed time for tmpA  = %.3f sec\n", tA /1000.0f);
    printf("Ellapsed time for tmpB  = %.3f sec\n", tB /1000.0f);
    printf("Ellapsed time for tmpC1 = %.3f sec\n", tC1/1000.0f);
    printf("Ellapsed time for tmpC2 = %.3f sec\n", tC2/1000.0f);
    printf("Ellapsed time for tmpD2 = %.3f sec\n", tD2/1000.0f);
    printf("Ellapsed time for tmpD  = %.3f sec\n", tD /1000.0f);
    printf("Ellapsed time for tmpL  = %.3f sec\n", tL /1000.0f);
    printf("Ellapsed time for tmpR  = %.3f sec\n", tR /1000.0f);
    printf("Ellapsed time for tmpE  = %.3f sec\n", tE /1000.0f);
    printf("Ellapsed time for tmpF  = %.3f sec\n", tF /1000.0f);
    printf("Ellapsed time for tmpU  = %.3f sec\n", tU /1000.0f);
    printf("Ellapsed time for DGEMM = %.3f sec\n", tDgemm/1000.0f);
    printf("Ellapsed time for DGEMV = %.3f sec\n", tDgemv/1000.0f);
    printf("Ellapsed time for Ddot  = %.3f sec\n", tDdot/1000.0f);
    printf("Ellapsed time for addT  = %.3f sec\n", t1/1000.0f);
    printf("Ellapsed time for Trans = %.3f sec\n", t2/1000.0f);
    printf("Ellapsed (effective) time on GPU for cutc_no = %.3f sec\n", time_tot/1000.0f);

    checkCudaErrors(cudaEventRecord(stop_tot, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_tot), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_tot, stop_tot), "cudaEventElapsedTime", __FILE__, __LINE__);
    printf("Ellapsed (total) time on GPU for cutc_no = %.3f sec\n", time_loc/1000.0f);


    return 0;

}


