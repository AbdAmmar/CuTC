#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>


#include "cutc_no.h"



int cutc_no(int n_grid1, int n_mo, int ne_a, int ne_b,
            double * h_wr1, double * h_mos_l_in_r, double * h_mos_r_in_r, double * h_int2_grad1_u12, 
            double * h_no_2e, double * h_no_1e, double * h_no_0e) {


    int n_mo2;

    size_t size_wr1;
    size_t size_mos_in_r;
    size_t size_int2;

    size_t sizeO;
    size_t sizeJ;

    size_t sizeA, sizeB;
    size_t sizeC1, sizeC2;
    size_t sizeD2;
    size_t sizeE;

    size_t size_2e;
    size_t size_1e;
    size_t size_0e;

    double * d_wr1;
    double * d_mos_l_in_r;
    double * d_mos_r_in_r;
    double * d_int2_grad1_u12;

    double * d_no_2e;

    double * d_tmpO;
    double * d_tmpJ;

    double * d_tmpA;
    double * d_tmpB;
    double * d_tmpC1;
    double * d_tmpC2;
    double * d_tmpD2;


    double alpha, beta;

    printf(" Computing 0e, 1e, and 2e-Elements for Normal-Ordering With CuTC\n");


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
    sizeC1 = 3 * n_grid1 * n_mo2 * sizeof(double);
    sizeC2 = n_grid1 * n_mo2 * sizeof(double);
    sizeD2 = n_grid1 * n_mo2 * sizeof(double);

    size_2e = n_mo2 * n_mo2 * sizeof(double);
    size_1e = n_mo2 * sizeof(double);
    size_0e = sizeof(double);


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
    checkCudaErrors(cudaMalloc((void**)&d_tmpD2, sizeD2), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_no_2e, size_2e), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_no_1e, size_1e), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_no_0e, size_0e), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMemcpy(d_wr1, h_wr1, size_wr1, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_mos_l_in_r, h_mos_l_in_r, size_mos_in_r, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_mos_r_in_r, h_mos_r_in_r, size_mos_in_r, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_int2_grad1_u12, h_int2_grad1_u12, size_int2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);

    if(ne_a == ne_b) {

        // Closed-Shell

        no_tmpO_cs(n_grid1, ne_b, d_mos_l_in_r, d_mos_r_in_r, d_tmpO);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_tmpJ_cs(n_grid1, n_mo, ne_b, d_int2_grad1_u12, d_tmpJ);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_2e_tmpA_cs(n_grid1, n_mo, ne_b, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpA);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_2e_tmpB_cs(n_grid1, n_mo, ne_b, d_wr1, d_mos_r_in_r, d_int2_grad1_u12, d_tmpB);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_tmpC2_cs(n_grid1, n_mo, ne_b, d_int2_grad1_u12, d_tmpC2);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

    } else {

        // Open-Shell

        no_tmpO_os(n_grid1, ne_b, ne_a, d_wr1, d_mos_l_in_r, d_mos_r_in_r, d_tmpO);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_tmpJ_os(n_grid1, n_mo, ne_b, ne_a, d_int2_grad1_u12, d_tmpJ);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_2e_tmpA_os(n_grid1, n_mo, ne_b, ne_a, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpA);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_2e_tmpB_os(n_grid1, n_mo, ne_b, ne_a, d_wr1, d_mos_r_in_r, d_int2_grad1_u12, d_tmpB);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_tmpC2_os(n_grid1, n_mo, ne_b, ne_a, d_int2_grad1_u12, d_tmpC2);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

    }

    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

    no_2e_tmpC1(n_grid1, n_mo, d_wr1, d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12,
                d_tmpJ, d_tmpO, d_tmpA, d_tmpB, d_tmpC1);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

    no_2e_tmpD2(n_grid1, n_mo, d_wr1, d_mos_l_in_r, d_mos_r_in_r, d_tmpD2);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

    alpha = 0.5;
    beta = 0.0;
    checkCublasErrors(cublasDgemm(myhandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  n_mo2, n_mo2, 3*n_grid1,
                                  &alpha,
                                  &d_tmpC1[0], 3*n_grid1,
                                  &d_int2_grad1_u12[0], 3*n_grid1,
                                  &beta,
                                  &d_no_2e[0], n_mo2), "cublasDgemm", __FILE__, __LINE__);

    beta = 1.0;
    checkCublasErrors(cublasDgemm(myhandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  n_mo2, n_mo2, n_grid1,
                                  &alpha,
                                  &d_tmpC2[0], n_grid1,
                                  &d_tmpD2[0], n_grid1,
                                  &beta,
                                  &d_no_2e[0], n_mo2), "cublasDgemm", __FILE__, __LINE__);

    // d_no_2e <-- d_no_2e + d_no_2e.T
    trans_inplace(d_no_2e, n_mo2);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

    trans_pqst_psqt_inplace(n_mo, d_no_2e);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

    checkCudaErrors(cudaMemcpy(h_no_2e, d_no_2e, size_2e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);


    checkCudaErrors(cudaFree(d_wr1), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_mos_l_in_r), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_mos_r_in_r), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_int2_grad1_u12), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_tmpO), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpJ), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_tmpA), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpB), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpC1), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpC2), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpD2), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_no_2e), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_no_1e), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_no_0e), "cudaFree", __FILE__, __LINE__);

    checkCublasErrors(cublasDestroy(myhandle), "cublasDestroy", __FILE__, __LINE__);


    return 0;

}


    size_t sizeM;
    size_t sizeS;
    size_t sizeD;
    size_t sizeC2;
    size_t sizeL;
    size_t sizeR;
    size_t sizeE;
    size_t sizeF;
    size_t sizeG;
    size_t sizeH;

    double * d_tmpM;
    double * d_tmpS;
    double * d_tmpD;
    double * d_tmpC2;
    double * d_tmpL;
    double * d_tmpR;
    double * d_tmpE;
    double * d_tmpF;
    double * d_tmpG;
    double * d_tmpH;



    sizeM = 3 * n_grid1 * sizeof(double);
    sizeS = n_grid1 * sizeof(double);
    sizeD = 4 * n_grid1 * sizeof(double);
    sizeC2 = n_grid1 * n_mo2 * sizeof(double);
    sizeL = 3 * n_grid1 * n_mo * sizeof(double);
    sizeR = 3 * n_grid1 * n_mo * sizeof(double);
    sizeE = 5 * n_grid1 * n_mo * sizeof(double);
    sizeF = 5 * n_grid1 * n_mo * sizeof(double);


    checkCudaErrors(cudaMalloc((void**)&d_tmpM, sizeM), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpS, sizeS), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpD, sizeD), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpC2, sizeC2), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpL, sizeL), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpR, sizeR), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpE, sizeE), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpF, sizeF), "cudaMalloc", __FILE__, __LINE__);

    if(ne_a != ne_b) {

        // Open-Shell

        sizeG = 3 * n_grid1 * n_mo * sizeof(double);
        sizeH = 3 * n_grid1 * n_mo * sizeof(double);

        checkCudaErrors(cudaMalloc((void**)&d_tmpG, sizeG), "cudaMalloc", __FILE__, __LINE__);
        checkCudaErrors(cudaMalloc((void**)&d_tmpH, sizeH), "cudaMalloc", __FILE__, __LINE__);

    }


    checkCudaErrors(cudaMemcpy(d_wr1, h_wr1, size_wr1, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_mos_l_in_r, h_mos_l_in_r, size_mos_in_r, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_mos_r_in_r, h_mos_r_in_r, size_mos_in_r, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_int2_grad1_u12, h_int2_grad1_u12, size_int2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);


    if(ne_a == ne_b) {

        // Closed-Shell

        no_tmpM_cs(n_grid1, n_mo, ne_b, d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12, d_tmpM);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
    
        no_tmpS_cs(n_grid1, n_mo, ne_b, d_int2_grad1_u12, d_tmpJ, d_tmpS);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    
        no_tmpC2_cs(n_grid1, n_mo, ne_b, d_int2_grad1_u12, d_tmpC2);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

    } else {

        // Open-Shell

        no_tmpM_os(n_grid1, n_mo, ne_b, ne_a, d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12, d_tmpM);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
    
        no_tmpS_os(n_grid1, n_mo, ne_b, ne_a, d_int2_grad1_u12, d_tmpJ, d_tmpS);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    
        no_tmpC2_os(n_grid1, n_mo, ne_b, ne_a, d_int2_grad1_u12, d_tmpC2);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

    }

    no_1e_tmpD(n_grid1, d_wr1, d_tmpO, d_tmpJ, d_tmpM, d_tmpD);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);


    alpha = 2.0;
    beta = 0.0;
    checkCublasErrors(cublasDgemv(myhandle, 
                                  CUBLAS_OP_T,
                                  3*n_grid1, n_mo2,
                                  &alpha,
                                  &d_int2_grad1_u12[0], 3*n_grid1,
                                  &d_tmpD[0], 1,
                                  &beta,
                                  &d_no_1e[0], 1), "cublasDgemv", __FILE__, __LINE__);

    beta = 1.0;
    checkCublasErrors(cublasDgemv(myhandle, 
                                  CUBLAS_OP_T,
                                  n_grid1, n_mo2,
                                  &alpha,
                                  &d_tmpC2[0], n_grid1,
                                  &d_tmpD[3*n_grid1], 1,
                                  &beta,
                                  &d_no_1e[0], 1), "cublasDgemv", __FILE__, __LINE__);


    if(ne_a == ne_b) {

        // Closed-Shell

        no_1e_tmpL_cs(n_grid1, n_mo, ne_b, d_mos_l_in_r, d_int2_grad1_u12, d_tmpL);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_1e_tmpR_cs(n_grid1, n_mo, ne_b, d_mos_r_in_r, d_int2_grad1_u12, d_tmpR);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

        no_1e_tmpE_cs(n_grid1, n_mo, ne_b, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpJ, d_tmpL, d_tmpE);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_1e_tmpF_cs(n_grid1, n_mo, ne_b, d_wr1, d_mos_r_in_r, d_int2_grad1_u12, d_tmpS, d_tmpJ, d_tmpR, d_tmpF);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

    } else {

        // Open-Shell

        no_1e_tmpL_os(n_grid1, n_mo, ne_b, ne_a, d_mos_l_in_r, d_int2_grad1_u12, d_tmpL);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_1e_tmpR_os(n_grid1, n_mo, ne_b, ne_a, d_mos_r_in_r, d_int2_grad1_u12, d_tmpR);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

        no_1e_tmpE_os(n_grid1, n_mo, ne_b, ne_a, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpJ, d_tmpL, d_tmpE);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_1e_tmpF_os(n_grid1, n_mo, ne_b, ne_a, d_wr1, d_mos_r_in_r, d_int2_grad1_u12, d_tmpS, d_tmpJ, d_tmpR, d_tmpF);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

    }

    alpha = 1.0;
    beta = 1.0;
    checkCublasErrors(cublasDgemm(myhandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  n_mo, n_mo, 5*n_grid1,
                                  &alpha,
                                  &d_tmpE[0], 5*n_grid1,
                                  &d_tmpF[0], 5*n_grid1,
                                  &beta,
                                  &d_no_1e[0], n_mo), "cublasDgemm", __FILE__, __LINE__);


    if(ne_a != ne_b) {

        // Open-Shell

        no_1e_tmpG_os(n_grid1, n_mo, ne_b, ne_a, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpG);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_1e_tmpH_os(n_grid1, n_mo, ne_b, ne_a, d_mos_r_in_r, d_int2_grad1_u12, d_tmpH);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        checkCublasErrors(cublasDgemm(myhandle,
                                      CUBLAS_OP_T, CUBLAS_OP_N,
                                      n_mo, n_mo, 3*n_grid1,
                                      &alpha,
                                      &d_tmpG[0], 3*n_grid1,
                                      &d_tmpH[0], 3*n_grid1,
                                      &beta,
                                      &d_no_1e[0], n_mo), "cublasDgemm", __FILE__, __LINE__);

    }



    checkCudaErrors(cudaMemcpy(h_no_1e, d_no_1e, size_1e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_wr1), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_mos_l_in_r), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_mos_r_in_r), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_int2_grad1_u12), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_tmpM), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpS), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpD), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpC2), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpL), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpR), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpE), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpF), "cudaFree", __FILE__, __LINE__);

    if(ne_a != ne_b) {

        // Open-Shell

        checkCudaErrors(cudaFree(d_tmpG), "cudaFree", __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_tmpH), "cudaFree", __FILE__, __LINE__);

    }

    checkCublasErrors(cublasDestroy(myhandle), "cublasDestroy", __FILE__, __LINE__);









    size_t sizeM, sizeS;
    size_t sizeL;
    size_t sizeR;

    double * d_tmpM;
    double * d_tmpS;
    double * d_tmpL;
    double * d_tmpR;
    double * d_tmpE;

    sizeM = 3 * n_grid1 * sizeof(double);
    sizeS = n_grid1 * sizeof(double);
    sizeL = 3 * n_grid1 * ne_a * sizeof(double);
    sizeR = 3 * n_grid1 * ne_a * sizeof(double);

    checkCudaErrors(cudaMalloc((void**)&d_tmpL, sizeL), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpR, sizeR), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpM, sizeM), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpS, sizeS), "cudaMalloc", __FILE__, __LINE__);


    if(ne_a == ne_b) {

        // Closed-Shell

        no_0e_tmpL_cs(n_grid1, n_mo, ne_b, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpL);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_0e_tmpR_cs(n_grid1, n_mo, ne_b, d_mos_r_in_r, d_int2_grad1_u12, d_tmpR);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    
    } else {

        // Open-Shell

        no_0e_tmpL_os(n_grid1, n_mo, ne_b, ne_a, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpL);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_0e_tmpR_os(n_grid1, n_mo, ne_b, ne_a, d_mos_r_in_r, d_int2_grad1_u12, d_tmpR);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

    }

    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

    checkCublasErrors(cublasDdot(myhandle, 3*n_grid1*ne_a,
                                 &d_tmpL[0], 1,
                                 &d_tmpR[0], 1,
                                 &d_no_0e[0]), "cublasDdot", __FILE__, __LINE__);

    checkCudaErrors(cudaMemcpy(h_no_0e, d_no_0e, size_0e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);


    if(ne_a == ne_b) {

        // Closed-Shell

        no_tmpM_cs(n_grid1, n_mo, ne_b, d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12, d_tmpM);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

        no_tmpS_cs(n_grid1, n_mo, ne_b, d_int2_grad1_u12, d_tmpJ, d_tmpS);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    
    } else {

        // Open-Shell

        no_tmpM_os(n_grid1, n_mo, ne_b, ne_a, d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12, d_tmpM);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

        no_tmpS_os(n_grid1, n_mo, ne_b, ne_a, d_int2_grad1_u12, d_tmpJ, d_tmpS);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

    }


    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);


    int i_block;
    int s_blocks = 32;
    int n_blocks = (n_grid1 + s_blocks - 1) / s_blocks;

    size_t sizeE = n_blocks * sizeof(double);

    checkCudaErrors(cudaMalloc((void**)&d_tmpE, sizeE), "cudaMalloc", __FILE__, __LINE__);

    no_0e_tmpE(n_grid1, n_blocks, s_blocks, d_wr1, d_tmpO, d_tmpS, d_tmpJ, d_tmpM, d_tmpE);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

    double * h_tmpE;
    h_tmpE = (double*) malloc(sizeE);
    if(h_tmpE == NULL) {
        fprintf(stderr, "Memory allocation failed for h_tmpE\n");
        exit(0);
    }
    checkCudaErrors(cudaMemcpy(h_tmpE, d_tmpE, sizeE, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_tmpE), "cudaFree", __FILE__, __LINE__);

    for(i_block=0; i_block<n_blocks; i_block++) {
        h_no_0e[0] += h_tmpE[i_block];
    }
    h_no_0e[0] = -2.0 * h_no_0e[0];


    if(ne_a != ne_b) {

        double * h_no_0e_os;
        h_no_0e_os = (double*) malloc(size_0e);
        if(h_no_0e_os == NULL) {
            fprintf(stderr, "Memory allocation failed for h_no_0e_os\n");
            exit(0);
        }

        double * d_no_0e_os;
        double * d_tmpG;
        double * d_tmpH;

        size_t sizeG = 3 * n_grid1 * ne_b * sizeof(double);
        size_t sizeH = 3 * n_grid1 * ne_b * sizeof(double);

        checkCudaErrors(cudaMalloc((void**)&d_no_0e_os, size_0e), "cudaMalloc", __FILE__, __LINE__);

        checkCudaErrors(cudaMalloc((void**)&d_tmpG, sizeG), "cudaMalloc", __FILE__, __LINE__);
        checkCudaErrors(cudaMalloc((void**)&d_tmpH, sizeH), "cudaMalloc", __FILE__, __LINE__);

        no_0e_tmpG_os(n_grid1, n_mo, ne_b, ne_a, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpG);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_0e_tmpH_os(n_grid1, n_mo, ne_b, ne_a, d_mos_r_in_r, d_int2_grad1_u12, d_tmpH);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

        checkCublasErrors(cublasDdot(myhandle, 3*n_grid1*ne_b,
                                     &d_tmpG[0], 1,
                                     &d_tmpH[0], 1,
                                     &d_no_0e_os[0]), "cublasDdot", __FILE__, __LINE__);

        checkCudaErrors(cudaFree(d_tmpG), "cudaFree", __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_tmpH), "cudaFree", __FILE__, __LINE__);

        checkCudaErrors(cudaMemcpy(h_no_0e_os, d_no_0e_os, size_0e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);

        checkCudaErrors(cudaFree(d_no_0e_os), "cudaFree", __FILE__, __LINE__);

        h_no_0e[0] -= 2.0 * h_no_0e_os[0];

    }



    checkCudaErrors(cudaFree(d_tmpL), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpR), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpM), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpS), "cudaFree", __FILE__, __LINE__);

