#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>






extern void checkCudaErrors(cudaError_t err, const char * msg, const char * file, int line);
extern void checkCublasErrors(cublasStatus_t status, const char * msg, const char * file, int line);


extern void no_tmpO_cs(int n_grid1, int ne_b, double * mos_l_in_r, double * mos_r_in_r, double * tmpO);
extern void no_tmpJ_cs(int n_grid1, int n_mo, int ne_b, double * int2_grad1_u12, double * tmpJ);

extern void no_tmpO_os(int n_grid1, int ne_b, int ne_a, double * mos_l_in_r, double * mos_r_in_r, double * tmpO);
extern void no_tmpJ_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * int2_grad1_u12, double * tmpJ);

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

extern void no_2e_tmpC1(int n_grid1, int n_mo,
                        double * wr1, double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                        double * tmpJ, double * tmpO, double * tmpA, double * tmpB,
                        double * tmpC1);

extern void no_tmpC2_cs(int n_grid1, int n_mo, int ne_b,
                        double * int2_grad1_u12,
                        double * tmpC2);

extern void no_tmpC2_os(int n_grid1, int n_mo, int ne_b, int ne_a,
                        double * int2_grad1_u12,
                        double * tmpC2);

extern void no_2e_tmpD2(int n_grid1, int n_mo,
                        double * wr1, double * mos_l_in_r, double * mos_r_in_r,
                        double * tmpD2);


extern void trans_inplace(double * data, int size);

extern void trans_pqst_psqt_inplace(int size, double * data);






int cutc_no_2e(int n_grid1, int n_mo, int ne_a, int ne_b,
               double * h_wr1, double * h_mos_l_in_r, double * h_mos_r_in_r, double * h_int2_grad1_u12, 
               double * h_no_2e) {


    int n_mo2;

    size_t size_wr1;
    size_t size_mos_in_r;
    size_t size_int2;

    size_t sizeO, sizeJ;
    size_t sizeA, sizeB;
    size_t sizeC1, sizeC2;
    size_t sizeD2;

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
    double * d_tmpC1;
    double * d_tmpC2;
    double * d_tmpD2;


    double alpha, beta;

    printf(" Computing 2e-Elements for Normal-Ordering With CuTC\n");


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

        no_tmpO_os(n_grid1, ne_b, ne_a, d_mos_l_in_r, d_mos_r_in_r, d_tmpO);
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

    checkCublasErrors(cublasDestroy(myhandle), "cublasDestroy", __FILE__, __LINE__);


    return 0;

}
