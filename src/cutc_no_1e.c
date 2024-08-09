#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>






extern void checkCudaErrors(cudaError_t err, const char * msg, const char * file, int line);
extern void checkCublasErrors(cublasStatus_t status, const char * msg, const char * file, int line);


extern void no_1e_tmpO_cs(int n_grid1, int ne_b,
                          double * mos_l_in_r, double * mos_r_in_r,
                          double * tmpO);

extern void no_1e_tmpJ_cs(int n_grid1, int n_mo, int ne_b,
                          double * int2_grad1_u12,
                          double * tmpJ);

extern void no_1e_tmpM_cs(int n_grid1, int n_mo, int ne_b,
                          double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                          double * tmpM);

extern void no_1e_tmpS_cs(int n_grid1, int n_mo, int ne_b,
                          double * int2_grad1_u12, double * tmpJ,
                          double * tmpS);

extern void no_1e_tmpD_cs(int n_grid1,
                          double * wr1, double * tmpO, double * tmpJ, double * tmpM,
                          double * tmpD);

extern void no_tmpC2_cs(int n_grid1, int n_mo, int ne_b,
                        double * int2_grad1_u12,
                        double * tmpC2);






int cutc_no_1e(int n_grid1, int n_mo, int ne_a, int ne_b,
               double * h_wr1, 
               double * h_mos_l_in_r, double * h_mos_r_in_r, 
               double * h_int2_grad1_u12, 
               double * h_no_1e) {


    int n_mo2;

    size_t size_wr1;
    size_t size_mos_in_r;
    size_t size_int2;

    size_t size_1e;

    size_t sizeO;
    size_t sizeJ;
    size_t sizeM;
    size_t sizeS;
    size_t sizeD;
    size_t sizeC2;

    double * d_wr1;
    double * d_mos_l_in_r;
    double * d_mos_r_in_r;
    double * d_int2_grad1_u12;

    double * d_no_1e;


    double * d_tmpO;
    double * d_tmpJ;
    double * d_tmpM;
    double * d_tmpS;
    double * d_tmpD;
    double * d_tmpC2;


    double alpha, beta;

    printf(" Computing 1e-Elements for Normal-Ordering With CuTC\n");


    cublasHandle_t myhandle;

    checkCublasErrors(cublasCreate(&myhandle), "cublasCreate", __FILE__, __LINE__);


    n_mo2 = n_mo * n_mo;

    size_wr1 = n_grid1 * sizeof(double);
    size_mos_in_r = n_grid1 * n_mo * sizeof(double);
    size_int2 = 3 * n_grid1 * n_mo2 * sizeof(double);

    size_1e = n_mo2 * sizeof(double);

    sizeO = n_grid1 * sizeof(double);
    sizeJ = 3 * n_grid1 * sizeof(double);
    sizeM = 3 * n_grid1 * sizeof(double);
    sizeS = n_grid1 * sizeof(double);
    sizeD = 4 * n_grid1 * sizeof(double);
    sizeC2 = n_grid1 * n_mo2 * sizeof(double);


    checkCudaErrors(cudaMalloc((void**)&d_wr1, size_wr1), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_mos_l_in_r, size_mos_in_r), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_mos_r_in_r, size_mos_in_r), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_int2_grad1_u12, size_int2), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_no_1e, size_1e), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_tmpO, sizeO), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpJ, sizeJ), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpM, sizeM), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpS, sizeS), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpD, sizeD), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpC2, sizeC2), "cudaMalloc", __FILE__, __LINE__);


    checkCudaErrors(cudaMemcpy(d_wr1, h_wr1, size_wr1, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_mos_l_in_r, h_mos_l_in_r, size_mos_in_r, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_mos_r_in_r, h_mos_r_in_r, size_mos_in_r, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_int2_grad1_u12, h_int2_grad1_u12, size_int2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);



    no_1e_tmpO_cs(n_grid1, ne_b, d_mos_l_in_r, d_mos_r_in_r, d_tmpO);
    no_1e_tmpJ_cs(n_grid1, n_mo, ne_b, d_int2_grad1_u12, d_tmpJ);
    no_1e_tmpM_cs(n_grid1, n_mo, ne_b, d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12, d_tmpM);
    no_1e_tmpS_cs(n_grid1, n_mo, ne_b, d_int2_grad1_u12, d_tmpJ, d_tmpS);
    no_1e_tmpD_cs(n_grid1, d_wr1, d_tmpO, d_tmpJ, d_tmpM, d_tmpD);
    no_tmpC2_cs(n_grid1, n_mo, ne_b, d_int2_grad1_u12, d_tmpC2);


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

















    checkCudaErrors(cudaMemcpy(h_no_1e, d_no_1e, size_1e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);


    checkCudaErrors(cudaFree(d_wr1), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_mos_l_in_r), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_mos_r_in_r), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_int2_grad1_u12), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_no_1e), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_tmpO), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpJ), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpM), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpS), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpD), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpC2), "cudaFree", __FILE__, __LINE__);

    checkCublasErrors(cublasDestroy(myhandle), "cublasDestroy", __FILE__, __LINE__);


    return 0;

}
