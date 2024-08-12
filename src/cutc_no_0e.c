#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>






extern void checkCudaErrors(cudaError_t err, const char * msg, const char * file, int line);
extern void checkCublasErrors(cublasStatus_t status, const char * msg, const char * file, int line);


extern void no_1e_tmpL_cs(int n_grid1, int n_mo, int ne_b, double * wr1, double * mos_l_in_r, double * int2_grad1_u12, double * tmpL);
extern void no_1e_tmpL_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * wr1, double * mos_l_in_r, double * int2_grad1_u12, double * tmpL);

extern void no_1e_tmpR_cs(int n_grid1, int n_mo, int ne_b, double * mos_r_in_r, double * int2_grad1_u12, double * tmpR);
extern void no_1e_tmpR_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * mos_r_in_r, double * int2_grad1_u12, double * tmpR);

extern void no_0e_tmpX_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * wr1, double * mos_l_in_r, double * int2_grad1_u12, double * tmpX);
extern void no_0e_tmpY_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * mos_r_in_r, double * int2_grad1_u12, double * tmpY);

extern void no_tmpO_cs(int n_grid1, int ne_b,
                       double * mos_l_in_r, double * mos_r_in_r,
                       double * tmpO);

extern void no_tmpO_os(int n_grid1, int ne_b, int ne_a,
                       double * mos_l_in_r, double * mos_r_in_r,
                       double * tmpO);

extern void no_tmpJ_cs(int n_grid1, int n_mo, int ne_b,
                       double * int2_grad1_u12,
                       double * tmpJ);

extern void no_tmpJ_os(int n_grid1, int n_mo, int ne_b, int ne_a,
                       double * int2_grad1_u12,
                       double * tmpJ);

extern void no_tmpM_cs(int n_grid1, int n_mo, int ne_b,
                       double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                       double * tmpM);

extern void no_tmpM_os(int n_grid1, int n_mo, int ne_b, int ne_a,
                       double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                       double * tmpM);

extern void no_tmpS_cs(int n_grid1, int n_mo, int ne_b,
                       double * int2_grad1_u12, double * tmpJ,
                       double * tmpS);

extern void no_tmpS_os(int n_grid1, int n_mo, int ne_b, int ne_a,
                       double * int2_grad1_u12, double * tmpJ,
                       double * tmpS);


extern void no_0e_tmpU(int n_grid1, int n_blocks, int s_blocks,
                       double * wr1, double * tmpO, double * tmpS, double * tmpJ, double * tmpM,
                       double * tmpE);



int cutc_no_0e(int n_grid1, int n_mo, int ne_a, int ne_b,
               double * h_wr1, 
               double * h_mos_l_in_r, double * h_mos_r_in_r, 
               double * h_int2_grad1_u12, 
               double * h_no_0e) {


    int n_mo2;

    size_t size_wr1;
    size_t size_mos_in_r;
    size_t size_int2;

    size_t size_0e;

    size_t sizeO, sizeJ;
    size_t sizeM, sizeS;
    size_t sizeL;
    size_t sizeR;

    double * d_wr1;
    double * d_mos_l_in_r;
    double * d_mos_r_in_r;
    double * d_int2_grad1_u12;

    double * d_no_0e;


    double * d_tmpO;
    double * d_tmpJ;
    double * d_tmpM;
    double * d_tmpS;
    double * d_tmpL;
    double * d_tmpR;
    double * d_tmpE;


    double alpha, beta;

    printf(" Computing 0e-Element for Normal-Ordering With CuTC\n");


    cublasHandle_t myhandle;

    checkCublasErrors(cublasCreate(&myhandle), "cublasCreate", __FILE__, __LINE__);
    cublasSetPointerMode(myhandle, CUBLAS_POINTER_MODE_DEVICE);


    n_mo2 = n_mo * n_mo;

    size_wr1 = n_grid1 * sizeof(double);
    size_mos_in_r = n_grid1 * n_mo * sizeof(double);
    size_int2 = 3 * n_grid1 * n_mo2 * sizeof(double);

    size_0e = sizeof(double);

    sizeO = n_grid1 * sizeof(double);
    sizeJ = 3 * n_grid1 * sizeof(double);
    sizeM = 3 * n_grid1 * sizeof(double);
    sizeS = n_grid1 * sizeof(double);
    sizeL = 3 * n_grid1 * n_mo * sizeof(double);
    sizeR = 3 * n_grid1 * n_mo * sizeof(double);


    checkCudaErrors(cudaMalloc((void**)&d_wr1, size_wr1), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_mos_l_in_r, size_mos_in_r), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_mos_r_in_r, size_mos_in_r), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_int2_grad1_u12, size_int2), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_no_0e, size_0e), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc((void**)&d_tmpL, sizeL), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpR, sizeR), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpO, sizeO), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpJ, sizeJ), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpM, sizeM), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_tmpS, sizeS), "cudaMalloc", __FILE__, __LINE__);


    checkCudaErrors(cudaMemcpy(d_wr1, h_wr1, size_wr1, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_mos_l_in_r, h_mos_l_in_r, size_mos_in_r, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_mos_r_in_r, h_mos_r_in_r, size_mos_in_r, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_int2_grad1_u12, h_int2_grad1_u12, size_int2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);


    if(ne_a == ne_b) {

        // Closed-Shell

        no_1e_tmpL_cs(n_grid1, n_mo, ne_b, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpL);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_1e_tmpR_cs(n_grid1, n_mo, ne_b, d_mos_r_in_r, d_int2_grad1_u12, d_tmpR);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    
    } else {

        // Open-Shell

        no_1e_tmpL_os(n_grid1, n_mo, ne_b, ne_a, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpL);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_1e_tmpR_os(n_grid1, n_mo, ne_b, ne_a, d_mos_r_in_r, d_int2_grad1_u12, d_tmpR);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

    }

    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

    checkCublasErrors(cublasDdot(myhandle, 3*n_grid1*ne_b,
                                 &d_tmpL[0], 1,
                                 &d_tmpR[0], 1,
                                 &d_no_0e[0]), "cublasDdot", __FILE__, __LINE__);

    checkCudaErrors(cudaMemcpy(h_no_0e, d_no_0e, size_0e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);


    if(ne_a == ne_b) {

        // Closed-Shell

        no_tmpO_cs(n_grid1, ne_b, d_mos_l_in_r, d_mos_r_in_r, d_tmpO);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_tmpJ_cs(n_grid1, n_mo, ne_b, d_int2_grad1_u12, d_tmpJ);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_tmpM_cs(n_grid1, n_mo, ne_b, d_mos_l_in_r, d_mos_r_in_r, d_int2_grad1_u12, d_tmpM);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

        no_tmpS_cs(n_grid1, n_mo, ne_b, d_int2_grad1_u12, d_tmpJ, d_tmpS);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    
    } else {

        // Open-Shell

        no_tmpO_os(n_grid1, ne_b, ne_a, d_mos_l_in_r, d_mos_r_in_r, d_tmpO);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_tmpJ_os(n_grid1, n_mo, ne_b, ne_a, d_int2_grad1_u12, d_tmpJ);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

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

    no_0e_tmpU(n_grid1, n_blocks, s_blocks, d_wr1, d_tmpO, d_tmpS, d_tmpJ, d_tmpM, d_tmpE);
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
        double * d_tmpX;
        double * d_tmpY;

        size_t sizeX = 3 * n_grid1 * ne_a * sizeof(double);
        size_t sizeY = 3 * n_grid1 * ne_a * sizeof(double);

        checkCudaErrors(cudaMalloc((void**)&d_no_0e_os, size_0e), "cudaMalloc", __FILE__, __LINE__);

        checkCudaErrors(cudaMalloc((void**)&d_tmpX, sizeX), "cudaMalloc", __FILE__, __LINE__);
        checkCudaErrors(cudaMalloc((void**)&d_tmpY, sizeY), "cudaMalloc", __FILE__, __LINE__);

        no_0e_tmpX_os(n_grid1, n_mo, ne_b, ne_a, d_wr1, d_mos_l_in_r, d_int2_grad1_u12, d_tmpX);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        no_0e_tmpY_os(n_grid1, n_mo, ne_b, ne_a, d_mos_r_in_r, d_int2_grad1_u12, d_tmpY);
        checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);

        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);

        checkCublasErrors(cublasDdot(myhandle, 3*n_grid1*ne_a,
                                     &d_tmpX[0], 1,
                                     &d_tmpY[0], 1,
                                     &d_no_0e_os[0]), "cublasDdot", __FILE__, __LINE__);

        checkCudaErrors(cudaFree(d_tmpX), "cudaFree", __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_tmpY), "cudaFree", __FILE__, __LINE__);

        checkCudaErrors(cudaMemcpy(h_no_0e_os, d_no_0e_os, size_0e, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);

        checkCudaErrors(cudaFree(d_no_0e_os), "cudaFree", __FILE__, __LINE__);

        h_no_0e[0] -= 2.0 * h_no_0e_os[0];

    }



    checkCudaErrors(cudaFree(d_wr1), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_mos_l_in_r), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_mos_r_in_r), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_int2_grad1_u12), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_no_0e), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaFree(d_tmpL), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpR), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpO), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpJ), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpM), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_tmpS), "cudaFree", __FILE__, __LINE__);

    checkCublasErrors(cublasDestroy(myhandle), "cublasDestroy", __FILE__, __LINE__);


    return 0;

}
