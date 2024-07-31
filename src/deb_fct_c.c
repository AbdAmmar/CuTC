#include <stdio.h>
#include <cublas_v2.h>


int deb_fct_c(int n, double *Tin, double *Tout) {

    int i, j;

    printf("Hello from C fct \n");

    printf("n = %d\n", n);

    printf("Tin:\n");
    for(i = 0; i < n; i++) {
        printf("Tin[%d] = %f\n", i, Tin[i]);
    } 

    for(i = 0; i < n; i++) {
        Tout[i] = Tin[i] + (double) i;
    } 

    int rowsA = 3, colsA = 2;
    int rowsB = 2, colsB = 3;

    double A[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double B[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double C[9];

    printf("A\n");
    for (i = 0; i < rowsA; ++i) {
        for (j = 0; j < colsA; ++j) {
            printf("%f ", A[i + j * rowsA]);
        }
        printf("\n");
    }

    printf("B\n");
    for (i = 0; i < rowsB; ++i) {
        for (j = 0; j < colsB; ++j) {
            printf("%f ", B[i + j * rowsB]);
        }
        printf("\n");
    }

    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, rowsA * colsA * sizeof(double));
    cudaMalloc((void**)&d_B, rowsB * colsB * sizeof(double));
    cudaMalloc((void**)&d_C, rowsA * colsB * sizeof(double));

    cudaMemcpy(d_A, A, rowsA * colsA * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, rowsB * colsB * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    double alpha = 1.0;
    double beta = 0.0;
    
    cublasDgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N, 
                rowsA, colsB, colsA, 
                &alpha, 
                &d_A[0], rowsA, 
                &d_B[0], rowsB, 
                &beta, 
                &d_C[0], rowsA);
    
    cublasDestroy(handle);

    cudaMemcpy(C, d_C, rowsA * colsB * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    printf("C\n");
    for (i = 0; i < rowsA; ++i) {
        for (j = 0; j < colsB; ++j) {
            printf("%f ", C[i + j * rowsA]);
        }
        printf("\n");
    }


    return 0;
}


