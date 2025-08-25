#include <stdio.h>
#include <stdlib.h>
#include "cutc_int.h"

// Define dimensions for dummy data
#define N_GRID1 10
#define N_GRID2 20
#define N_AO 5
#define N_NUC 3
#define SIZE_BH 7

int main() {
    // Allocate host memory for input and output arrays
    double *r1 = (double *)malloc(N_GRID1 * 3 * sizeof(double));
    double *r2 = (double *)malloc(N_GRID2 * 3 * sizeof(double));
    double *wr1 = (double *)malloc(N_GRID1 * sizeof(double));
    double *wr2 = (double *)malloc(N_GRID2 * sizeof(double));
    double *rn = (double *)malloc(N_NUC * 3 * sizeof(double));
    double *aos_data1 = (double *)malloc(N_GRID1 * N_AO * sizeof(double));
    double *aos_data2 = (double *)malloc(N_GRID2 * N_AO * sizeof(double));
    double *c_bh = (double *)malloc(SIZE_BH * sizeof(double));
    int *m_bh = (int *)malloc(SIZE_BH * sizeof(int));
    int *n_bh = (int *)malloc(SIZE_BH * sizeof(int));
    int *o_bh = (int *)malloc(SIZE_BH * sizeof(int));
    double *int2_grad1_u12_ao = (double *)malloc(N_GRID1 * N_AO * 3 * sizeof(double));
    double *tc_int_2e_ao = (double *)malloc(N_AO * N_AO * sizeof(double));

    // Check for allocation errors
    if (!r1 || !r2 || !wr1 || !wr2 || !rn || !aos_data1 || !aos_data2 || !c_bh || !m_bh || !n_bh || !o_bh || !int2_grad1_u12_ao || !tc_int_2e_ao) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    // Define CUDA grid and block dimensions
    dim3 dimBlock;
    dimBlock.x = 16; dimBlock.y = 1; dimBlock.z = 1;
    dim3 dimGrid;
    dimGrid.x = (N_GRID1 * N_AO + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = 1;
    dimGrid.z = 1;


    printf("Calling get_int2_grad1_u12_ao...\n");
    get_int2_grad1_u12_ao(dimGrid, dimBlock, N_GRID1, N_GRID2, N_AO, N_NUC, SIZE_BH,
                          r1, r2, wr2, rn, aos_data2, c_bh, m_bh, n_bh, o_bh,
                          int2_grad1_u12_ao);
    printf("...done.\n");

    printf("Calling get_int_2e_ao...\n");
    get_int_2e_ao(N_GRID1, N_AO, wr1, aos_data1, int2_grad1_u12_ao, tc_int_2e_ao);
    printf("...done.\n");


    // Free allocated memory
    free(r1);
    free(r2);
    free(wr1);
    free(wr2);
    free(rn);
    free(aos_data1);
    free(aos_data2);
    free(c_bh);
    free(m_bh);
    free(n_bh);
    free(o_bh);
    free(int2_grad1_u12_ao);
    free(tc_int_2e_ao);

    printf("Smoke test passed successfully!\n");

    return 0;
}
