# define N

// extern "C" void kernel_wrapper_(int *n_grid1, int *n_grid2, int *ao_num, int *n_nuc,
//                                 double *h_r1, double *h_r2, double *h_rn,
//                                 double *h_aos_data1, double *h_aos_data2,
//                                 int *n_bh, double *h_c_bh, int *h_m_bh, int *h_n_bh, int *h_o_bh,
//                                 double *h_int2_grad1_u12, double *h_tc_int_2e_ao)

int main(void ) {

    int n_grid1, n_grid2; 
    int ao_num;
    int n_nuc;
    int n_bh;

    int *h_m_bh, *h_n_bh, *h_o_bh;
    double *h_c_bh; 

    double *h_r1, *h_r2, *h_rn;
    double *h_aos_data1, *h_aos_data2;

    double *h_int2_grad1_u12;
    double *h_tc_int_2e_ao;

    double *d_r1, *d_r2, *d_rn;

    double *d_aos_data1, *d_aos_data2;

    double *d_int2_grad1_u12;
    double *d_tc_int_2e_ao;

    size_t size_r1, size_r2, size_rn;
    size_t size_aos_r1, size_aos_r2;
    size_t size_r12;
    size_t size_int1, size_int2;
    size_t size_jbh1, size_jbh2;

    int threadsPerBlock, numBlocks;

    int i, j;

    ao_num  = 50;
    n_grid1 = 1000;
    n_grid2 = 10000;
    n_nuc = 5;
    n_bh = 10;

    size_r1 = 3 * n_grid1 * sizeof(double);
    size_r2 = 3 * n_grid2 * sizeof(double);
    size_rn = 3 * n_nuc   * sizeof(double);

    size_r12 = 4 * n_grid1 * n_grid2 * sizeof(double);

    size_aos_r1 = 4 * n_grid1 * ao_num * sizeof(double);
    size_aos_r2 = 4 * n_grid2 * ao_num * sizeof(double);

    size_int1 = 4 * n_grid2 * ao_num * ao_num * sizeof(double);
    size_int2 = ao_num * ao_num * ao_num * ao_num * sizeof(double);

    size_jbh1 = n_bh * sizeof(double);
    size_jbh2 = n_bh * sizeof(int);

    h_r1 = (double*) malloc(size_r1);
    h_r2 = (double*) malloc(size_r2);
    h_rn = (double*) malloc(size_rn);

    h_c_bh = (double*) malloc(size_jbh1);
    h_m_bh = (int*) malloc(size_jbh2);
    h_n_bh = (int*) malloc(size_jbh2);
    h_o_bh = (int*) malloc(size_jbh2);

    for(i = 0; i < n_grid1; i++) {
        h_r1[i] = 0.1;
        h_r1[i+n_grid1] = 0.1;
        h_r1[i+2*n_grid1] = 0.1;
        //h_r1[i][1] = 0.1;
        // h_r1[i][2] = 0.1;
        // h_r1[i][3] = 0.1;
    }
    for(i = 0; i < n_grid2; i++) {
        h_r2[i][1] = 0.2;
        h_r2[i][2] = 0.2;
        h_r2[i][3] = 0.2;
    }
    for(i = 0; i < n_nuc; i++) {
        h_rn[i][1] = 0.3;
        h_rn[i][2] = 0.3;
        h_rn[i][3] = 0.3;
    }
    for (j = 0; j < n_nuc; j++) {
        for (i = 0; i < n_bh; i++) {
            h_c_bh[j][i] = 0.5;
            h_m_bh[j][i] = 2;
            h_n_bh[j][i] = 3;
            h_o_bh[j][i] = 4;
        }
    }

    cudaMalloc(&d_r1, size_r1);
    cudaMalloc(&d_r2, size_r2);
    cudaMalloc(&d_rn, size_rn);

    // cudaMalloc(&d_aos_data1, size_aos_r1);
    // cudaMalloc(&d_aos_data2, size_aos_r2);

    cudaMalloc(&d_grad1_u12, size_r12);

    // cudaMalloc(&d_int2_grad1_u12, size_int1);
    // cudaMalloc(&d_tc_int_2e_ao, size_int2);

    cudaMalloc(&d_c_bh, size_jbh1);
    cudaMalloc(&d_m_bh, size_jbh2);
    cudaMalloc(&d_n_bh, size_jbh2);
    cudaMalloc(&d_o_bh, size_jbh2);

    cudaMemcpy(d_r1, h_r1, size_r1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r2, h_r2, size_r2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rn, h_rn, size_rn, cudaMemcpyHostToDevice);

    // cudaMemcpy(d_aos_data1, h_aos_data1, size_aos_r1, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_aos_data2, h_aos_data2, size_aos_r2, cudaMemcpyHostToDevice);

    cudaMemcpy(d_c_bh, h_c_bh, size_jbh1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_bh, h_m_bh, size_jbh2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_bh, h_n_bh, size_jbh2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_o_bh, h_o_bh, size_jbh2, cudaMemcpyHostToDevice);


    threadsPerBlock = 16;
    numBlocks = (n_grid1 + threadsPerBlock - 1) / threadsPerBlock;

    tc_int_bh<<<numBlocks, threadsPerBlock>>>(n_grid1, n_grid2, n_nuc, n_bh,
                                              d_r1, d_r2, d_rn,
                                              d_c_bh, d_m_bh, d_n_bh, d_o_bh,
                                              d_grad1_u12);



    //cudaMemcpy(h_int2_grad1_u12, d_int2_grad1_u12, size_int1, cudaMemcpyDeviceToHost);

    cudaFree(d_r1);
    cudaFree(d_r2);
    cudaFree(d_rn);

    return 0;
}


