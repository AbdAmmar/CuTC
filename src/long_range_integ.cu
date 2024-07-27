

__global__ void int_long_range_kernel(int n_grid2, int ao_num,
                                      double *r2, double *wr2, double* aos_data2,
                                      double *int_fct_long_range) {


    int i_grid2;
    int i_ao, j_ao;
    int ll0, ll1, ll2;

    double r2_x, r2_y, r2_z;
    double wr2_tmp;
    double ao_val_i, ao_val_j;


    i_grid2 = blockIdx.x * blockDim.x + threadIdx.x;

    ll0 = ao_num * n_grid2;

    while(i_grid2 < n_grid2) {

        r2_x = r2[i_grid2          ];
        r2_y = r2[i_grid2+  n_grid2];
        r2_z = r2[i_grid2+2*n_grid2];

        wr2_tmp = wr2[i_grid2];

        for(i_ao = 0; i_ao < ao_num; i_ao++) {

            ll1 = i_ao * ll0;

            ao_val_i = aos_data2[i_ao];

            for(j_ao = 0; j_ao < ao_num; j_ao++) {

                ll2 = i_grid2 + j_ao * n_grid2 + ll1;

                ao_val_j = aos_data2[j_ao];

                int_fct_long_range[ll2] = wr2_tmp * ao_val_i * ao_val_j;

            } // j_ao
        } // i_ao

        i_grid2 += blockDim.x * gridDim.x;

    } // i_grid2

}



extern "C" void int_long_range(int nBlocks, int blockSize,
                               int n_grid2, int ao_num,
                               double *r2, double *wr2, double* aos_data2,
                               double *int_fct_long_range) {

    int_long_range_kernel<<<nBlocks, blockSize>>>(n_grid2, ao_num, r2, wr2, aos_data2, int_fct_long_range);

}


