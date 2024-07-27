

__global__ void int_short_range_herm_kernel(int n_grid1, int ao_num, double *wr1, double* aos_data1, double *int_fct_short_range_herm) {


    int i_grid1;
    int i_ao, j_ao;
    int ll0, ll1, ll2;

    double wr1_tmp;
    double ao_val_i, ao_val_j;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

    ll0 = ao_num * n_grid1;

    while(i_grid1 < n_grid1) {

        wr1_tmp = wr1[i_grid1];

        for(i_ao = 0; i_ao < ao_num; i_ao++) {

            ll1 = i_ao * ll0;

            ao_val_i = aos_data1[i_ao];

            for(j_ao = 0; j_ao < ao_num; j_ao++) {

                ll2 = i_grid1 + j_ao * n_grid1 + ll1;

                ao_val_j = aos_data1[j_ao];

                int_fct_short_range_herm[ll2] = wr1_tmp * ao_val_i * ao_val_j;

            } // j_ao
        } // i_ao

        i_grid1 += blockDim.x * gridDim.x;

    } // i_grid1

}



extern "C" void int_short_range_herm(int nBlocks, int blockSize,
                                     int n_grid1, int ao_num, double *wr1, double* aos_data1,
                                     double *int_fct_short_range_herm) {

    int_short_range_herm_kernel<<<nBlocks, blockSize>>>(n_grid1, ao_num, wr1, aos_data1, int_fct_short_range_herm);

}


