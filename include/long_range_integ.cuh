#ifndef LONG_RANGE_INTEG

#define LONG_RANGE_INTEG

__global__ void int_long_range_kernel(int jj0, int n_grid2_eff, int n_grid2_tot,
                                      int n_grid2, int n_ao, double *wr2, double* aos_data2, double *int_fct_long_range);


#endif

