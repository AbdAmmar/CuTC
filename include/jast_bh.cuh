#ifndef JAST_BH
#define JAST_BH

__global__ void cutc_int_bh_kernel(int ii0, int n_grid1_eff, int n_grid1_tot,
                                   int jj0, int n_grid2_eff, int n_grid2_tot,
                                   int n_nuc, int size_bh,
                                   double *r1, double *r2, double *rn,
                                   double *c_bh, int *m_bh, int *n_bh, int *o_bh,
                                   double *grad1_u12);
                 
#endif


