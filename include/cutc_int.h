#ifndef CUTC_INT
#define CUTC_INT

extern void checkCudaErrors(cudaError_t err, const char * msg, const char * file, int line);
extern void checkCublasErrors(cublasStatus_t status, const char * msg, const char * file, int line);


extern void get_int2_grad1_u12_ao(dim3 dimGrid, dim3 dimBlock, int offset_dev,
                                  int n_grid1_dev, int n_grid2, int n_ao, int n_nuc, int size_bh,
                                  double * r1, double * r2, double * wr2, double * rn, double * aos_data2,
                                  double * c_bh, int * m_bh, int * n_bh, int * o_bh,
                                  double * int2_grad1_u12_ao);

extern void get_int_2e_ao(int n_grid1, int n_ao, double * wr1, double * aos_data1,
                          double * int2_grad1_u12_ao, double * tc_int_2e_ao);

extern int checkPeerToPeerSupport(int device1, int device2);

#endif
