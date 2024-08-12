#ifndef CUTC_NO

#define CUTC_NO

/* ERROR HANDLING */

extern void checkCudaErrors(cudaError_t err, const char * msg, const char * file, int line);
extern void checkCublasErrors(cublasStatus_t status, const char * msg, const char * file, int line);



/* CLOSED-SHELL kernels */

extern void no_tmpO_cs(int n_grid1, int ne_b, double * mos_l_in_r, double * mos_r_in_r, double * tmpO);
extern void no_tmpJ_cs(int n_grid1, int n_mo, int ne_b, double * int2_grad1_u12, double * tmpJ);
extern void no_tmpC2_cs(int n_grid1, int n_mo, int ne_b, double * int2_grad1_u12, double * tmpC2);
extern void no_tmpM_cs(int n_grid1, int n_mo, int ne_b, double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12, double * tmpM);
extern void no_tmpS_cs(int n_grid1, int n_mo, int ne_b, double * int2_grad1_u12, double * tmpJ, double * tmpS);

extern void no_2e_tmpA_cs(int n_grid1, int n_mo, int ne_b, double * wr1, double * mos_l_in_r, double * int2_grad1_u12, double * tmpA);
extern void no_2e_tmpB_cs(int n_grid1, int n_mo, int ne_b, double * wr1, double * mos_r_in_r, double * int2_grad1_u12, double * tmpB);

extern void no_1e_tmpL_cs(int n_grid1, int n_mo, int ne_b, double * mos_l_in_r, double * int2_grad1_u12, double * tmpL);
extern void no_1e_tmpR_cs(int n_grid1, int n_mo, int ne_b, double * mos_r_in_r, double * int2_grad1_u12, double * tmpR);
extern void no_1e_tmpE_cs(int n_grid1, int n_mo, int ne_b, double * wr1, double * mos_l_in_r, double * int2_grad1_u12,
                          double * tmpJ, double * tmpL, double * tmpE);
extern void no_1e_tmpF_cs(int n_grid1, int n_mo, int ne_b, double * wr1, double * mos_r_in_r, double * int2_grad1_u12,
                          double * tmpS, double * tmpJ, double * tmpR, double * tmpF);

extern void no_0e_tmpL_cs(int n_grid1, int n_mo, int ne_b, double * wr1, double * mos_l_in_r, double * int2_grad1_u12, double * tmpL);
extern void no_0e_tmpR_cs(int n_grid1, int n_mo, int ne_b, double * mos_r_in_r, double * int2_grad1_u12, double * tmpR);


/* OEPN-SHELL kernels */

extern void no_tmpO_os(int n_grid1, int ne_b, int ne_a, double * mos_l_in_r, double * mos_r_in_r, double * tmpO);
extern void no_tmpJ_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * int2_grad1_u12, double * tmpJ);
extern void no_tmpC2_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * int2_grad1_u12, double * tmpC2);
extern void no_tmpM_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12, double * tmpM);
extern void no_tmpS_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * int2_grad1_u12, double * tmpJ, double * tmpS);

extern void no_2e_tmpA_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * wr1, double * mos_l_in_r, double * int2_grad1_u12, double * tmpA);
extern void no_2e_tmpB_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * wr1, double * mos_r_in_r, double * int2_grad1_u12, double * tmpB);

extern void no_1e_tmpL_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * mos_l_in_r, double * int2_grad1_u12, double * tmpL);
extern void no_1e_tmpR_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * mos_r_in_r, double * int2_grad1_u12, double * tmpR);
extern void no_1e_tmpE_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * wr1, double * mos_l_in_r, double * int2_grad1_u12,
                          double * tmpJ, double * tmpL, double * tmpE);
extern void no_1e_tmpF_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * wr1, double * mos_r_in_r, double * int2_grad1_u12,
                          double * tmpS, double * tmpJ, double * tmpR, double * tmpF);
extern void no_1e_tmpG_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * wr1, double * mos_l_in_r, double * int2_grad1_u12, double * tmpG);
extern void no_1e_tmpH_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * mos_r_in_r, double * int2_grad1_u12, double * tmpH);

extern void no_0e_tmpL_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * wr1, double * mos_l_in_r, double * int2_grad1_u12, double * tmpL);
extern void no_0e_tmpR_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * mos_r_in_r, double * int2_grad1_u12, double * tmpR);
extern void no_0e_tmpG_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * wr1, double * mos_l_in_r, double * int2_grad1_u12, double * tmpG);
extern void no_0e_tmpH_os(int n_grid1, int n_mo, int ne_b, int ne_a, double * mos_r_in_r, double * int2_grad1_u12, double * tmpH);




/* COMMON-SHELL kernels */

extern void no_2e_tmpC1(int n_grid1, int n_mo, double wr1, double * mos_l_in_r, double * mos_r_in_r, double * int2_grad1_u12,
                        double * tmpJ, double * tmpO, double * tmpA, double * tmpB, double * tmpC1);
extern void no_2e_tmpD2(int n_grid1, int n_mo, double * wr1, double * mos_l_in_r, double * mos_r_in_r, double * tmpD2);

extern void no_1e_tmpD(int n_grid1, double * wr1, double * tmpO, double * tmpJ, double * tmpM, double * tmpD);

extern void no_0e_tmpU(int n_grid1, int n_blocks, int s_blocks, double * wr1, double * tmpO, double * tmpS, double * tmpJ, double * tmpM, double * tmpE);

extern void trans_inplace(double * data, int size);

extern void trans_pqst_psqt_inplace(int size, double * data);


#endif
