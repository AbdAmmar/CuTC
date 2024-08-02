

///* compute a**b, where b in [0,10] */
__device__ double powd_int(double a, int b) {

    double r;

    double a2, a3, a4;

    if (b == 0) {
        r = 1.0;
    } else if (b == 1) {
        r = a;
    } else if (b == 2) {
        r = a * a;
    } else if (b == 3) {
        r = a * a * a;
    } else if (b == 4) {
        a2 = a * a;
        r = a2 * a2;
    } else if (b == 5) {
        a2 = a * a;
        r = a * a2 * a2;
    } else if (b == 6) {
        a2 = a * a;
        r = a2 * a2 * a2;
    } else if (b == 7) {
        a2 = a * a;
        a3 = a2 * a;
        r = a2 * a2 * a3;
    } else if (b == 8) {
        a2 = a * a;
        a4 = a2 * a2;
        r = a4 * a4;
    } else if (b == 9) {
        a3 = a * a * a;
        r = a3 * a3 * a3;
    } else if (b == 10) {
        a2 = a * a;
        a4 = a2 * a2;
        r = a2 * a4 * a4;
    }

    return r;
}


__global__ void tc_int_bh_kernel(int ii0, int n_grid1_eff, int n_grid1_tot,
                                 int n_grid1, int n_grid2, int n_nuc, int size_bh,
                                 double *r1, double *r2, double *rn, 
                                 double *c_bh, int *m_bh, int *n_bh, int *o_bh,
                                 double *grad1_u12) {

    /*
        grad1_u12[1] =      [grad1 u(r1,r2)]_x1
        grad1_u12[2] =      [grad1 u(r1,r2)]_y1
        grad1_u12[3] =      [grad1 u(r1,r2)]_z1
        grad1_u12[4] = -0.5 [grad1 u(r1,r2)]^2
    */


//    extern __shared__ char shared_data[];
//    double *shared_c = (double*) shared_data; 
//    int *shared_m = (int*) (shared_data + n_nuc * size_bh * sizeof(double)); 
//    int *shared_n = (int*) (shared_data + n_nuc * size_bh * (sizeof(double)+sizeof(int))); 
//    int *shared_o = (int*) (shared_data + n_nuc * size_bh * (sizeof(double)+2*sizeof(int))); 

    int i_grid1, i_grid2;
    int ii_grid1, ii_grid2, ii_nuc, ii_12;
    int i_nuc;
    int i_bh;
    int ii;
    int jj;
    int kk;

    int m, n, o;
    double c;

    double dx, dy, dz, dist;
    double r1_x, r1_y, r1_z;
    double r2_x, r2_y, r2_z;
    double rn_x, rn_y, rn_z;

    double g12, g12x, g12y, g12z;
    double f1n, f1nx, f1ny, f1nz;
    double f2n;
    double f1n_mm1, f1n_m, f1n_nm1, f1n_n;
    double f2n_m, f2n_n;
    double g12_om1, g12_o;
    double tmp1, tmp2;


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x;

//    i_grid2 = blockIdx.y * blockDim.y + threadIdx.y;
//    if((i_grid1 < n_grid1_eff) && (i_grid2 < n_grid2)) {
//        for(i_nuc = 0; i_nuc < n_nuc; i_nuc++) {
//            ii_nuc = size_bh * i_nuc;
//            for(i_bh = 0; i_bh < size_bh; i_bh++) {
//                kk = i_bh + ii_nuc;
//                shared_c[kk] = c_bh[kk];
//                shared_m[kk] = m_bh[kk];
//                shared_n[kk] = n_bh[kk];
//                shared_o[kk] = o_bh[kk];
//            }
//        }
//    }
//    __syncthreads();


    ii_12 = n_grid1_tot * n_grid2;

    while(i_grid1 < n_grid1_eff) {

        ii = 3 * (ii0 + i_grid1);
        r1_x = r1[ii    ];
        r1_y = r1[ii + 1];
        r1_z = r1[ii + 2];

        ii_grid1 = i_grid1 * n_grid2;

        i_grid2 = blockIdx.y * blockDim.y + threadIdx.y;
        while(i_grid2 < n_grid2) {

            ii_grid2 = ii_grid1 + i_grid2;

            grad1_u12[ii_grid2          ] = 0.0;
            grad1_u12[ii_grid2 +   ii_12] = 0.0;
            grad1_u12[ii_grid2 + 2*ii_12] = 0.0;

            jj = 3 * i_grid2;
            r2_x = r2[jj    ];
            r2_y = r2[jj + 1];
            r2_z = r2[jj + 2];

            // e1-e2 term
            dx = r1_x - r2_x;
            dy = r1_y - r2_y;
            dz = r1_z - r2_z;
            dist = dx * dx + dy * dy + dz * dz;
            if(dist > 1e-15) {
                dist = sqrt(dist);
                tmp1 = 1.0 / (1.0 + dist);
                g12  = dist * tmp1;
                tmp2 = tmp1 * tmp1 / dist;
                g12x = tmp2 * dx;
                g12y = tmp2 * dy;
                g12z = tmp2 * dz;
            } else {
                g12  = 0.0;
                g12x = 0.0;
                g12y = 0.0;
                g12z = 0.0;
            }
            
            for(i_nuc = 0; i_nuc < n_nuc; i_nuc++) {

                rn_x = rn[3*i_nuc  ];
                rn_y = rn[3*i_nuc+1];
                rn_z = rn[3*i_nuc+2];

                // e1-n term
                dx = r1_x - rn_x;
                dy = r1_y - rn_y;
                dz = r1_z - rn_z;
                dist = dx * dx + dy * dy + dz * dz;
                if(dist > 1e-15) {
                    dist = sqrt(dist);
                    tmp1 = 1.0 / (1.0 + dist);
                    f1n  = dist * tmp1;
                    tmp2 = tmp1 * tmp1 / dist;
                    f1nx = tmp2 * dx;
                    f1ny = tmp2 * dy;
                    f1nz = tmp2 * dz;
                } else {
                    f1n  = 0.0;
                    f1nx = 0.0;
                    f1ny = 0.0;
                    f1nz = 0.0;
                }

                // e2-n term
                dx = r2_x - rn_x;
                dy = r2_y - rn_y;
                dz = r2_z - rn_z;
                dist = dx * dx + dy * dy + dz * dz;
                if(dist > 1e-15) {
                    dist = sqrt(dist);
                    f2n  = dist / (1.0 + dist);
                } else {
                    f2n  = 0.0;
                }

                ii_nuc = size_bh * i_nuc;

                for(i_bh = 0; i_bh < size_bh; i_bh++) {

                    kk = i_bh + ii_nuc;

                    //c = shared_c[kk];
                    //if(fabs(c) < 1e-10)
                    //    continue;
                    //m = shared_m[kk];
                    //n = shared_n[kk];
                    //o = shared_o[kk];

                    c = c_bh[kk];
                    if(fabs(c) < 1e-10)
                        continue;
                    m = m_bh[kk];
                    n = n_bh[kk];
                    o = o_bh[kk];

                    // TODO remove
                    if(m == n)
                        c *= 0.5;

                    f1n_m = 1.0;
                    f2n_m = 1.0;
                    if(m > 0) {
                        f1n_mm1 = powd_int(f1n, m-1);
                        f1n_m = f1n_mm1 * f1n;
                        f2n_m = powd_int(f2n, m);
                    }

                    f1n_n = 1.0;
                    f2n_n = 1.0;
                    if(n > 0) {
                        f1n_nm1 = powd_int(f1n, n-1);
                        f1n_n = f1n_nm1 * f1n;
                        f2n_n = powd_int(f2n, n);
                    }

                    tmp1 = 0.0;
                    tmp2 = 0.0;

                    if(m > 0)
                        tmp1 += __int2double_rn(m) * f1n_mm1 * f2n_n;
                    if(n > 0)
                        tmp1 += __int2double_rn(n) * f1n_nm1 * f2n_m;

                    if(o > 0) {

                        g12_om1 = powd_int(g12, o-1);
                        g12_o = g12_om1 * g12;

                        tmp2 = c * __int2double_rn(o) * g12_om1 * (f1n_m * f2n_n + f1n_n * f2n_m);
                        tmp1 = c * tmp1 * g12_o;
                    } else {
                        tmp1 *= c;
                    }

                    grad1_u12[ii_grid2          ] += tmp1 * f1nx + tmp2 * g12x;
                    grad1_u12[ii_grid2 +   ii_12] += tmp1 * f1ny + tmp2 * g12y;
                    grad1_u12[ii_grid2 + 2*ii_12] += tmp1 * f1nz + tmp2 * g12z;

                } // i_bh

            } // i_nuc

            grad1_u12[ii_grid2 + 3*ii_12] = -0.5 * ( grad1_u12[ii_grid2          ] * grad1_u12[ii_grid2          ]
                                                   + grad1_u12[ii_grid2 +   ii_12] * grad1_u12[ii_grid2 +   ii_12]
                                                   + grad1_u12[ii_grid2 + 2*ii_12] * grad1_u12[ii_grid2 + 2*ii_12] );

            i_grid2 += blockDim.y * gridDim.y;

        } // i_grid2

        i_grid1 += blockDim.x * gridDim.x;

    } // i_grid1

}



extern "C" void tc_int_bh(dim3 dimGrid, dim3 dimBlock,
                          int ii0, int n_grid1_eff, int n_grid1_tot,
                          int n_grid1, int n_grid2, int n_nuc, int size_bh,
                          double *r1, double *r2, double *rn, 
                          double *c_bh, int *m_bh, int *n_bh, int *o_bh,
                          double *grad1_u12) {

    //size_t size_sh;
    //size_sh = n_nuc * size_bh * (sizeof(double) + 3 * sizeof(int));
    //tc_int_bh_kernel<<<dimGrid, dimBlock, size_sh>>>(ii0, n_grid1_eff, n_grid1_tot,
    tc_int_bh_kernel<<<dimGrid, dimBlock>>>(ii0, n_grid1_eff, n_grid1_tot,
                                                     n_grid1, n_grid2, n_nuc, size_bh,
                                                     r1, r2, rn, 
                                                     c_bh, m_bh, n_bh, o_bh,
                                                     grad1_u12);
               
}


