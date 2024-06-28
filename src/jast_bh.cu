

__global__ void tc_int_bh_kernel(int n_grid1, int n_grid2, int n_nuc, int size_bh,
                                 double *r1, double *r2, double *rn,
                                 double *c_bh, int *m_bh, int *n_bh, int *o_bh,
                                 double *grad1_u12) {

    /*
        grad1_u12[1] =      [grad1 u(r1,r2)]_x1
        grad1_u12[2] =      [grad1 u(r1,r2)]_y1
        grad1_u12[3] =      [grad1 u(r1,r2)]_z1
        grad1_u12[4] = -0.5 [grad1 u(r1,r2)]^2
    */

    int i_grid1, i_grid2;
    int i_nuc;
    int i_bh;
    int i;

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


    i_grid1 = blockIdx.x * blockDim.x + threadIdx.x ;

    if(i_grid1 < n_grid1) {

        r1_x = r1[i_grid1          ];
        r1_y = r1[i_grid1+  n_grid1];
        r1_z = r1[i_grid1+2*n_grid1];

        for(i_grid2 = 0; i_grid2 < n_grid2; i_grid2++) {

            grad1_u12[i_grid1*n_grid1+i_grid2          ] = 0.0;
            grad1_u12[i_grid1*n_grid1+i_grid2+  n_grid2] = 0.0;
            grad1_u12[i_grid1*n_grid1+i_grid2+2*n_grid2] = 0.0;

            r2_x = r2[i_grid2          ];
            r2_y = r2[i_grid2+  n_grid2];
            r2_z = r2[i_grid2+2*n_grid2];

            // e1-e2 term
            dx = r1_x - r2_x;
            dy = r1_y - r2_y;
            dz = r1_z - r2_z;
            dist = dx * dx + dy * dy + dz * dz;
            if(dist < 1e-15) {
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

                rn_x = rn[i_nuc        ];
                rn_y = rn[i_nuc+  n_nuc];
                rn_z = rn[i_nuc+2*n_nuc];

                // e1-n term
                dx = r1_x - rn_x;
                dy = r1_y - rn_y;
                dz = r1_z - rn_z;
                dist = dx * dx + dy * dy + dz * dz;
                if(dist < 1e-15) {
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
                if(dist < 1e-15) {
                    dist = sqrt(dist);
                    f2n  = dist / (1.0 + dist);
                } else {
                    f2n  = 0.0;
                }

                for(i_bh = 0; i_bh < size_bh; i_bh++) {

                    c = c_bh[i_bh + size_bh*i_nuc];
                    if(fabs(c) < 1e-10)
                        break;

                    m = m_bh[i_bh + size_bh*i_nuc];
                    n = n_bh[i_bh + size_bh*i_nuc];
                    o = o_bh[i_bh + size_bh*i_nuc];

                    // TODO remove
                    if(m == n)
                        c *= 0.5;

                    f1n_m = 1.0;
                    f2n_m = 1.0;
                    if(m > 0) {
                        f1n_mm1 = 1.0;
                        for(i = 0; i < m-1; i++) {
                            f1n_mm1 *= f1n;
                            f2n_m   *= f2n;
                        }
                        f1n_m = f1n_mm1 * f1n;
                        f2n_m = f2n_m   * f2n;
                    }

                    f1n_n = 1.0;
                    f2n_n = 1.0;
                    if(n > 0) {
                        f1n_nm1 = 1.0;
                        for(i = 0; i < n-1; i++) {
                            f1n_nm1 *= f1n;
                            f2n_n   *= f2n;
                        }
                        f1n_n = f1n_nm1 * f1n;
                        f2n_n = f2n_n   * f2n;
                    }

                    tmp1 = 0.0;
                    tmp2 = 0.0;

                    if(m > 0)
                        tmp1 += __int2double_rn(m) * f1n_mm1 * f2n_n;
                    if(n > 0)
                        tmp1 += __int2double_rn(n) * f1n_nm1 * f2n_m;

                    if(o > 0) {
                        g12_om1 = 1.0;
                        for(i = 0; i < o-1; i++) {
                            g12_om1 *= g12;
                        }
                        g12_o = g12_om1 * g12;

                        tmp2 = c * __int2double_rn(o) * g12_om1 * (f1n_m * f2n_n + f1n_n * f2n_m);
                        tmp1 = c * tmp1 * g12_o;
                    } else {
                        tmp1 *= c;
                    }

                    grad1_u12[i_grid1*n_grid1+i_grid2          ] += tmp1 * f1nx + tmp2 * g12x;
                    grad1_u12[i_grid1*n_grid1+i_grid2+  n_grid2] += tmp1 * f1ny + tmp2 * g12y;
                    grad1_u12[i_grid1*n_grid1+i_grid2+2*n_grid2] += tmp1 * f1nz + tmp2 * g12z;

                } // i_bh

            } // i_nuc

        } // i_grid2

        for(i_grid2 = 0; i_grid2 < n_grid2; i_grid2++) {
            grad1_u12[i_grid1*n_grid1+i_grid2+3*n_grid2] = -0.5 * ( grad1_u12[i_grid1*n_grid1+i_grid2          ] * grad1_u12[i_grid1*n_grid1+i_grid2          ] 
                                                                  + grad1_u12[i_grid1*n_grid1+i_grid2+  n_grid2] * grad1_u12[i_grid1*n_grid1+i_grid2+  n_grid2] 
                                                                  + grad1_u12[i_grid1*n_grid1+i_grid2+2*n_grid2] * grad1_u12[i_grid1*n_grid1+i_grid2+2*n_grid2] ) ;
        }

    } // i_grid1

}


// int tc_int_bh(void) {

    //int n_grid1, n_grid2; 
    ////int ao_num;
    //int n_nuc;
    //int size_bh;

    //int *h_m_bh, *h_n_bh, *h_o_bh;
    //double *h_c_bh; 

    //double *h_r1, *h_r2, *h_rn;
    ////double *h_aos_data1, *h_aos_data2;

    ////double *h_int2_grad1_u12;
    ////double *h_tc_int_2e_ao;

    //int i, j;

    //// ao_num  = 50;
    //n_grid1 = 1000;
    //n_grid2 = 10000;
    //n_nuc = 5;
    //size_bh = 10;

    //h_r1 = (double*) malloc(size_r1);
    //h_r2 = (double*) malloc(size_r2);
    //h_rn = (double*) malloc(size_rn);

    //h_c_bh = (double*) malloc(size_jbh1);
    //h_m_bh = (int*) malloc(size_jbh2);
    //h_n_bh = (int*) malloc(size_jbh2);
    //h_o_bh = (int*) malloc(size_jbh2);

    //for(i = 0; i < n_grid1; i++) {
    //    h_r1[i          ] = 0.1;
    //    h_r1[i+  n_grid1] = 0.1;
    //    h_r1[i+2*n_grid1] = 0.1;
    //}
    //for(i = 0; i < n_grid2; i++) {
    //    h_r2[i          ] = 0.2;
    //    h_r2[i+  n_grid2] = 0.2;
    //    h_r2[i+2*n_grid2] = 0.2;
    //}
    //for(i = 0; i < n_nuc; i++) {
    //    h_rn[i        ] = 0.3;
    //    h_rn[i+  n_nuc] = 0.3;
    //    h_rn[i+2*n_nuc] = 0.3;
    //}
    //for (j = 0; j < n_nuc; j++) {
    //    for (i = 0; i < size_bh; i++) {
    //        h_c_bh[i + j*n_nuc] = 0.5;
    //        h_m_bh[i + j*n_nuc] = 2;
    //        h_n_bh[i + j*n_nuc] = 3;
    //        h_o_bh[i + j*n_nuc] = 4;
    //    }
    //}


extern "C" void tc_int_bh(int n_grid1, int n_grid2, int ao_num, int n_nuc, int size_bh,
                          int *h_m_bh, int *h_n_bh, int *h_o_bh, double *h_c_bh,
                          double *h_r1, double *h_r2, double *h_rn,
                          double *h_aos_data1, double *h_aos_data2,
                          double *h_int2_grad1_u12, double *h_tc_int_2e_ao) {

    int *d_m_bh, *d_n_bh, *d_o_bh;
    double *d_c_bh; 

    double *d_r1, *d_r2, *d_rn;

    //double *d_aos_data1, *d_aos_data2;


    double *d_grad1_u12;
    //double *d_int2_grad1_u12;
    //double *d_tc_int_2e_ao;

    size_t size_r1, size_r2, size_rn;
    //size_t size_aos_r1, size_aos_r2;
    size_t size_r12;
    //size_t size_int1, size_int2;
    size_t size_jbh1, size_jbh2;

    int threadsPerBlock, numBlocks;

    size_r1 = 3 * n_grid1 * sizeof(double);
    size_r2 = 3 * n_grid2 * sizeof(double);
    size_rn = 3 * n_nuc   * sizeof(double);

    size_r12 = 4 * n_grid1 * n_grid2 * sizeof(double);

    //size_aos_r1 = 4 * n_grid1 * ao_num * sizeof(double);
    //size_aos_r2 = 4 * n_grid2 * ao_num * sizeof(double);

    //size_int1 = 4 * n_grid2 * ao_num * ao_num * sizeof(double);
    //size_int2 = ao_num * ao_num * ao_num * ao_num * sizeof(double);

    size_jbh1 = size_bh * sizeof(double);
    size_jbh2 = size_bh * sizeof(int);


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

    tc_int_bh_kernel<<<numBlocks, threadsPerBlock>>>(n_grid1, n_grid2, n_nuc, size_bh,
                                                     d_r1, d_r2, d_rn,
                                                     d_c_bh, d_m_bh, d_n_bh, d_o_bh,
                                                     d_grad1_u12);



    //cudaMemcpy(h_int2_grad1_u12, d_int2_grad1_u12, size_int1, cudaMemcpyDeviceToHost);

    cudaFree(d_r1);
    cudaFree(d_r2);
    cudaFree(d_rn);

}


