#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <time.h>


#include "cutc_int.h"


int cutc_int(int nxBlocks, int nyBlocks, int nzBlocks, int blockxSize, int blockySize, int blockzSize,
             int n_grid1, int n_grid2, int n_ao, int n_nuc, int size_bh,
             double * h_r1, double * h_wr1, double * h_r2, double * h_wr2, double * h_rn,
             double * h_aos_data1, double * h_aos_data2,
             double * h_c_bh, int * h_m_bh, int * h_n_bh, int * h_o_bh, 
             double * h_int2_grad1_u12_ao, double * h_int_2e_ao) {


    double * d_wr1;
    double * d_aos_data1;



    double *d_int_2e_ao;

    size_t size_wr1, size_r2, size_wr2, size_rn;
    size_t size_aos_r1, size_aos_r2;
    size_t size_int1, size_int2, size_int2_send;
    size_t size_jbh_d, size_jbh_i;

    int num_devices;
    int dev, peer;

    cudaEvent_t start_tot, stop_tot;
    cudaEvent_t start_loc, stop_loc;

    struct cudaDeviceProp deviceProp;

    dim3 dimGrid;
    dim3 dimBlock;


    float time_loc=0.0f;
    float time_tot=0.0f;
    float tHD=0.0f;

    clock_t time_req;



    printf(" Computing TC-Integrals With CuTC\n");

    //time_req = clock();


    cudaGetDeviceCount(&num_devices);
    printf(" Number of CUDA devices: %d\n", num_devices);

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0), "cudaGetDeviceProperties", __FILE__, __LINE__);
    printf(" Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf(" Max block dimensions: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf(" Max grid dimensions: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

    // Query P2P capability between all pairs of devices
    //for (dev = 0; dev < num_devices; dev++) {
    //    for (peer = 0; peer < num_devices; peer++) {
    //        if (dev != peer) {
    //            if (checkPeerToPeerSupport(dev, peer)) {
    //                printf("Device %d can access Device %d for P2P.\n", dev, peer);
    //            } else {
    //                printf("Device %d cannot access Device %d for P2P.\n", dev, peer);
    //                return 1;
    //            }
    //        }
    //    }
    //}


    checkCudaErrors(cudaEventCreate(&start_tot), "cudaEventCreate", __FILE__, __LINE__);
    checkCudaErrors(cudaEventCreate(&stop_tot), "cudaEventCreate", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(start_tot, 0), "cudaEventRecord", __FILE__, __LINE__);

    checkCudaErrors(cudaEventCreate(&start_loc), "cudaEventCreate", __FILE__, __LINE__);
    checkCudaErrors(cudaEventCreate(&stop_loc), "cudaEventCreate", __FILE__, __LINE__);



    dimGrid.x = nxBlocks;
    dimGrid.y = nyBlocks;
    dimGrid.z = nzBlocks;
    dimBlock.x = blockxSize;
    dimBlock.y = blockySize;
    dimBlock.z = blockzSize;

    if(dimBlock.x * dimBlock.y * dimBlock.z > deviceProp.maxThreadsPerBlock) {
        printf("Error: Too many threads per block!\n");
        return -1;
    }
    if(dimGrid.x > deviceProp.maxGridSize[0] || dimGrid.y > deviceProp.maxGridSize[1] || dimGrid.z > deviceProp.maxGridSize[2]) {
        printf("Error: Grid dimensions exceed device capabilities!\n");
        return -1;
    }
    //printf("Grid Size: (%u, %u, %u)\n", dimGrid.x, dimGrid.y, dimGrid.z);
    //printf("Block Size: (%u, %u, %u)\n", dimBlock.x, dimBlock.y, dimBlock.z);



    size_r2 = 3 * n_grid2 * sizeof(double);
    size_wr2 = n_grid2 * sizeof(double);
    size_rn = 3 * n_nuc * sizeof(double);

    size_aos_r2 = 4 * n_grid2 * n_ao * sizeof(double);

    size_jbh_d = size_bh * n_nuc * sizeof(double);
    size_jbh_i = size_bh * n_nuc * sizeof(int);

    size_int2 = 4 * n_grid1 * n_ao * n_ao * sizeof(double);
    size_int2_send = 3 * n_grid1 * n_ao * n_ao * sizeof(double);




    int i_dev, i_peer;
    int n_grid1_loc0, n_grid1_loc1, n_grid1_rem;
    int ind_dev[num_devices];
    int n_grid1_dev[num_devices];

    double * d_r2[num_devices];
    double * d_wr2[num_devices];
    double * d_rn[num_devices];
    double * d_aos_data2[num_devices];
    double * d_c_bh[num_devices]; 
    int * d_m_bh[num_devices];
    int * d_n_bh[num_devices];
    int * d_o_bh[num_devices];

    size_t size_r1_dev[num_devices];
    size_t size_int2_dev[num_devices];
    double * d_r1_dev[num_devices];
    double * d_int2_grad1_u12_ao[num_devices];

    n_grid1_rem = n_grid1 % num_devices;
    n_grid1_loc1 = (n_grid1 - n_grid1_rem) / num_devices;
    n_grid1_loc0 = n_grid1_loc1 + n_grid1_rem;


    ind_dev[0] = 0;
    n_grid1_dev[0] = n_grid1_loc0;
    size_r1_dev[0] = n_grid1_loc0 * sizeof(double);
    size_int2_dev[0] = 4 * n_grid1_loc0 * n_ao * n_ao * sizeof(double);
    for (dev = 1; dev < num_devices; dev++) {
        ind_dev[dev] = ind_dev[dev-1] + n_grid1_dev[dev-1];
        n_grid1_dev[dev] = n_grid1_loc1;
        size_r1_dev[dev] = n_grid1_loc1 * sizeof(double);
        size_int2_dev[dev] = 4 * n_grid1_loc1 * n_ao * n_ao * sizeof(double);
    }

    // send data to GPUs
    for (dev = 0; dev < num_devices; dev++) {

        checkCudaErrors(cudaSetDevice(dev), "cudaSetDevice", __FILE__, __LINE__);

        i_dev = ind_dev[dev];

        checkCudaErrors(cudaMalloc((void**)&d_r1_dev[dev], size_r1_dev[dev]), "cudaMalloc", __FILE__, __LINE__);
        checkCudaErrors(cudaMalloc((void**)&d_r2[dev], size_r2), "cudaMalloc", __FILE__, __LINE__);
        checkCudaErrors(cudaMalloc((void**)&d_wr2[dev], size_wr2), "cudaMalloc", __FILE__, __LINE__);
        checkCudaErrors(cudaMalloc((void**)&d_rn[dev], size_rn), "cudaMalloc", __FILE__, __LINE__);
        checkCudaErrors(cudaMalloc((void**)&d_aos_data2[dev], size_aos_r2), "cudaMalloc", __FILE__, __LINE__);
        checkCudaErrors(cudaMalloc((void**)&d_c_bh[dev], size_jbh_d), "cudaMalloc", __FILE__, __LINE__);
        checkCudaErrors(cudaMalloc((void**)&d_m_bh[dev], size_jbh_i), "cudaMalloc", __FILE__, __LINE__);
        checkCudaErrors(cudaMalloc((void**)&d_n_bh[dev], size_jbh_i), "cudaMalloc", __FILE__, __LINE__);
        checkCudaErrors(cudaMalloc((void**)&d_o_bh[dev], size_jbh_i), "cudaMalloc", __FILE__, __LINE__);

        // allocate full size for the the full 2e-integrals
        checkCudaErrors(cudaMalloc((void**)&d_int2_grad1_u12_ao[dev], size_int2), "cudaMalloc", __FILE__, __LINE__);

        checkCudaErrors(cudaMemcpy(d_r1_dev[dev], &h_r1[i_dev], size_r1_dev[dev], cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
        checkCudaErrors(cudaMemcpy(d_r2[dev], h_r2, size_r2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
        checkCudaErrors(cudaMemcpy(d_wr2[dev], h_wr2, size_wr2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
        checkCudaErrors(cudaMemcpy(d_rn[dev], h_rn, size_rn, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
        checkCudaErrors(cudaMemcpy(d_aos_data2[dev], h_aos_data2, size_aos_r2, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
        checkCudaErrors(cudaMemcpy(d_c_bh[dev], h_c_bh, size_jbh_d, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
        checkCudaErrors(cudaMemcpy(d_m_bh[dev], h_m_bh, size_jbh_i, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
        checkCudaErrors(cudaMemcpy(d_n_bh[dev], h_n_bh, size_jbh_i, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
        checkCudaErrors(cudaMemcpy(d_o_bh[dev], h_o_bh, size_jbh_i, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);

        //// Enable peer access
        //for (peer = 0; peer < num_devices; peer++) {
        //    if (peer != dev) {
        //        checkCudaErrors(cudaDeviceEnablePeerAccess(peer, 0), "cudaDeviceEnablePeerAccess", __FILE__, __LINE__);
        //    }
        //}

    }

    // lunch kernels
    for (dev = 0; dev < num_devices; dev++) {

        checkCudaErrors(cudaSetDevice(dev), "cudaSetDevice", __FILE__, __LINE__);

        i_dev = ind_dev[dev];

        printf(" dev = %d, i_dev = %d \n", dev, i_dev);

        get_int2_grad1_u12_ao(dimGrid, dimBlock, i_dev,
                              n_grid1_dev[dev], n_grid2, n_ao, n_nuc, size_bh,
                              d_r1_dev[dev], d_r2[dev], d_wr2[dev], d_rn[dev], d_aos_data2[dev],
                              d_c_bh[dev], d_m_bh[dev], d_n_bh[dev], d_o_bh[dev],
                              d_int2_grad1_u12_ao[dev]);
    }


    // synchronize
    for (dev = 0; dev < num_devices; dev++) {
        checkCudaErrors(cudaSetDevice(dev), "cudaSetDevice", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
    }


//    // send to Host
//    // TODO openmp ?
//    for (dev = 0; dev < num_devices; dev++) {
//
//        checkCudaErrors(cudaSetDevice(dev), "cudaSetDevice", __FILE__, __LINE__);
//
//        i_dev = ind_dev[dev];
//
//        checkCudaErrors(cudaMemcpyAsync(h_int2_grad1_u12_ao[i_dev], d_int2_grad1_u12_ao[dev], 
//                                        size_int2_send_loc[dev], cudaMemcpyDeviceToHost), 
//                        "cudaMemcpyAsync", __FILE__, __LINE__);
//    }




    float * h_tmp = (float*) malloc(size_int2_dev[0]);

    for (dev = 0; dev < num_devices; dev++) {

        checkCudaErrors(cudaSetDevice(dev), "cudaSetDevice", __FILE__, __LINE__);

        checkCudaErrors(cudaFree(d_r1_dev[dev]), "cudaFree", __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_r2[dev]), "cudaFree", __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_wr2[dev]), "cudaFree", __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_rn[dev]), "cudaFree", __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_aos_data2[dev]), "cudaFree", __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_c_bh[dev]), "cudaFree", __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_m_bh[dev]), "cudaFree", __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_n_bh[dev]), "cudaFree", __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_o_bh[dev]), "cudaFree", __FILE__, __LINE__);

        // Free & Peer-to-peer memory copy
        //for (peer = 0; peer < num_devices; peer++) {
        //    if (peer != dev) {
        //        i_peer = ind_dev[peer];
        //        cudaMemcpyPeer(d_int2_grad1_u12_ao[dev]+i_peer, dev, 
        //                       d_int2_grad1_u12_ao[peer]+i_peer, peer, 
        //                       size_int2_dev[peer]);
        //    }
        //}

        // Use host as intermediary to transfer data between devices
        cudaMemcpy(&h_tmp[0], d_int2_grad1_u12_ao[dev], size_int2_dev[dev], cudaMemcpyDeviceToHost);
        for (peer = 0; peer < num_devices; peer++) {
            if (peer != dev) {
                checkCudaErrors(cudaSetDevice(peer), "cudaSetDevice", __FILE__, __LINE__);
                cudaMemcpy(d_int2_grad1_u12_ao[peer], &h_tmp[0], size_int2_dev[dev], cudaMemcpyHostToDevice);
            }
        }
    }

    free(h_tmp);










    // synchronize
    for (dev = 0; dev < num_devices; dev++) {
        checkCudaErrors(cudaSetDevice(dev), "cudaSetDevice", __FILE__, __LINE__);
        checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
    }


    // // //




    // 2-e integral
    // TODO
    // prallelize over mo_num * mo_num / num_devices


    dev = 0;
    checkCudaErrors(cudaSetDevice(dev), "cudaSetDevice", __FILE__, __LINE__);

    size_wr1 = n_grid1 * sizeof(double);
    size_aos_r1 = 4 * n_grid1 * n_ao * sizeof(double);
    size_int1 = n_ao * n_ao * n_ao * n_ao * sizeof(double);

    checkCudaErrors(cudaMalloc((void**)&d_wr1, size_wr1), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_aos_data1, size_aos_r1), "cudaMalloc", __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&d_int_2e_ao, size_int1), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_wr1, h_wr1, size_wr1, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(d_aos_data1, h_aos_data1, size_aos_r1, cudaMemcpyHostToDevice), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tHD += time_loc;
    time_tot += time_loc;


    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    get_int_2e_ao(n_grid1, n_ao, d_wr1, d_aos_data1, d_int2_grad1_u12_ao[dev], d_int_2e_ao);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    time_tot += time_loc;
    printf("Ellapsed time for get_int_2e_ao = %.3f sec\n", time_loc/1000.0f);

    checkCudaErrors(cudaFree(d_wr1), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_aos_data1), "cudaFree", __FILE__, __LINE__);

    // // //


    // transfer data to Host

    checkCudaErrors(cudaEventRecord(start_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_int_2e_ao, d_int_2e_ao, size_int1, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpy(h_int2_grad1_u12_ao, d_int2_grad1_u12_ao[0], size_int2_send, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
    checkCudaErrors(cudaEventRecord(stop_loc, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_loc), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_loc, stop_loc), "cudaEventElapsedTime", __FILE__, __LINE__);
    tHD += time_loc;
    time_tot += time_loc;

    checkCudaErrors(cudaFree(d_int_2e_ao), "cudaFree", __FILE__, __LINE__);

    for (dev = 0; dev < num_devices; dev++) {
        checkCudaErrors(cudaSetDevice(dev), "cudaSetDevice", __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_int2_grad1_u12_ao[dev]), "cudaFree", __FILE__, __LINE__);
    }

    // // //


    dev = 0;
    checkCudaErrors(cudaSetDevice(dev), "cudaSetDevice", __FILE__, __LINE__);

    checkCudaErrors(cudaEventDestroy(start_loc), "cudaEventDestroy", __FILE__, __LINE__);
    checkCudaErrors(cudaEventDestroy(stop_loc), "cudaEventDestroy", __FILE__, __LINE__);


    printf("Ellapsed time for Device <-> Host transf = %.3f sec\n", tHD/1000.0f);
    //printf("Ellapsed (effective) time on GPU for cutc_int = %.3f sec\n", time_tot/1000.0f);

    checkCudaErrors(cudaEventRecord(stop_tot, 0), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop_tot), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&time_loc, start_tot, stop_tot), "cudaEventElapsedTime", __FILE__, __LINE__);
    printf("Ellapsed (total) time on GPU for cutc_int = %.3f sec\n", time_loc/1000.0f);

    //time_req = clock() - time_req;
    //printf("Ellapsed time (sec) : %f sec\n", (float)time_req / CLOCKS_PER_SEC);



    return 0;
}


