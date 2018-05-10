#include "main.h"

__global__ void pool_forward(double* in, double* out) {
  int t_id = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
  int o_id = threadIdx.x/2 + threadIdx.y/2*(blockDim.x/2) + threadIdx.z*(blockDim.y/2)*(blockDim.x/2);
  if (in[t_id] > out[o_id]) {
    out[o_id] = 8;
  } else {
    out[o_id] = 0.1111;
  }
}

void pool_device_forward(double* in, double* out) {
  dim3 block_size(28*28*32);
  double *d_in, *d_out;
  cudaMalloc((double**)&d_in, sizeof(double)*28*28*32);
  cudaMalloc((double**)&d_out, sizeof(double)*14*14*32);
  cudaMemcpy(d_in, in, sizeof(double)*28*28*32, cudaMemcpyHostToDevice);
  pool_forward<<<1, block_size>>>(d_in, d_out);
  cudaMemcpy(out, d_out, sizeof(double)*14*14*32, cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
}

// __global__ void conv_forward(double* weight, double* input, double* output) {
//
// }
//
// void conv_device_forward(double * w, double * i, double * o) {
//
// }



// __global__ void full_forward(double* weight, double* input, double* output) {
//   int threadId = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
//   int blockId = blockIdx.x;
//   output[blockId] += weight[threadId+blockId]*input[threadId];
// }
//
// void full_device_forward(double * w, double * i, double * o) {
//   printf("test of cuda forward function.\n");
//   double *d_w, *d_i, *d_o;
//   cudaMalloc((double**)&d_w, sizeof(double)*7*7*64*1024);
//   cudaMalloc((double**)&d_i, sizeof(double)*7*7*64);
//   cudaMalloc((double**)&d_o, sizeof(double)*1024);
//   cudaMemcpy(d_w, w, sizeof(double)*7*7*64*1024,cudaMemcpyHostToDevice);
//   cudaMemcpy(d_i, i, sizeof(double)*7*7*64,cudaMemcpyHostToDevice);
//   cudaMemcpy(d_o, o, sizeof(double)*1024,cudaMemcpyHostToDevice);
//   dim3 grid_size(1024,1,1);
//   dim3 block_size(7,7,64);
//   full_forward<<<grid_size, block_size>>>(d_w, d_i, d_o);
//   cudaMemcpy(o, d_o, sizeof(double)*1024,cudaMemcpyDeviceToHost);
//   cudaFree(d_w);
//   cudaFree(d_i);
//   cudaFree(d_o);
// }
// __global__ void conv_forward() {
//
// }
//
// __global__ void padding(double* origin, double* padded) {
//   // double blockId = blockIdx.x + blockDim.x*blockIdx.y;
//   // double threadId = threadIdx.x + threadDim.x*threadIdx.y + blockId*threadDim.x*threadDim.y;
//   double threadId = threadIdx.x + threadDim.x * threadidx.y;
//   if (threadIdx.x < 2 || threadIdx.x > 29 || threadIdx.y < 2 || threadIdx.y > 29) {
//     padded[threadId] = 0.0;
//     return;
//   }
//   double ori_id = (threadIdx.x-2) + (threadDim.x-2)*(threadIdx.y-2);
//   padded[threadId] = origin[ori_id];
// }
//
// template <typename IN_DIMS, size_t N_FILTERS>
// void
// conv_forward_device() {
//     // dim3 grids(2,2,1);
//     // dim3 blocks(7*7*64*1024, 7*7*64*1024, 1);
//     dim3 grids(1,1,1);
//     dim3 blocks(32, 32, 1);
//     double *in, *out;
//     double *d_in, *d_out;
//     in = (double*)malloc(sizeof(double)*28*28);
//     out = (double*)malloc(sizeof(double)*32*32);
//     cudaMalloc((double**)&d_in, sizeof(double)*28*28);
//     cudaMalloc((double**)&d_out, sizeof(double)*32*32);
//     cudaMemcpy(d_in, in, sizeof(double)*28*28, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_out, out, sizeof(double)*32*32, cudaMemcpyHostToDevice);
//
//     padding<<<grids,blocks>>>(in, out);
//
//     __syncthreads();
//
//     cudaMemcpy(d_out, out, sizeof(double)*32*32, cudaMemcpyDeviceToHost);
//     free(in); free(out);
//     cudaFree(d_in); cudaFree(d_out);
//     // __syncthreads();
// }
