#include "main.h"

__global__ void pool_forward(double* in, double* out) {
  int out_id = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
  int in_id = threadIdx.x*2 + (threadIdx.y*2)*(blockDim.x*2) + blockIdx.x*(blockDim.x*blockDim.y)*4;
  out[out_id] = in[in_id];
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      if (out[out_id] < in[in_id+i+j*blockDim.x*2]) {
        out[out_id] = in[in_id+i+j*blockDim.x*2];
      }
    }
  }
}

void pool_forward_device(double* in, double* out) {
  dim3 block_size(14,14,1);
  dim3 grid_size(32,1,1);
  double *d_in, *d_out;
  cudaMalloc((double**)&d_in, sizeof(double)*28*28*32);
  cudaMalloc((double**)&d_out, sizeof(double)*14*14*32);
  cudaMemcpy(d_in, in, sizeof(double)*28*28*32, cudaMemcpyHostToDevice);

  pool_forward<<<grid_size, block_size>>>(d_in, d_out);

  cudaMemcpy(out, d_out, sizeof(double)*14*14*32, cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
}

void pool_forward_device(double* in, double* out, int edge) {
  dim3 block_size(edge,edge,1);
  dim3 grid_size(32,1,1);
  double *d_in, *d_out;
  cudaMalloc((double**)&d_in, sizeof(double)*(edge*2)*(edge*2)*32);
  cudaMalloc((double**)&d_out, sizeof(double)*edge*edge*32);
  cudaMemcpy(d_in, in, sizeof(double)*(edge*2)*(edge*2)*32, cudaMemcpyHostToDevice);

  pool_forward<<<grid_size, block_size>>>(d_in, d_out);

  cudaMemcpy(out, d_out, sizeof(double)*edge*edge*32, cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
}

// __global__ void add(int *x, int *y, int *z) {
//   z[threadIdx.x] = y[threadIdx.x] + x[threadIdx.x];
//   printf("Hello %d\n", threadIdx.x);
// }
//
// void test_device (int* x, int* y, int* z) {
//   int* d_x, *d_y, *d_z;
//   cudaMalloc((int**)&d_x, sizeof(int)*16);
//   cudaMalloc((int**)&d_y, sizeof(int)*16);
//   cudaMalloc((int**)&d_z, sizeof(int)*16);
//   cudaMemcpy(d_x, x, sizeof(int)*16, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_y, y, sizeof(int)*16, cudaMemcpyHostToDevice);
//
//   add<<<1,16>>> (d_x, d_y, d_z);
//
//   cudaMemcpy(z, d_z, sizeof(int)*16, cudaMemcpyDeviceToHost);
//   cudaFree(d_x);
//   cudaFree(d_y);
//   cudaFree(d_z);
// }
