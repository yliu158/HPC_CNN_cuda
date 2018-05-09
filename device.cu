#include "main.h"

__global__ void full_forward(int* weight, int* input, int* output) {

}

template <typename IN_DIMS, size_t N_NEURONS>
void
FullyConnectedLayer<IN_DIMS, N_NEURONS>::device_forward(const Input &input, const Array<Input, N_NEURONS> &weight, const Array<double, N_NEURONS> &bias, const Array<double, N_NEURONS> &dropped, Output &output) {
  printf("test of cuda forward function.");
  int *w, *i, *o;
  w = (int*)malloc(sizeof(int)); 
  i = (int*)malloc(sizeof(int));
  o = (int*)malloc(sizeof(int));
  int *d_w, *d_i, *d_o;
  cudaMalloc((int**)&d_w, sizeof(int));
  cudaMalloc((int**)&d_i, sizeof(int));
  cudaMalloc((int**)&d_o, sizeof(int));
  cudaMemcpy(d_w, w, sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_i, i, sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_o, o, sizeof(int),cudaMemcpyHostToDevice);

  free(w);
  free(i);
  free(o);
}
// __global__ void conv_forward() {
//
// }
//
// __global__ void padding(float* origin, float* padded) {
//   // int blockId = blockIdx.x + blockDim.x*blockIdx.y;
//   // int threadId = threadIdx.x + threadDim.x*threadIdx.y + blockId*threadDim.x*threadDim.y;
//   int threadId = threadIdx.x + threadDim.x * threadidx.y;
//   if (threadIdx.x < 2 || threadIdx.x > 29 || threadIdx.y < 2 || threadIdx.y > 29) {
//     padded[threadId] = 0.0;
//     return;
//   }
//   int ori_id = (threadIdx.x-2) + (threadDim.x-2)*(threadIdx.y-2);
//   padded[threadId] = origin[ori_id];
// }
//
// template <typename IN_DIMS, size_t N_FILTERS>
// void
// conv_forward_device() {
//     // dim3 grids(2,2,1);
//     // dim3 blocks(16, 16, 1);
//     dim3 grids(1,1,1);
//     dim3 blocks(32, 32, 1);
//     float *in, *out;
//     float *d_in, *d_out;
//     in = (float*)malloc(sizeof(float)*28*28);
//     out = (float*)malloc(sizeof(float)*32*32);
//     cudaMalloc((float**)&d_in, sizeof(float)*28*28);
//     cudaMalloc((float**)&d_out, sizeof(float)*32*32);
//     cudaMemcpy(d_in, in, sizeof(float)*28*28, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_out, out, sizeof(float)*32*32, cudaMemcpyHostToDevice);
//
//     padding<<<grids,blocks>>>(in, out);
//
//     __syncthreads();
//
//     cudaMemcpy(d_out, out, sizeof(float)*32*32, cudaMemcpyDeviceToHost);
//     free(in); free(out);
//     cudaFree(d_in); cudaFree(d_out);
//     // __syncthreads();
// }
