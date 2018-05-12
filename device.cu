#include "main.h"

__global__ void pool_forward(double *in, double *out, size_t size_out) {
  int o_id = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
  int i_id = threadIdx.x*2 + threadIdx.y*2*blockDim.x*2 + blockIdx.x*blockDim.x*2*blockDim.y*2;

  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) {
      if (out[o_id] < in[i_id+i*2*size_out+j]) {
        out[o_id] = in[i_id+i*2*size_out+j];
      }
    }
  }
}

void pool_forward_device(double* in, double* out, size_t size_out, size_t img_d) {
  double *d_in, *d_out;
  cudaMalloc((double**)&d_in, sizeof(double)*size_out*2*size_out*2*img_d);
  cudaMalloc((double**)&d_out, sizeof(double)*size_out*size_out*img_d);
  cudaMemcpy(d_in, in, sizeof(double)*size_out*2*size_out*2*img_d, cudaMemcpyHostToDevice);

  dim3 block_size(size_out, size_out, 1);
  dim3 grid_size(img_d, 1, 1);
  pool_forward<<<grid_size, block_size>>>(d_in, d_out, size_out);

  cudaMemcpy(out, d_out, sizeof(double)*size_out*size_out*img_d, cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
}


__global__ void conv_forward(double* in, double* filter, double* bias, double* out) {
  int i_id = (threadIdx.x+2)+(threadIdx.y+2)*(blockDim.x+4)+blockIdx.x*(blockDim.x+4)*(blockDim.y+4);
  int o_id = threadIdx.x+threadIdx.y*blockDim.x+blockIdx.y*blockDim.x*blockDim.y;
  int f_id = 12+blockIdx.x*25+blockIdx.y*25*gridDim.x;
  for (int i = -2; i < 3; i++) {
    for (int j = -2; j < 3; j++) {
      out[o_id] += in[i_id+i*(blockDim.x+4)+j]*filter[f_id+i*5+j];
    }
  }
  out[o_id] += bias[blockIdx.y];
  if (out[o_id] < 0) out[o_id] = 0.0;
}


void conv_forward_device(double* in, double* filter, double* bias, double* out, size_t size, size_t img_d, size_t fil_d) {
  double *d_i, *d_f, *d_b, *d_o;
  cudaMalloc((double**)&d_i, sizeof(double)*(size+4)*(size+4)*img_d);
  cudaMalloc((double**)&d_f, sizeof(double)*5*5*img_d*fil_d);
  cudaMalloc((double**)&d_b, sizeof(double)*fil_d);
  cudaMalloc((double**)&d_o, sizeof(double)*size*size*fil_d);
  cudaMemcpy(d_i, in, sizeof(double)*(size+4)*(size+4)*img_d, cudaMemcpyHostToDevice);
  cudaMemcpy(d_f, filter, sizeof(double)*5*5*img_d*fil_d, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, bias, sizeof(double)*fil_d, cudaMemcpyHostToDevice);
  dim3 block_size(size,size,1);
  dim3 grid_size(img_d,fil_d,1);
  conv_forward<<<grid_size, block_size>>>(d_i, d_f, d_b, d_o);
  cudaMemcpy(out, d_o, sizeof(double)*size*size*fil_d, cudaMemcpyDeviceToHost);
  cudaFree(d_i);
  cudaFree(d_f);
  cudaFree(d_b);
  cudaFree(d_o);
}


__global__ void full_forward() {

}

void full_forward_device(double * w, double * i, double * o) {

}
