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
  // printf("Hello o_id:\n", threadIdx.x);
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

__global__ void conv_forward_all(double* in, double* filter, double* bias, double* out) {
  // gridDim.x:1  blockDim.x:3  blockDim.y:3  gridDim.y:32
  int x_out = threadIdx.x;
  int y_out = threadIdx.y*blockDim.x;
  int z_out = blockIdx.x*blockDim.x*blockDim.y;
  int w_out = blockIdx.y*gridDim.x*blockDim.x*blockDim.y;
  int o_id = x_out + y_out + z_out + w_out;
  int x_in = threadIdx.x+2;// 3
  int y_in = (threadIdx.y+2)*(blockDim.x+4); //21
  int z_in = blockIdx.x*(blockDim.x+4)*(blockDim.y+4);
  int i_id = x_in + y_in + z_in -(threadIdx.x+4)*2-2;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      out[o_id] += filter[blockIdx.y*25*gridDim.x+i*5+j] * in[i_id+i*(blockDim.x+4)+j];
      // printf("threadIdx.x%d  threadIdx.y%d  blockIdx.x%d   blockIdx.y%d\nfilter (%d,%d)  id:%d\n in_id %d\n\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, i, j, blockIdx.y*25*gridDim.x+(i+2)*5+j+2, i_id+i*(blockDim.x+4)+j);
    }
  }
  out[o_id] += bias[blockIdx.y];
  printf("%lf\n", blockIdx.y*25*gridDim.x+i*5+j);
  // printf("%lf\n", out[o_id]);
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
  conv_forward_all<<<grid_size, block_size>>>(d_i, d_f, d_b, d_o);
  cudaMemcpy(out, d_o, sizeof(double)*size*size*fil_d, cudaMemcpyDeviceToHost);
  cudaFree(d_i);
  cudaFree(d_f);
  cudaFree(d_b);
  cudaFree(d_o);
}
