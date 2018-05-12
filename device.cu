#include "main.h"

// __global__ void pool_forward(double* in, double* out) {
//   int out_id = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
//   int in_id = threadIdx.x*2 + (threadIdx.y*2)*(blockDim.x*2) + blockIdx.x*(blockDim.x*blockDim.y)*4;
//   out[out_id] = in[in_id];
//   for (int i = 0; i < 2; ++i) {
//     for (int j = 0; j < 2; ++j) {
//       if (out[out_id] < in[in_id+i+j*blockDim.x*2]) {
//         out[out_id] = in[in_id+i+j*blockDim.x*2];
//       }
//     }
//   }
// }

// void pool_forward_device_first(double* in, double* out) {
//   dim3 block_size(14,14,1);
//   dim3 grid_size(32,1,1);
//   double *d_in, *d_out;
//   cudaMalloc((double**)&d_in, sizeof(double)*28*28*32);
//   cudaMalloc((double**)&d_out, sizeof(double)*14*14*32);
//   cudaMemcpy(d_in, in, sizeof(double)*28*28*32, cudaMemcpyHostToDevice);
//
//   pool_forward<<<grid_size, block_size>>>(d_in, d_out);
//
//   cudaMemcpy(out, d_out, sizeof(double)*14*14*32, cudaMemcpyDeviceToHost);
//   cudaFree(d_in);
//   cudaFree(d_out);
// }
//
// void pool_forward_device_second(double* in, double* out) {
//   dim3 block_size(7,7,1);
//   dim3 grid_size(64,1,1);
//   double *d_in, *d_out;
//   cudaMalloc((double**)&d_in, sizeof(double)*14*14*64);
//   cudaMalloc((double**)&d_out, sizeof(double)*7*7*64);
//   cudaMemcpy(d_in, in, sizeof(double)*14*14*64, cudaMemcpyHostToDevice);
//
//   pool_forward<<<grid_size, block_size>>>(d_in, d_out);
//
//   cudaMemcpy(out, d_out, sizeof(double)*7*7*64, cudaMemcpyDeviceToHost);
//   cudaFree(d_in);
//   cudaFree(d_out);
// }

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



// __global__ void conv_forward(double* in, double* filter, double* bias, double* out) {
//   int t_id = threadIdx.x + threadIdx.y*blockDim.x + blockDim.x*blockDim.y*blockIdx.x;
//   int i_id = threadIdx.x+2 + threadIdx.y*(blockDim.x+4) + (blockDim.x+4)*(blockDim.y+4)*blockIdx.x;
//   double res = 0;
//   for (int i = -2; i <= 2; ++i) {
//     for (int j = -2; j <= 2; ++j) {
//       res += in[i_id+i*32+j]*filter[blockIdx.x*25+i*5+j];
//     }
//   }
//   out[t_id] = res + bias[blockIdx.x];
//   // printf("tid: %d\n", t_id);
// }
//
// void conv_forward_device_first(double* in, double* filter, double* bias, double* out) {
//   double *d_i, *d_f, *d_b, *d_o;
//   cudaMalloc((double**)&d_i, sizeof(double)*32*32*1);
//   cudaMalloc((double**)&d_f, sizeof(double)*5*5*32);
//   cudaMalloc((double**)&d_b, sizeof(double)*32);
//   cudaMalloc((double**)&d_o, sizeof(double)*28*28*32);
//   cudaMemcpy(d_i, in, sizeof(double)*32*32*1, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_f, filter, sizeof(double)*5*5*32, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_b, bias, sizeof(double)*32, cudaMemcpyHostToDevice);
//
//   dim3 block_size(28,28,1);
//   dim3 grid_size(32,1,1);
//   conv_forward<<<grid_size, block_size>>>(d_i, d_f, d_b, d_o);
//
//   cudaMemcpy(out, d_o, sizeof(double)*28*28*32, cudaMemcpyDeviceToHost);
//   cudaFree(d_i);
//   cudaFree(d_f);
//   cudaFree(d_b);
//   cudaFree(d_o);
// }

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
  int i_id = x_in + y_in + z_in;
  for (int i = -2; i <= 2; ++i) {
    for (int j = -2; j <= 2; ++j) {
      out[o_id] += filter[blockIdx.y*25*gridDim.x+(i+2)*5+j+2] * in[i_id+i*(blockDim.x+4)+j+2];
    }
  }
  // out[o_id] += bias[blockIdx.y];
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


// __global__ void conv_forward_test(double* in, double* filter, double* bias, double* out) {
//   int t_id = threadIdx.x + threadIdx.y*blockDim.x + blockDim.x*blockDim.y*blockIdx.x;
//   int i_id = threadIdx.x+1 + threadIdx.y*(blockDim.x+2) + (blockDim.x+2)*(blockDim.y+2)*blockIdx.x;
//   double res = 0;
//   for (int i = -1; i <= 1; ++i) {
//     for (int j = -1; j <= 1; ++j) {
//       res += in[i_id+i*5+j]*filter[blockIdx.x*9+i*3+j];
//     }
//   }
//   out[t_id] = res + bias[blockIdx.x];
//   // printf("tid: %d\n", t_id);
// }
//
// void conv_forward_device_test(double* in, double* filter, double* bias, double* out) {
//   double *d_i, *d_f, *d_b, *d_o;
//   cudaMalloc((double**)&d_i, sizeof(double)*5*5*1);
//   cudaMalloc((double**)&d_f, sizeof(double)*3*3*32);
//   cudaMalloc((double**)&d_b, sizeof(double)*32);
//   cudaMalloc((double**)&d_o, sizeof(double)*3*3*32);
//   cudaMemcpy(d_i, in, sizeof(double)*5*5*1, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_f, filter, sizeof(double)*3*3*32, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_b, bias, sizeof(double)*32, cudaMemcpyHostToDevice);
//
//   dim3 block_size(3,3,1);
//   dim3 grid_size(32,1,1);
//   conv_forward_test<<<grid_size, block_size>>>(d_i, d_f, d_b, d_o);
//
//   cudaMemcpy(out, d_o, sizeof(double)*3*3*32, cudaMemcpyDeviceToHost);
//   cudaFree(d_i);
//   cudaFree(d_f);
//   cudaFree(d_b);
//   cudaFree(d_o);
// }


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
