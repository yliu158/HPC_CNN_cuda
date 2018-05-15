#include "main.h"

__global__ void pool_forward(double *in, double *out) {
  int o_id = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
  int i_id = threadIdx.x*2 + threadIdx.y*2*blockDim.x*2 + blockIdx.x*blockDim.x*2*blockDim.y*2;
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) {
      if (out[o_id] < in[i_id+i*2*blockDim.x+j]) {
        out[o_id] = in[i_id+i*2*blockDim.x+j];
      }
    }
  }
}

void pool_forward_device(double* in, double* out, size_t size_out, size_t img_d) {
  double *d_in, *d_out;
  cudaMalloc((double**)&d_out, sizeof(double)*size_out*size_out*img_d);
  cudaMalloc((double**)&d_in, sizeof(double)*size_out*2*size_out*2*img_d);
  cudaMemcpy(d_in, in, sizeof(double)*size_out*2*size_out*2*img_d, cudaMemcpyHostToDevice);

  dim3 block_size(size_out, size_out, 1);
  dim3 grid_size(img_d, 1, 1);
  pool_forward<<<grid_size, block_size>>>(d_in, d_out);

  cudaMemcpy(out, d_out, sizeof(double)*size_out*size_out*img_d, cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
}

__global__ void pool_backprop(double *down_deriv, double *up_deriv, int *max_i , int *max_j) {
  int id = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
  int down_id = max_j[id] + max_i[id]*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
  down_deriv[down_id] = up_deriv[id];
}

void pool_backprop_device(double *down_deriv, double *up_deriv, int *max_i, int *max_j, size_t size, size_t img_d) {
  double *d_down_deriv, *d_up_deriv;
  int *d_max_i, *d_max_j;
  cudaMalloc((double**)&d_down_deriv, sizeof(double)*size*size*img_d);
  cudaMalloc((double**)&d_up_deriv, sizeof(double)*size*size*img_d);
  cudaMalloc((double**)&d_max_i, sizeof(size_t)*size*size*img_d);
  cudaMalloc((double**)&d_max_j, sizeof(size_t)*size*size*img_d);
  cudaMemcpy(d_up_deriv, up_deriv,sizeof(double)*size*size*img_d, cudaMemcpyHostToDevice);
  cudaMemcpy(d_max_i, max_i, sizeof(size_t)*size*size*img_d, cudaMemcpyHostToDevice);
  cudaMemcpy(d_max_j, max_j, sizeof(size_t)*size*size*img_d, cudaMemcpyHostToDevice);
  dim3 block_size(size,size,1);
  dim3 grid_size(img_d,1,1);
  pool_backprop<<<grid_size, block_size>>>(d_down_deriv, d_up_deriv, d_max_i, d_max_j);
  cudaMemcpy(down_deriv, d_down_deriv,sizeof(double)*size*size*img_d, cudaMemcpyDeviceToHost);
  cudaFree(d_down_deriv);
  cudaFree(d_up_deriv);
  cudaFree(d_max_i);
  cudaFree(d_max_j);
}

__global__ void conv_backprop_down_deriv(double* down_deriv, double* filter, double* up_deriv) {
  size_t d_id = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y + blockDim.y*gridDim.x*blockDim.x*blockDim.y;
  size_t f_id = blockIdx.x*25 + blockIdx.y*gridDim.x*25;
  size_t u_id = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.y*(blockDim.x-4)*(blockDim.y-4);
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 5; j++) {
      if (threadIdx.x >= i && threadIdx.y >= j && threadIdx.x-i+4 < blockDim.x && threadIdx.y-j+4 < blockDim.y) {
        down_deriv[d_id] += up_deriv[u_id-(blockDim.x-4)*i-j] * filter[f_id+i*5+j];
      }
    }
  }
}

__global__ void conv_backprop_down_deriv_sum(double* d_down_deriv_tmp, double* d_down_deriv, size_t fil_d) {
  int id = threadIdx.x+2 + (threadIdx.y+2)*(blockDim.x+4) + blockIdx.x*(blockDim.x+4)*(blockDim.y+4);
  int offset = gridDim.x*(blockDim.x+4)*(blockDim.y+4);
  for (size_t i = 0; i < fil_d; i++) {
    d_down_deriv[id] += d_down_deriv_tmp[id+i*offset];
  }
}

void conv_backprop_downstream_device(double* down_deriv, double* up_deriv, double* filter, size_t size, size_t img_d, size_t fil_d) {

  double *d_down_deriv, *d_down_deriv_tmp, *d_up_deriv, *d_filter;
  cudaMalloc((double**)&d_down_deriv_tmp, sizeof(double)*(size+4)*(size+4)*img_d*fil_d);
  cudaMalloc((double**)&d_down_deriv, sizeof(double)*(size+4)*(size+4)*img_d);
  cudaMalloc((double**)&d_up_deriv, sizeof(double)*size*size*fil_d);
  cudaMalloc((double**)&d_filter, sizeof(double)*5*5*img_d*fil_d);

  cudaMemcpy(d_up_deriv, up_deriv, sizeof(double)*size*size*fil_d, cudaMemcpyHostToDevice);
  cudaMalloc((double**)&d_filter, sizeof(double)*5*5*img_d*fil_d);

  dim3 block_size_d(size+4, size+4, 1);
  dim3 grid_size_d(img_d, fil_d, 1);
  conv_backprop_down_deriv<<<grid_size_d, block_size_d>>>(d_down_deriv_tmp, d_filter, d_up_deriv);
  conv_backprop_down_deriv_sum<<<img_d, block_size_d>>>(d_down_deriv_tmp, down_deriv, fil_d);

  cudaMemcpy(down_deriv, d_down_deriv, sizeof(double)*(size+4)*(size+4)*img_d, cudaMemcpyDeviceToHost);

  cudaFree(d_down_deriv_tmp);
  cudaFree(d_down_deriv);
  cudaFree(d_up_deriv);
  cudaFree(d_filter);
}

void conv_backprop_filter_device () {

}



// void conv_backprop_device(double* input, double* output, double* down_deriv, double* up_deriv, double* filter_deriv, double* filter, double* bias_deriv, size_t size, size_t img_d, size_t fil_d) {
//   double *d_input, *d_output, *d_down_deriv, *d_down_deriv_tmp, *d_up_deriv, *d_filter_deriv, *d_filter, *d_bias_deriv;
//   cudaMalloc((double**)&d_input, sizeof(double)*(size+4)*(size+4)*img_d);
//   cudaMalloc((double**)&d_output, sizeof(double)*size*size*fil_d);
//   cudaMalloc((double**)&d_down_deriv_tmp, sizeof(double)*(size+4)*(size+4)*img_d*fil_d);
//   cudaMalloc((double**)&d_down_deriv, sizeof(double)*(size+4)*(size+4)*img_d);
//
//   cudaMalloc((double**)&d_up_deriv, sizeof(double)*size*size*fil_d);
//   cudaMalloc((double**)&d_filter_deriv, sizeof(double)*5*5*img_d*fil_d);
//   cudaMalloc((double**)&d_filter, sizeof(double)*5*5*img_d*fil_d);
//   cudaMalloc((double**)&d_bias_deriv, sizeof(double)*fil_d);
//
//   cudaMemcpy(d_input, input, sizeof(double)*(size+4)*(size+4)*img_d, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_output, output, sizeof(double)*size*size*fil_d, cudaMemcpyHostToDevice);
//
//   cudaMemcpy(d_up_deriv, up_deriv, sizeof(double)*size*size*fil_d, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_filter_deriv, filter_deriv, sizeof(double)*5*5*img_d*fil_d, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_filter, filter, sizeof(double)*5*5*img_d*fil_d, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_bias_deriv, bias_deriv, sizeof(double)*fil_d, cudaMemcpyHostToDevice);
//
//   dim3 block_size_d(size+4, size+4, 1);
//   dim3 grid_size_d(img_d, fil_d, 1);
//   conv_backprop_down_deriv<<<grid_size_d, block_size_d>>>(d_down_deriv_tmp, d_filter, d_up_deriv, d_output);
//   conv_backprop_down_deriv_sum<<<img_d, block_size_d>>>(d_down_deriv_tmp, down_deriv, fil_d);
//
//   cudaMemcpy(down_deriv, d_down_deriv, sizeof(double)*(size+4)*(size+4)*img_d, cudaMemcpyDeviceToHost);
//   cudaMemcpy(filter_deriv, d_filter_deriv, sizeof(double)*5*5*img_d*fil_d, cudaMemcpyDeviceToHost);
//
//   cudaFree(d_input);
//   cudaFree(d_output);
//   cudaFree(d_down_deriv_tmp);
//   cudaFree(d_down_deriv);
//
//   cudaFree(d_up_deriv);
//   cudaFree(d_filter_deriv);
//   cudaFree(d_filter);
//   cudaFree(d_bias_deriv);
// }

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

__global__ void full_forward(double * in, double * out, double * weight) {
  int i_id = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
  int w_id = i_id + blockIdx.y*gridDim.x*blockDim.x*blockDim.y;
  out[w_id] = in[i_id]*weight[w_id];
}

void full_forward_device(double * in, double * out, double * weight, double* bias, double* drop, size_t size, size_t img_d, size_t n_nro) {
  double *d_in, *d_out, *d_weight;
  cudaMalloc((double**)&d_in, sizeof(double)*size*size*img_d);
  cudaMalloc((double**)&d_out, sizeof(double)*size*size*img_d*n_nro);
  cudaMalloc((double**)&d_weight, sizeof(double)*size*size*img_d*n_nro);
  cudaMemcpy(d_in, in, sizeof(double)*size*size*img_d, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, weight, sizeof(double)*size*size*img_d*n_nro, cudaMemcpyHostToDevice);
  dim3 block_size(size, size, 1);
  dim3 grid_size(img_d, n_nro, 1);
  full_forward<<<grid_size,block_size>>>(d_in, d_out, d_weight);
  double* tmp = (double*)malloc(sizeof(double)*size*size*img_d*n_nro);
  cudaMemcpy(tmp, d_out, sizeof(double)*size*size*img_d*n_nro, cudaMemcpyDeviceToHost);
  double res = 0;
  for (size_t j = 0; j < n_nro; j++) {
    res = 0;
    for (size_t i = 0; i < size*size*img_d; i++) {
      res += tmp[j*size*size*img_d+i];
    }
    out[j] = res;
    out[j] += bias[j];
    if (out[j] < 0) out[j] = 0.0;
    out[j] *= drop[j];
  }
  free(tmp);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_weight);
}

__global__ void full_backprop() {

}

void full_backprop_device() {

}
