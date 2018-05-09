#include "main.h"

__global__ void forward_cuda() {

}

template <typename IN_DIMS, size_t N_NEURONS>
int full_forward_device(const Input &input, const Array<Input, N_NEURONS> &weight, const Array<double, N_NEURONS> &bias,
  const Array<double, N_NEURONS> &dropped, Output &output) {
  printf("test of cuda forward function.");

  return 0;
}
