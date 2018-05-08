#include "main.h"

__global__ void forward_cuda(int* x, int * y, int * z) {

}
extern "C" int full_forward_device(int *host_a, int *host_b, int *host_c ) {
  printf("test of cuda forward function.");
  return 0;
}
