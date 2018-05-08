#include "main.h"

__global__ void forward_cuda() {

}
extern "C" int full_forward_device(int *host_a, int *host_b, int *host_c ) {
  printf("test of cuda forward function.");
  return 0;
}
