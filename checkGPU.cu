#include "cuda_runtime.h"
#include <iostream>

int main()
{
    cudaError_t cudaStatus;

    // 初获取设备数量
    int num = 0;
    cudaStatus = cudaGetDeviceCount(&num);
    std::cout << "Number of GPU: " << num << std::endl;

    // 获取GPU设备属性
    cudaDeviceProp prop;
    if (num > 0)
    {
        cudaGetDeviceProperties(&prop, 0);
        // 打印设备名称
        std::cout << "Device: " <<prop.name << std::endl;
    }

    system("pause");
    return 0;
}
