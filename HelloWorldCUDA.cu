// nvcc -o HelloWorldCUDA.dll --shared HelloWorldCUDA.cu
#include "cuda_runtime.h"
#include <string>
#define DllExport __declspec(dllexport)

extern "C" 
{
	DllExport char* CUDA_device_name();
	char* CUDA_device_name()
	{
		cudaDeviceProp device;
		cudaGetDeviceProperties(&device, 0);
		char* label = new char[256];
		strcpy(label,device.name);
		return label ;
	}
}