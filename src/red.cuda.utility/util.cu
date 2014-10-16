#include <string>

#include "util.h"
#include "red_type.h"
#include "red_macro.h"

void allocate_device_vector(void **d_ptr, size_t size, const char *file, int line)
{
	cudaMalloc(d_ptr, size);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("cudaMalloc failed (allocate_device_vector)");
	}
}

void copy_vector_to_device(void* dst, const void *src, size_t count)
{
	cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("cudaMemcpy failed (copy_vector_to_device)");
	}
}

void copy_vector_to_host(void* dst, const void *src, size_t count)
{
	cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("cudaMemcpy failed (copy_vector_to_host)");
	}
}

void copy_constant_to_device(const void* dst, const void *src, size_t count)
{
	cudaMemcpyToSymbol(dst, src, count);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("cudaMemcpyToSymbol failed (copy_constant_to_device)");
	}
}
