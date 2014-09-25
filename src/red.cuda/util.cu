#include "util.h"
#include "nbody_exception.h"
#include "red_type.h"
#include "red_macro.h"

void allocate_device_vector(void **d_ptr, size_t size, const char *file, int line)
{
	cudaError_t cudaStatus = cudaSuccess;
	cudaMalloc(d_ptr, size);
	cudaStatus = HandleError(cudaStatus, file, line);
	if (cudaSuccess != cudaStatus)
	{
		throw nbody_exception("cudaMalloc failed", cudaStatus);
	}
}

void copy_vector_to_device(void* dst, const void *src, size_t count)
{
	cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw nbody_exception("cudaMemcpy failed (copy_vector_to_device)", cudaStatus);
	}
}

void copy_vector_to_host(void* dst, const void *src, size_t count)
{
	cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw nbody_exception("cudaMemcpy failed (copy_vector_to_host)", cudaStatus);
	}
}

void copy_constant_to_device(const void* dst, const void *src, size_t count)
{
	cudaMemcpyToSymbol(dst, src, count);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw nbody_exception("cudaMemcpyToSymbol failed (copy_constant_to_device)", cudaStatus);
	}
}
