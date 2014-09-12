#pragma once

// include CUDA
#include "cuda_runtime.h"

// These macro functions must be enclosed in parentheses in order to give
// correct results in the case of a division i.e. 1/SQR(x) -> 1/((x)*(x))
#define	SQR(x)		((x)*(x))
#define	CUBE(x)		((x)*(x)*(x))
#define FORTH(x)	((x)*(x)*(x)*(x))
#define FIFTH(x)	((x)*(x)*(x)*(x)*(x))

static cudaError_t HandleError(cudaError_t cudaStatus, const char *file, int line)
{
    if (cudaSuccess != cudaStatus) 
	{
        printf( "%s in %s at line %d\n", cudaGetErrorString( cudaStatus ), file, line );
        return cudaStatus;
    }
	return cudaStatus;
}
#define HANDLE_ERROR(cudaStatus) (HandleError(cudaStatus, __FILE__, __LINE__))
