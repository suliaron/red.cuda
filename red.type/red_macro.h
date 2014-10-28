#pragma once
// includes system
#include <stdio.h>
// include CUDA
#include "cuda_runtime.h"

#define NDIM		4		// Number of dimensions, 4 to coalesce memory copies

#define K			(var_t)0.01720209895
#define K2			(var_t)0.0002959122082855911025

#define	PI			(var_t)3.1415926535897932384626
#define	TWOPI		(var_t)6.2831853071795864769253
#define	TORAD		(var_t)0.0174532925199432957692
#define TODEG		(var_t)57.295779513082320876798

#define SEP			' '

#define THREADS_PER_BLOCK	256

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
