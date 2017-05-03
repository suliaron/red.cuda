#pragma once

// includes system
#include <stdio.h>
#include <string>

// include CUDA
#include "cuda_runtime.h"

#define NDIM		4		// Number of dimensions, 4 to coalesce memory copies
#define OUTPUT_ORDINAL_NUMBER_WIDTH 9

#define K			(var_t)0.01720209895
#define K2			(var_t)0.0002959122082855911025

#define	PI			(var_t)3.1415926535897932384626
#define	TWOPI		(var_t)6.2831853071795864769253
#define	TORAD		(var_t)0.0174532925199432957692
#define TODEG		(var_t)57.295779513082320876798

#define SEP			' '

#define THREADS_PER_BLOCK 256

// These macro functions must be enclosed in parentheses in order to give
// correct results in the case of a division i.e. 1/SQR(x) -> 1/((x)*(x))
#define	SQR(x)      ((x)*(x))
#define	CUBE(x)     ((x)*(x)*(x))
#define FORTH(x)    ((x)*(x)*(x)*(x))
#define FIFTH(x)    ((x)*(x)*(x)*(x)*(x))

// Define this to turn on error checking
#define CUDA_ERROR_CHECK
 
#define DBG()  printf("<<DEBUG: %s %s %4d>> ", __FILE__, __FUNCTION__, __LINE__)

#define CUDA_SAFE_CALL( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHECK_ERROR()    __cudaCheckError( __FILE__, __LINE__ )
 
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
		char buffer[500];

#if defined(_MSC_VER) && _MSC_VER < 1900 || defined(__GNUC__)
		sprintf(buffer,"cudaSafeCall() failed at %s:%i : %s", file, line, cudaGetErrorString( err ) );
#else
        snprintf(buffer, sizeof(buffer) - 1, "cudaSafeCall() failed at %s:%i : %s", file, line, cudaGetErrorString(err));
#endif
		std::string msg(buffer);
		throw msg;
    }
#endif
 
    return;
}
 
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
		char buffer[500];

#if defined(_MSC_VER) && _MSC_VER < 1900 || defined(__GNUC__)
		sprintf(buffer,"cudaCheckError() failed at %s:%i : %s", file, line, cudaGetErrorString( err ) );
#else
        snprintf(buffer, sizeof(buffer) - 1, "cudaCheckError() failed at %s:%i : %s", file, line, cudaGetErrorString( err ));
#endif
		std::string msg(buffer);
		throw msg;
    }
 
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
		char buffer[500];

#if defined(_MSC_VER) && _MSC_VER < 1900 || defined(__GNUC__)
		sprintf(buffer,"cudaCheckError() with sync failed at %s:%i : %s", file, line, cudaGetErrorString( err ) );
#else
        snprintf(buffer, sizeof(buffer) - 1, "cudaCheckError() with sync failed at %s:%i : %s", file, line, cudaGetErrorString(err));
#endif
        std::string msg(buffer);
		throw msg;
    }
#endif
 
    return;
}

/*
 * Using these error checking functions is easy:
 * 
 * CudaSafeCall( cudaMalloc( &fooPtr, fooSize ) );
 *  
 * fooKernel<<< x, y >>>(); // Kernel call
 * CudaCheckError();
 */
