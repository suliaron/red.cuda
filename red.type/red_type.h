#pragma once

// includes system
#include <cstdint>

// include CUDA
#include "cuda_runtime.h"

// General settings for the integrator

//! Type of time variables
typedef double		ttt_t;
//! Type of variables
typedef double		var_t;
//! Type of tuple
typedef double2		var2_t;
//! Type of vectors
//typedef double4		vec_t;
//! Type of boolean variables
typedef bool		bool_t;
//! Type of integer variables
typedef int			int_t;
//! Type of integer tuples variables
typedef int2		int2_t;

struct __device_builtin__ __builtin_align__(16) posm
{
    double x, y, z, m;
};

struct __device_builtin__ __builtin_align__(16) velR
{
    double x, y, z, R;
};

struct __device_builtin__ __builtin_align__(16) int4
{
	int32_t id;
	int32_t active;
	int32_t body_type;
	int32_t mig_type;
};


typedef __device_builtin__ struct posm posm_t;
typedef __device_builtin__ struct velR velR_t;
typedef __device_builtin__ struct int4 body_metadata_t;
