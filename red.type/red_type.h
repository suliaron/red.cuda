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
    var_t x, y, z, m;
};

struct __device_builtin__ __builtin_align__(16) velR
{
    var_t x, y, z, R;
};

struct __device_builtin__ __builtin_align__(16) param
{
	var_t mass;
    var_t density;
	var_t cd;
	var_t mig_stop_at;
};

struct __device_builtin__ __builtin_align__(16) body_metadata
{
	int32_t id;
	int32_t active;
	int32_t body_type;
	int32_t mig_type;
};

typedef __device_builtin__ struct posm			posm_t;
typedef __device_builtin__ struct velR			velR_t;
typedef __device_builtin__ struct param			param_t;
typedef __device_builtin__ struct body_metadata	body_metadata_t;

typedef struct sim_data
{
	posm_t*				pos;
	velR_t*				vel;
	param_t*			params;
	body_metadata_t*	body_md;
} sim_data_t;

struct	interaction_bound {
	int2_t	sink;
	int2_t	source;

	interaction_bound(int2_t sink, int2_t source) : 
		sink(sink),
		source(source) 
	{ }
};
