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
//! Type of boolean variables
typedef bool		bool_t;
//! Type of integer variables
typedef int			int_t;
//! Type of integer tuples variables
typedef int2		int2_t;

// int4_t gets aligned to 16 bytes.
typedef struct __builtin_align__(16) _int4
{
    int_t x, y, z, w;
} int4_t;

typedef struct __builtin_align__(16) _var2
{
    var_t x, y;
} var2_t;

// vec_t gets aligned to 16 bytes.
typedef struct __builtin_align__(16) vec
{
    var_t x, y, z, w;
} vec_t;

// posm_t gets aligned to 16 bytes.
typedef struct __builtin_align__(16) posm
{
    var_t x, y, z, m;
} posm_t;

// velR_t gets aligned to 16 bytes.
typedef struct __builtin_align__(16) velR
{
    var_t x, y, z, R;
} velR_t;

// param_t gets aligned to 16 bytes.
typedef struct __builtin_align__(16) param
{
	var_t mass;
    var_t density;
	var_t cd;
	var_t mig_stop_at;
} param_t;

typedef struct __builtin_align__(16) body_metadata
{
	int32_t id;
	int32_t active;
	int32_t body_type;
	int32_t mig_type;
} body_metadata_t;

typedef struct sim_data
{
	posm_t			*pos, *d_pos, *d_pos_out;
	velR_t			*vel, *d_vel, *d_vel_out;
	param_t			*params, *d_params;
	body_metadata_t *body_md, *d_body_md;
	ttt_t			*epoch, *d_epoch;
} sim_data_t;

struct	interaction_bound {
	int2_t	sink;
	int2_t	source;

	interaction_bound(int2_t sink, int2_t source) : 
		sink(sink),
		source(source) 
	{ }
};

