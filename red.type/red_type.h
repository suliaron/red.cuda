#pragma once

// includes system
#include <cstdint>
#include <vector>

// include CUDA
#include "cuda_runtime.h"

using namespace std;

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

// var2_t gets aligned to 16 bytes.
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
//typedef struct __builtin_align__(16) posm
//{
//    var_t x, y, z, m;
//} posm_t;

// velR_t gets aligned to 16 bytes.
//typedef struct __builtin_align__(16) velR
//{
//    var_t x, y, z, R;
//} velR_t;

// param_t gets aligned to 16 bytes.
typedef struct __builtin_align__(16) param
{
	var_t mass;
    var_t radius;
    var_t density;
	var_t cd;
} param_t;

typedef struct __builtin_align__(16) body_metadata
{
	int32_t id;
//	int32_t active;
	int32_t body_type;
	int32_t mig_type;
	var_t	mig_stop_at;
} body_metadata_t;

typedef struct sim_data
{
	vector<vec_t*>	y;							//!< Host vectors of initial position and velocity of the bodies on the host
	vector<vec_t*>	d_y;						//!< Device vectors of ODE variables at the beginning of the step (at time t)
	vector<vec_t*>	d_yout;						//!< Device vectors of ODE variables at the end of the step (at time tout)
	param_t			*params;					//!< Host vector of body parameters
	param_t			*d_params;					//!< Device vector of body parameters
	body_metadata_t *body_md; 					//!< Host vector of additional body parameters
	body_metadata_t *d_body_md; 				//!< Device vector of additional body parameters
	ttt_t			*epoch;						//!< Host vector of epoch of the bodies
	ttt_t			*d_epoch;					//!< Device vector of epoch of the bodies
} sim_data_t;

struct	interaction_bound {
	int2_t	sink;
	int2_t	source;

	interaction_bound(int2_t sink, int2_t source) : 
		sink(sink),
		source(source) 
	{ }
};
