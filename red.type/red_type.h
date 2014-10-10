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

typedef enum frame_center
		{
			FRAME_CENTER_BARY,
			FRAME_CENTER_ASTRO
		} frame_center_t;

typedef enum threshold
		{
			THRESHOLD_HIT_CENTRUM_DISTANCE,
			THRESHOLD_EJECTION_DISTANCE,
			THRESHOLD_COLLISION_FACTOR,
			THRESHOLD_N
		} threshold_t;

typedef enum integrator_type
		{ 
			INTEGRATOR_EULER,
			INTEGRATOR_RUNGEKUTTA2,
			INTEGRATOR_RUNGEKUTTA4,
			INTEGRATOR_RUNGEKUTTAFEHLBERG78,
			INTEGRATOR_RUNGEKUTTANYSTROM,
		} integrator_type_t;

typedef enum event_name
		{
			EVENT_NAME_NONE,
			EVENT_NAME_HIT_CENTRUM,
			EVENT_NAME_EJECTION,
			EVENT_NAME_CLOSE_ENCOUNTER,
			EVENT_NAME_COLLISION,
			EVENT_NAME_N
		} event_name_t;

typedef enum migration_type
		{
			MIGRATION_TYPE_NO,
			MIGRATION_TYPE_TYPE_I,
			MIGRATION_TYPE_TYPE_II
		} migration_type_t;

typedef enum body_type
		{
			BODY_TYPE_STAR,
			BODY_TYPE_GIANTPLANET,
			BODY_TYPE_ROCKYPLANET,
			BODY_TYPE_PROTOPLANET,
			BODY_TYPE_SUPERPLANETESIMAL,
			BODY_TYPE_PLANETESIMAL,
			BODY_TYPE_TESTPARTICLE,
			BODY_TYPE_N
		} body_type_t;

// int4_t gets aligned to 16 bytes.
typedef struct /*__builtin_align__(16)*/ _int4
		{
			int_t x;
			int_t y;
			int_t z;
			int_t w;
		} int4_t;

// var2_t gets aligned to 16 bytes.
typedef struct /*__builtin_align__(16)*/ _var2
		{
			var_t x;
			var_t y;
		} var2_t;

// vec_t gets aligned to 16 bytes.
typedef struct /*__builtin_align__(16)*/ vec
		{
			var_t x;
			var_t y;
			var_t z;
			var_t w;
		} vec_t;

// param_t gets aligned to 16 bytes.
typedef struct /*__builtin_align__(16)*/ param
		{
			var_t mass;
			var_t radius;
			var_t density;
			var_t cd;
		} param_t;

// body_metadata_t gets aligned to 16 bytes.
typedef struct /*__builtin_align__(16)*/ body_metadata
		{
			int32_t id;
			int32_t body_type;
			int32_t mig_type;
			var_t	mig_stop_at;
		} body_metadata_t;

typedef struct sim_data
		{
			vector<vec_t*>	y;				//!< Host vectors of initial position and velocity of the bodies on the host
			vector<vec_t*>	d_y;			//!< Device vectors of ODE variables at the beginning of the step (at time t)
			vector<vec_t*>	d_yout;			//!< Device vectors of ODE variables at the end of the step (at time tout)
			param_t			*p;				//!< Host vector of body parameters
			param_t			*d_p;			//!< Device vector of body parameters
			body_metadata_t *body_md; 		//!< Host vector of additional body parameters
			body_metadata_t *d_body_md; 	//!< Device vector of additional body parameters
			ttt_t			*epoch;			//!< Host vector of epoch of the bodies
			ttt_t			*d_epoch;		//!< Device vector of epoch of the bodies
		} sim_data_t;

typedef struct event_data
		{
			event_name_t	event_name;	//!< Name of the event
			ttt_t	t;			//!< Time of the event
			int2_t	id;			//!< ids of the bodies
			int2_t	idx;		//!< indicies of the bodies
			var_t	d;			//!< distance of the bodies
			vec_t	r1;			//!< Position of body 1
			vec_t	v1;			//!< Velocity of body 1
			vec_t	r2;			//!< Position of body 2
			vec_t	v2;			//!< Velocity of body 2
		} event_data_t;

struct interaction_bound
{
	int2_t	sink;
	int2_t	source;

	interaction_bound(int2_t sink, int2_t source) : 
		sink(sink),
		source(source) 
	{ }

	interaction_bound(int x0, int y0, int x1, int y1)
	{
		sink.x = x0;
		sink.y = y0;
		source.x = x1;
		source.y = y1;
	}
};
