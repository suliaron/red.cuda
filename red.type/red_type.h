#pragma once

// includes system
#include <stdint.h>
#include <vector>

// include CUDA
#include "cuda_runtime.h"

using namespace std;

//! Type of time variables
typedef double ttt_t;
//! Type of variables
typedef double var_t;
//! Type of boolean variables
typedef bool   bool_t;
//! Type of integer variables
typedef int    int_t;
//! Type of integer tuples variables
typedef int2   int2_t;


typedef enum computing_device
		{
			COMPUTING_DEVICE_CPU,
			COMPUTING_DEVICE_GPU,
			COMPUTING_DEVICE_N
		} computing_device_t;

typedef enum threshold
		{
			THRESHOLD_HIT_CENTRUM_DISTANCE,
			THRESHOLD_EJECTION_DISTANCE,
			THRESHOLD_RADII_ENHANCE_FACTOR,
			THRESHOLD_HIT_CENTRUM_DISTANCE_SQUARED,
			THRESHOLD_EJECTION_DISTANCE_SQUARED,
			THRESHOLD_N
		} threshold_t;

typedef enum integrator_type
		{ 
			INTEGRATOR_EULER,
			INTEGRATOR_RUNGEKUTTA2,
			INTEGRATOR_RUNGEKUTTA4,
			INTEGRATOR_RUNGEKUTTA5,
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

typedef enum event_counter_name
		{
			EVENT_COUNTER_NAME_TOTAL,
			EVENT_COUNTER_NAME_LAST_CLEAR,
			EVENT_COUNTER_NAME_LAST_STEP,
			EVENT_COUNTER_NAME_N
		} event_counter_name_t;

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
			BODY_TYPE_PADDINGPARTICLE,
			BODY_TYPE_N
		} body_type_t;

typedef struct orbelem
		{
			//! Semimajor-axis of the body
			var_t sma;
			//! Eccentricity of the body
			var_t ecc;
			//! Inclination of the body
			var_t inc;
			//! Argument of the pericenter
			var_t peri;
			//! Longitude of the ascending node
			var_t node;
			//! Mean anomaly
			var_t mean;
		} orbelem_t;

#ifdef _WIN64
#define __BUILTIN_ALIGN__ __builtin_align__(16)
#else
#define __BUILTIN_ALIGN__
#endif

// int4_t gets aligned to 16 bytes.
typedef struct __BUILTIN_ALIGN__ _int4
		{
			int_t x;
			int_t y;
			int_t z;
			int_t w;
		} int4_t;

// var2_t gets aligned to 16 bytes.
typedef struct __BUILTIN_ALIGN__ _var2
		{
			var_t x;
			var_t y;
		} var2_t;

// vec_t gets aligned to 16 bytes.
typedef struct __BUILTIN_ALIGN__ vec
		{
			var_t x;
			var_t y;
			var_t z;
			var_t w;
		} vec_t;

// param_t gets aligned to 16 bytes.
typedef struct __BUILTIN_ALIGN__ param
		{
			var_t mass;
			var_t radius;
			var_t density;
			var_t cd;
		} param_t;

// body_metadata_t gets aligned to 16 bytes.
typedef struct __BUILTIN_ALIGN__ body_metadata
		{
			int32_t id;
			int32_t body_type;
			int32_t mig_type;
			var_t	mig_stop_at;
		} body_metadata_t;

typedef struct sim_data
		{
			vector<vec_t*>	 y;				//!< Vectors of initial position and velocity of the bodies on the host (either in the DEVICE or HOST memory)
			vector<vec_t*>	 yout;			//!< Vectors of ODE variables at the end of the step (at time tout) (either in the DEVICE or HOST memory)
			param_t*		 p;   			//!< Vector of body parameters (either in the DEVICE or HOST memory)
			body_metadata_t* body_md; 		//!< Vector of additional body parameters (either in the DEVICE or HOST memory)
			ttt_t*			 epoch;			//!< Vector of epoch of the bodies (either in the DEVICE or HOST memory)

			vector<vec_t*>	 d_y;			//!< Device vectors of ODE variables at the beginning of the step (at time t)
			vector<vec_t*>	 d_yout;		//!< Device vectors of ODE variables at the end of the step (at time tout)
			param_t*		 d_p;			//!< Device vector of body parameters
			body_metadata_t* d_body_md; 	//!< Device vector of additional body parameters
			ttt_t*			 d_epoch;		//!< Device vector of epoch of the bodies

			vector<vec_t*>	 h_y;			//!< Host vectors of initial position and velocity of the bodies on the host
			vector<vec_t*>	 h_yout;		//!< Host vectors of ODE variables at the end of the step (at time tout)
			param_t*		 h_p;			//!< Host vector of body parameters
			body_metadata_t* h_body_md; 	//!< Host vector of additional body parameters
			ttt_t*			 h_epoch;		//!< Host vector of epoch of the bodies

			orbelem_t*		 h_oe;			//!< Host vector of the orbital elements

			sim_data()
			{
				p = d_p = h_p = 0x0;
				body_md = d_body_md = h_body_md = 0x0;
				epoch = d_epoch = h_epoch = 0x0;
				h_oe = 0x0;
			}

			void create_aliases(computing_device_t comp_dev)
			{
				switch (comp_dev)
				{
				case COMPUTING_DEVICE_CPU:
					for (int i = 0; i < 2; i++)
					{
						y[i]    = h_y[i];
						yout[i] = h_yout[i];
					}
					p       = h_p;
					body_md = h_body_md;
					epoch   = h_epoch;
					break;
				case COMPUTING_DEVICE_GPU:
					for (int i = 0; i < 2; i++)
					{
						y[i]    = d_y[i];
						yout[i] = d_yout[i];
					}
					p       = d_p;
					body_md = d_body_md;
					epoch   = d_epoch;
					break;
				}
			}

		} sim_data_t;

typedef struct event_data
		{
			event_name_t	event_name;	//!< Name of the event

			ttt_t	t;			//!< Time of the event
			var_t	d;			//!< distance of the bodies

			int		id1;		//!< Id of the survivor
			int		idx1;		//!< Index of the survivor
			param_t p1;			//!< Parameters of the survivor before the event
			vec_t	r1;			//!< Position of survisor
			vec_t	v1;			//!< Velocity of survisor

			int		id2;		//!< Id of the merger
			int		idx2;		//!< Index of the merger
			param_t p2;			//!< Parameters of the merger before the event
			vec_t	r2;			//!< Position of merger
			vec_t	v2;			//!< Velocity of merger

			param_t ps;			//!< Parameters of the survivor after the event
			vec_t	rs;			//!< Position of survivor after the event
			vec_t	vs;			//!< Velocity of survivor after the event

			event_data()
			{
				event_name = EVENT_NAME_NONE;
				t = 0.0;
				d = 0.0;

				id1 = idx1 = 0;
				id2 = idx2 = 0;
				
				param_t p_zero = {0.0, 0.0, 0.0, 0.0};
				vec_t v_zero = {0.0, 0.0, 0.0, 0.0};

				p1 = p2 = ps = p_zero;
				r1 = r2 = rs = v_zero;
				v1 = v2 = vs = v_zero;
			}

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
