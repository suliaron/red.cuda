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
//! Type of unsigned integer tuples variables
typedef uint2  uint2_t;

typedef enum output_name
		{
			OUTPUT_NAME_DUMP,
			OUTPUT_NAME_DUMP_AUX,
			OUTPUT_NAME_EVENT,
			OUTPUT_NAME_INFO,
			OUTPUT_NAME_LOG,
			OUTPUT_NAME_RESULT,
			OUTPUT_NAME_INTEGRAL,
			OUTPUT_NAME_INTEGRAL_EVENT,
			OUTPUT_NAME_N
		} output_name_t;

typedef enum input_name
		{
			INPUT_NAME_BODYLIST,
			INPUT_NAME_PARAMETER,
			INPUT_NAME_GAS_DISK_MODEL,
			INPUT_NAME_N
		} input_name_t;

typedef enum directory_name
		{
			DIRECTORY_NAME_IN,
			DIRECTORY_NAME_OUT,
			DIRECTORY_NAME_N
		} directory_name_t;

typedef enum data_representation
		{
			DATA_REPRESENTATION_ASCII,
			DATA_REPRESENTATION_BINARY,
			DATA_REPRESENTATION_N,
		} data_representation_t;
//static const char* data_representation_name[] = 
//{
//			"ASCII",
//			"BINARY"
//};

typedef enum gas_decrease
		{ 
			GAS_DENSITY_CONSTANT,
			GAS_DENSITY_DECREASE_LINEAR,
			GAS_DENSITY_DECREASE_EXPONENTIAL,
			GAS_DENSITY_N
		} gas_decrease_t;
//static const char* gas_decrease_name[] = 
//{
//			"CONSTANT",
//			"DECREASE_LINEAR",
//			"DECREASE_EXPONENTIAL"
//};

typedef enum gas_disk_model
		{
			GAS_DISK_MODEL_NONE,
			GAS_DISK_MODEL_ANALYTIC,
			GAS_DISK_MODEL_FARGO,
			GAS_DISK_MODEL_N,
		} gas_disk_model_t;
//static const char* gas_disk_model_name[] = 
//{
//			"NONE",
//			"ANALYTIC",
//			"FARGO"
//};

typedef enum collision_detection_model
		{
			COLLISION_DETECTION_MODEL_STEP,
			COLLISION_DETECTION_MODEL_SUB_STEP,
			COLLISION_DETECTION_MODEL_INTERPOLATION,
			COLLISION_DETECTION_MODEL_N,
		} collision_detection_model_t;
//static const char* collision_detection_model_name[] = 
//{
//			"STEP",
//			"SUB_STEP",
//			"INTERPOLATION"
//};

typedef enum computing_device
		{
			COMPUTING_DEVICE_CPU,
			COMPUTING_DEVICE_GPU,
			COMPUTING_DEVICE_N
		} computing_device_t;
static const char* computing_device_name[] = 
{
			"CPU",
			"GPU"
};

typedef enum threshold
		{
			THRESHOLD_HIT_CENTRUM_DISTANCE,
			THRESHOLD_EJECTION_DISTANCE,
			THRESHOLD_RADII_ENHANCE_FACTOR,
			THRESHOLD_N
		} threshold_t;
//static const char* threshold_name[] = 
//{
//			"HIT_CENTRUM_DISTANCE",
//			"EJECTION_DISTANCE",
//			"RADII_ENHANCE_FACTOR"
//};

typedef enum integrator_type
		{ 
			INTEGRATOR_EULER,
			INTEGRATOR_RUNGEKUTTA2,
			INTEGRATOR_RUNGEKUTTA4,
			INTEGRATOR_RUNGEKUTTA5,
			INTEGRATOR_RUNGEKUTTAFEHLBERG78,
			INTEGRATOR_RUNGEKUTTANYSTROM,
		} integrator_type_t;
//static const char* integrator_type_name[] = 
//{
//			"EULER",
//			"RUNGEKUTTA2",
//			"RUNGEKUTTA4",
//			"RUNGEKUTTA5",
//			"RUNGEKUTTAFEHLBERG78",
//			"RUNGEKUTTANYSTROM"
//};
static const char* integrator_type_short_name[] = 
{
			"E",
			"RK2",
			"RK4",
			"RK5",
			"RKF8",
			"RKN"
};

typedef enum event_name
		{
			EVENT_NAME_NONE,
			EVENT_NAME_HIT_CENTRUM,
			EVENT_NAME_EJECTION,
			EVENT_NAME_CLOSE_ENCOUNTER,
			EVENT_NAME_COLLISION,
			EVENT_NAME_N
		} event_name_t;
//static const char* event_name_name[] = 
//{
//			"NONE",
//			"HIT_CENTRUM",
//			"EJECTION",
//			"CLOSE_ENCOUNTER",
//			"COLLISION",
//};

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
static const char* migration_type_name[] = 
{
			"NO",
			"TYPE_I",
			"TYPE_II"
};

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
static const char* body_type_name[] = 
		{
			"STAR",
			"GIANTPLANET",
			"ROCKYPLANET",
			"PROTOPLANET",
			"SUPERPLANETESIMAL",
			"PLANETESIMAL",
			"TESTPARTICLE"
		};

typedef enum integral_name
        {
            INTEGRAL_NAME_POSITION_OF_BC,     /* Position vector of the system's barycenter */
            INTEGRAL_NAME_VELOCITY_OF_BC,     /* Velocity vector of the system's barycenter */
            INTEGRAL_NAME_ANGULAR_MOMENTUM,   /* Angular momentum vector of the system      */
            INTEGRAL_NAME_ENERGY,             /* Total energy of the system                 */
        } integral_name_t;

typedef struct dump_aux_data
		{
			ttt_t dt;
		} dump_aux_data_t;

typedef struct orbelem
		{			
			var_t sma;   //!< Semimajor-axis of the body       [8 byte]
			var_t ecc;   //!< Eccentricity of the body         [8 byte]
			var_t inc;   //!< Inclination of the body          [8 byte]
			var_t peri;  //!< Argument of the pericenter       [8 byte]
			var_t node;  //!< Longitude of the ascending node  [8 byte]
			var_t mean;  //!< Mean anomaly                     [8 byte] -> 48 byte
		} orbelem_t;

#ifdef _WIN64
#define __BUILTIN_ALIGN__ __builtin_align__(16)
#else
#define __BUILTIN_ALIGN__
#endif

// int4_t gets aligned to 16 bytes.
typedef struct __BUILTIN_ALIGN__ _int4
		{
			int_t x;  // [ 4 byte]
			int_t y;  // [ 4 byte]
			int_t z;  // [ 4 byte]
			int_t w;  // [ 4 byte]
		} int4_t;     // [16 byte]

// var2_t gets aligned to 16 bytes.
typedef struct __BUILTIN_ALIGN__ _var2
		{
			var_t x;  // [ 8 byte]
			var_t y;  // [ 8 byte]
		} var2_t;     // [16 byte]

// vec_t gets aligned to 16 bytes.
typedef struct __BUILTIN_ALIGN__ vec
		{
			var_t x;  // [ 8 byte]
			var_t y;  // [ 8 byte]
			var_t z;  // [ 8 byte]
			var_t w;  // [ 8 byte]
		} vec_t;      // [32 byte]

// param_t gets aligned to 16 bytes.
typedef struct __BUILTIN_ALIGN__ param
		{
			var_t mass;      // [ 8 byte]
			var_t radius;    // [ 8 byte]
			var_t density;   // [ 8 byte]
			var_t cd;	     // [ 8 byte]
		} param_t;           // [32 byte]

// body_metadata_t gets aligned to 16 bytes.
typedef struct __BUILTIN_ALIGN__ body_metadata
		{
			int32_t id;             // [ 4 byte]
			int32_t body_type;	    // [ 4 byte]
			int32_t mig_type;	    // [ 4 byte]
			var_t	mig_stop_at;    // [ 8 byte]
		} body_metadata_t;          // [20 byte]

// body_metadata_t gets aligned to 16 bytes.
typedef struct __BUILTIN_ALIGN__ body_metadata_new
		{
			int32_t id;            // [ 4 byte]
			char    body_type;     // [ 1 byte]
			char    mig_type;      // [ 1 byte]
			bool	active;        // [ 1 byte]
			bool	unused;        // [ 1 byte]
			var_t   mig_stop_at;   // [ 8 byte]
		} body_metadata_new_t;     // [16 byte]

typedef struct integral
		{			
            vec_t R;   //!< Position vector of the system's barycenter [24 byte]
			vec_t V;   //!< Velocity vector of the system's barycenter [24 byte]
			vec_t C;   //!< Angular momentum vector of the system      [24 byte]
			var_t E;   //!< Total energy of the system                 [ 8 byte]
		} integral_t;

typedef struct analytic_gas_disk_params
		{
			var2_t rho;   //!< The density of the gas disk in the midplane (time dependent)	
			var2_t sch;   //!< The scale height of the gas disk
			var2_t eta;   //!< Describes how the velocity of the gas differs from the circular velocity	
			var2_t tau;   //!< Describes the Type 2 migartion of the giant planets

			var2_t mfp;   //!< The mean free path of the gas molecules (calculated based on rho, time dependent)	
			var2_t temp;  //!< The temperaterure of the gas (calculated based on sch)
	
			var_t c_vth;  //!< Constant for computing the mean thermal velocity (calculated, constant)

			gas_decrease_t gas_decrease;  //!< The decrease type for the gas density

			ttt_t t0;   //!< Time when the decrease of gas starts (for linear and exponential)
			ttt_t t1;   //!< Time when the linear decrease of the gas ends
			ttt_t e_folding_time; //!< The exponent for the exponential decrease

			var_t alpha;  //!< The viscosity parameter for the Shakura & Sunyaev model (constant)
			var_t mean_molecular_weight;  //!< The mean molecular weight in units of the proton mass (constant)
			var_t particle_diameter;  //!< The mean molecular diameter (constant)
		} analytic_gas_disk_params_t;

typedef struct fargo_gas_disk_params
		{
			var_t aspect_ratio;        // Thickness over Radius in the disc
			var_t sigma_0;             // Surface Density at r=1
			var_t alpha_viscosity;     // Uniform kinematic viscosity
			var_t sigma_slope;         // Slope of surface density profile.
			var_t flaring_index;       // gamma; H(r) = h * r^(1 + gamma)

			bool exclude_hill;

			//Planet parameters
			var_t thickness_smoothing; // Smoothing parameters in disk thickness

			// Numerical method parameters
			var_t omega_frame;

			// Mesh parameters
			int n_rad;                 // Radial number of zones
			int n_sec;                 // Azimuthal number of zones (sectors)
			var_t r_min;               // Inner boundary radius
			var_t r_max;               // Outer boundary radius

			// Output control parameters
			int n_tot;                 // Total number of time steps
			int n_interm;              // Time steps between outputs
			var_t dT;                  // Time step length. 2PI = 1 orbit

			// Viscosity damping due to a dead zone
			var_t visc_mod_r1;         // Inner radius of dead zone
			var_t visc_mod_delta_r1;   // Width of viscosity transition at inner radius
			var_t visc_mod_r2;         // Outer radius of dead zone
			var_t visc_mod_delta_r2;   // Width of viscosity transition at outer radius
			var_t visc_mod;            // Viscosity damp
		} fargo_gas_disk_params_t;

typedef struct sim_data
		{
			vector<vec_t*>	 y;				//!< Vectors of initial position and velocity of the bodies on the host (either in the DEVICE or HOST memory)
			vector<vec_t*>	 yout;			//!< Vectors of ODE variables at the end of the step (at time tout) (either in the DEVICE or HOST memory)
			param_t*		 p;   			//!< Vector of body parameters (either in the DEVICE or HOST memory)
			body_metadata_t* body_md; 		//!< Vector of additional body parameters (either in the DEVICE or HOST memory)
			ttt_t*			 epoch;			//!< Vector of epoch of the bodies (either in the DEVICE or HOST memory)
			orbelem_t*		 oe;			//!< Vector of of the orbital elements (either in the DEVICE or HOST memory)

			vector<vec_t*>	 d_y;			//!< Device vectors of ODE variables at the beginning of the step (at time t)
			vector<vec_t*>	 d_yout;		//!< Device vectors of ODE variables at the end of the step (at time tout)
			param_t*		 d_p;			//!< Device vector of body parameters
			body_metadata_t* d_body_md; 	//!< Device vector of additional body parameters
			ttt_t*			 d_epoch;		//!< Device vector of epoch of the bodies
			orbelem_t*		 d_oe;			//!< Device vector of the orbital elements

			vector<vec_t*>	 h_y;			//!< Host vectors of initial position and velocity of the bodies on the host
			vector<vec_t*>	 h_yout;		//!< Host vectors of ODE variables at the end of the step (at time tout)
			param_t*		 h_p;			//!< Host vector of body parameters
			body_metadata_t* h_body_md; 	//!< Host vector of additional body parameters
			ttt_t*			 h_epoch;		//!< Host vector of epoch of the bodies
			orbelem_t*		 h_oe;			//!< Host vector of the orbital elements

			sim_data()
			{
				p       = d_p       = h_p       = 0x0;
				body_md = d_body_md = h_body_md = 0x0;
				epoch   = d_epoch   = h_epoch   = 0x0;
				oe      = d_oe      = h_oe      = 0x0;
			}
		} sim_data_t;

typedef struct event_data
		{
			event_name_t	event_name;	//!< Name of the event

			ttt_t	t;			//!< Time of the event
			var_t	d;			//!< distance of the bodies

			int id1;            //!< Id of the survivor
			unsigned int idx1;	//!< Index of the survivor
			param_t p1;			//!< Parameters of the survivor before the event
			vec_t	r1;			//!< Position of survisor
			vec_t	v1;			//!< Velocity of survisor

			int		id2;		//!< Id of the merger
			unsigned int idx2; //!< Index of the merger
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
	uint2_t	sink;
	uint2_t	source;

	interaction_bound()
	{
		sink.x   = sink.y   = 0;
		source.x = source.y = 0;
	}

	interaction_bound(uint2_t sink, uint2_t source) : 
		sink(sink),
		source(source) 
	{ }

	interaction_bound(unsigned int x0, unsigned int y0, unsigned int x1, unsigned int y1)
	{
		sink.x = x0;		sink.y = y0;
		source.x = x1;		source.y = y1;
	}
};
