#pragma once

#include <cstring>     // memcpy
#include <stdint.h>
#include <vector>

// include CUDA
#include "cuda_runtime.h"

#ifdef _WIN64
#define __BUILTIN_ALIGN__ __builtin_align__(16)
#else
#define __BUILTIN_ALIGN__
#endif

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
typedef unsigned char uchar_t;

typedef enum dyn_model
		{
			DYN_MODEL_TBP1D,
			DYN_MODEL_TBP3D,

			DYN_MODEL_RTBP1D,
			DYN_MODEL_RTBP3D,

			DYN_MODEL_N
		} dyn_model_t;

typedef enum input_format_name
		{
			INPUT_FORMAT_RED,
			INPUT_FORMAT_NONAME,
			INPUT_FORMAT_HIPERION,
            INPUT_FORMAT_N
		} input_format_name_t;

typedef enum copy_direction
		{
			COPY_DIRECTION_TO_HOST,
			COPY_DIRECTION_TO_DEVICE,
			COPY_DIRECTION_TO_N
		} copy_direction_t;

typedef enum output_name
		{
			OUTPUT_NAME_LOG,
			OUTPUT_NAME_INFO,
			OUTPUT_NAME_EVENT,
			OUTPUT_NAME_DATA,
			OUTPUT_NAME_DATA_INFO,
			OUTPUT_NAME_INTEGRAL,
			OUTPUT_NAME_INTEGRAL_EVENT,
			OUTPUT_NAME_N
			//OUTPUT_NAME_DUMP,
			//OUTPUT_NAME_DUMP_AUX,
			//OUTPUT_NAME_EVENT,
			//OUTPUT_NAME_INFO,
			//OUTPUT_NAME_LOG,
			//OUTPUT_NAME_RESULT,
			//OUTPUT_NAME_INTEGRAL,
			//OUTPUT_NAME_INTEGRAL_EVENT,
			//OUTPUT_NAME_N
		} output_name_t;

typedef enum input_name
		{
			INPUT_NAME_START_FILES,
			INPUT_NAME_DATA,
			INPUT_NAME_DATA_INFO,
			INPUT_NAME_PARAMETER,
			INPUT_NAME_GAS_DISK_MODEL,
			INPUT_NAME_N
			//INPUT_NAME_BODYLIST,
			//INPUT_NAME_PARAMETER,
			//INPUT_NAME_GAS_DISK_MODEL,
			//INPUT_NAME_N
		} input_name_t;

typedef enum directory_name
		{
			DIRECTORY_NAME_IN,
			DIRECTORY_NAME_OUT,
			DIRECTORY_NAME_N
		} directory_name_t;

typedef enum data_rep
		{
			DATA_REPRESENTATION_ASCII,
			DATA_REPRESENTATION_BINARY,
			DATA_REPRESENTATION_N,
		} data_rep_t;

typedef enum gas_decrease
		{ 
			GAS_DENSITY_CONSTANT,
			GAS_DENSITY_DECREASE_LINEAR,
			GAS_DENSITY_DECREASE_EXPONENTIAL,
			GAS_DENSITY_N
		} gas_decrease_t;

typedef enum gas_disk_model
		{
			GAS_DISK_MODEL_NONE,
			GAS_DISK_MODEL_ANALYTIC,
			GAS_DISK_MODEL_FARGO,
			GAS_DISK_MODEL_N,
		} gas_disk_model_t;

typedef enum computing_device
		{
			COMPUTING_DEVICE_CPU,
			COMPUTING_DEVICE_GPU,
			COMPUTING_DEVICE_N
		} computing_device_t;

typedef enum threshold
		{
			THRESHOLD_HIT_CENTRUM_DISTANCE,  //! inside this limit the body is considered to have hitted the central body and removed from the simulation [AU]
			THRESHOLD_EJECTION_DISTANCE,     //! beyond this limit the body is removed from the simulation [AU]
			THRESHOLD_RADII_ENHANCE_FACTOR,  //! two bodies collide when their mutual distance is smaller than the sum of their radii multiplied by this number. Real physical collision corresponds to the value of 1.0.
			THRESHOLD_N
		} threshold_t;

typedef enum integrator_type
		{ 
			INTEGRATOR_EULER,
			INTEGRATOR_RUNGEKUTTA2,
			INTEGRATOR_RUNGEKUTTA4,
			INTEGRATOR_RUNGEKUTTAFEHLBERG56,
			INTEGRATOR_RUNGEKUTTAFEHLBERG78,
		} integrator_type_t;

typedef enum event_name
		{
			EVENT_NAME_NONE,
			EVENT_NAME_HIT_CENTRUM,
			EVENT_NAME_EJECTION,
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
			BODY_TYPE_N
		} body_type_t;

// int4_t gets aligned to 16 bytes.
typedef struct __BUILTIN_ALIGN__ _int4
		{
			int_t x;   // [ 4 byte]
			int_t y;   // [ 4 byte]
			int_t z;   // [ 4 byte]
			int_t w;   // [ 4 byte]
		} int4_t;      // [16 byte]

// var2_t gets aligned to 16 bytes.
typedef struct __BUILTIN_ALIGN__ _var2
		{
			var_t x;   // [ 8 byte]
			var_t y;   // [ 8 byte]
		} var2_t;      // [16 byte]

// var3_t gets aligned to 16 bytes.
typedef struct __BUILTIN_ALIGN__ var3
		{
			var_t x;   // [ 8 byte]
			var_t y;   // [ 8 byte]
			var_t z;   // [ 8 byte]
		} var3_t;      // [32 byte]

// var4_t gets aligned to 16 bytes.
typedef struct __BUILTIN_ALIGN__ var4
		{
			var_t x;   // [ 8 byte]
			var_t y;   // [ 8 byte]
			var_t z;   // [ 8 byte]
			var_t w;   // [ 8 byte]
		} var4_t;      // [32 byte]

typedef struct orbelem
		{			
			var_t sma;   //!< Semimajor-axis of the body       [8 byte]
			var_t ecc;   //!< Eccentricity of the body         [8 byte]
			var_t inc;   //!< Inclination of the body          [8 byte]
			var_t peri;  //!< Argument of the pericenter       [8 byte]
			var_t node;  //!< Longitude of the ascending node  [8 byte]
			var_t mean;  //!< Mean anomaly                     [8 byte] -> 48 byte
		} orbelem_t;

typedef struct ode_data
{
	ttt_t t;                       //! Current time (the ODE variables in y are are valid for this time)
	ttt_t tout;                    //! Time at the end of the integration step (the ODE variables in yout are are valid for this time)

	std::vector<var_t*>   y;	   //!< Vectors of initial position and velocity of the bodies on the host (either in the DEVICE or HOST memory)
	std::vector<var_t*>   yout;    //!< Vectors of ODE variables at the end of the step (at time tout) (either in the DEVICE or HOST memory)
	var_t*           p;            //!< Vector of body parameters (either in the DEVICE or HOST memory)
	void*            obj_md;       //!< Vector of additional object parameters (either in the DEVICE or HOST memory)

	std::vector<var_t*>   d_y;     //!< Device vectors of ODE variables at the beginning of the step (at time t)
	std::vector<var_t*>   d_yout;  //!< Device vectors of ODE variables at the end of the step (at time tout)
	var_t*           d_p;          //!< Device vector of body parameters
	void*            d_obj_md;     //!< Device vector of additional body parameters

	std::vector<var_t*>   h_y;     //!< Host vectors of ODE variables at the beginning of the step (at time t)
	std::vector<var_t*>   h_yout;  //!< Host vectors of ODE variables at the end of the step (at time tout)
	var_t*           h_p;          //!< Host vector of body parameters
	void*            h_obj_md;     //!< Host vector of additional body parameters

	uint32_t n_obj;                //! The total number of objets in the problem
	uint16_t n_dim;                //! The space dimension of the problem 

	uint16_t n_vpo;                //! The number of variables per object (vpo)
	uint16_t n_ppo;                //! The number of parameters per object (ppo)

	uint32_t n_var;                //! The total number of variables of the problem
	uint32_t n_par;                //! The total number of parameters of the problem

	dim3 grid;                     //! Defines the grid of the blocks of the current execution
	dim3 block;                    //! Defines the block of the threads of the current execution
	uint16_t n_tpb;                //! Holds the number of threads per block
	computing_device_t comp_dev;   //! The execution is performed on this device

	ode_data()
	{
		comp_dev = COMPUTING_DEVICE_CPU;
        t = tout = 0.0;

		p      = d_p      = h_p      = 0x0;
		obj_md = d_obj_md = h_obj_md = 0x0;
		n_obj = n_dim = 0;
		n_vpo = n_ppo = 0;
		n_var = n_par = 0;
		n_tpb = 1;
	}
} ode_data_t;


namespace tbp1D_t
{
	typedef struct metadata
	{
		int32_t id;
	} metadata_t;

	typedef struct param
	{
		var_t mu;
	} param_t;
} /* namespace tbp1D_t */

namespace tbp3D_t
{
	typedef struct metadata
	{
		int32_t id;
	} metadata_t;

	typedef struct param
	{
		var_t mu;
	} param_t;
} /* namespace tbp3D_t */

namespace threebody_t
{
	typedef struct metadata
	{
		int32_t id;
	} metadata_t;

	typedef struct param
	{
		var_t m;
	} param_t;
} /* namespace threebody_t */

namespace nbody_t
{
	// param_t gets aligned to 16 bytes.
	typedef struct __BUILTIN_ALIGN__ param
	{
		var_t mass;             // [ 8 byte]
		var_t radius;           // [ 8 byte]
		var_t density;          // [ 8 byte]
		var_t cd;	            // [ 8 byte]
	} param_t;                  // [32 byte]

	// body_metadata_t gets aligned to 16 bytes.
	typedef struct __BUILTIN_ALIGN__ body_metadata
	{
		int32_t id;             // [ 4 byte]
		char    body_type;      // [ 1 byte]
		char    mig_type;       // [ 1 byte]
		bool	active;         // [ 1 byte]
		bool	unused;         // [ 1 byte]
		var_t   mig_stop_at;    // [ 8 byte]
	} body_metadata_t;      // [16 byte]

	typedef struct integral
	{			
		var3_t R;               //!< Position vector of the system's barycenter [24 byte]
		var3_t V;               //!< Velocity vector of the system's barycenter [24 byte]
		var3_t C;               //!< Angular momentum vector of the system      [24 byte]
		var_t E;                //!< Total energy of the system                 [ 8 byte]
	} integral_t;               // [80 byte]

} /* nbody_t */

namespace pp_disk_t
{
	// param_t gets aligned to 16 bytes.
	typedef struct __BUILTIN_ALIGN__ param
	{
		var_t mass;             // [ 8 byte]
		var_t radius;           // [ 8 byte]
		var_t density;          // [ 8 byte]
		var_t cd;	            // [ 8 byte]
	} param_t;                  // [32 byte]

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
		int32_t id;             // [ 4 byte]
		char    body_type;      // [ 1 byte]
		char    mig_type;       // [ 1 byte]
		bool	active;         // [ 1 byte]
		bool	unused;         // [ 1 byte]
		var_t   mig_stop_at;    // [ 8 byte]
	} body_metadata_new_t;      // [16 byte]

	typedef struct integral
	{			
		var4_t R;               //!< Position vector of the system's barycenter [32 byte]
		var4_t V;               //!< Velocity vector of the system's barycenter [32 byte]
		var4_t C;               //!< Angular momentum vector of the system      [32 byte]
		var_t E;                //!< Total energy of the system                 [ 8 byte]
	} integral_t;               // [104 byte]

	typedef struct sim_data
	{
		std::vector<var4_t*>	 y;		//!< Vectors of initial position and velocity of the bodies on the host (either in the DEVICE or HOST memory)
		std::vector<var4_t*>	 yout;	//!< Vectors of ODE variables at the end of the step (at time tout) (either in the DEVICE or HOST memory)
		param_t*		 p;   			//!< Vector of body parameters (either in the DEVICE or HOST memory)
		body_metadata_t* body_md; 		//!< Vector of additional body parameters (either in the DEVICE or HOST memory)
		ttt_t*			 epoch;			//!< Vector of epoch of the bodies (either in the DEVICE or HOST memory)
		orbelem_t*		 oe;			//!< Vector of of the orbital elements (either in the DEVICE or HOST memory)

		std::vector<var4_t*>	 d_y;	//!< Device vectors of ODE variables at the beginning of the step (at time t)
		std::vector<var4_t*>	 d_yout;//!< Device vectors of ODE variables at the end of the step (at time tout)
		param_t*		 d_p;			//!< Device vector of body parameters
		body_metadata_t* d_body_md; 	//!< Device vector of additional body parameters
		ttt_t*			 d_epoch;		//!< Device vector of epoch of the bodies
		orbelem_t*		 d_oe;			//!< Device vector of the orbital elements

		std::vector<var4_t*>	 h_y;	//!< Host vectors of initial position and velocity of the bodies on the host
		std::vector<var4_t*>	 h_yout;//!< Host vectors of ODE variables at the end of the step (at time tout)
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
		event_name_t event_name;       //!< Name of the event

		ttt_t	t;                     //!< Time of the event
		var_t	d;                     //!< distance of the bodies

		int id0;                       //!< Id of the star
		uint32_t idx0;                 //!< Index of the star
		param_t p0;                    //!< Parameters of the star
		var4_t	r0;                    //!< Position of star
		var4_t	v0;                    //!< Velocity of star

		int id1;                       //!< Id of the survivor
		uint32_t idx1;                 //!< Index of the survivor
		param_t p1;                    //!< Parameters of the survivor before the event
		var4_t	r1;                    //!< Position of survisor
		var4_t	v1;                    //!< Velocity of survisor

		int		id2;                   //!< Id of the merger
		uint32_t idx2;                 //!< Index of the merger
		param_t p2;                    //!< Parameters of the merger before the event
		var4_t	r2;                    //!< Position of merger
		var4_t	v2;                    //!< Velocity of merger

		param_t ps;                    //!< Parameters of the survivor after the event
		var4_t	rs;                    //!< Position of survivor after the event
		var4_t	vs;                    //!< Velocity of survivor after the event

		event_data()
		{
			event_name = EVENT_NAME_NONE;
			t = 0.0;
			d = 0.0;

            id0 = idx0 = 0;
			id1  = id2  = 0;
			idx1 = idx2 = 0;
				
			param_t p_zero = {0.0, 0.0, 0.0, 0.0};
			var4_t v_zero = {0.0, 0.0, 0.0, 0.0};

			p0 = p1 = p2 = ps = p_zero;
			r0 = r1 = r2 = rs = v_zero;
			v0 = v1 = v2 = vs = v_zero;
		}
	} event_data_t;
} /* namespace pp_disk_t */

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

	interaction_bound(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1)
	{
		sink.x = x0;		sink.y = y0;
		source.x = x1;		source.y = y1;
	}
};

typedef struct n_objects
{
	n_objects(uint32_t n_s, uint32_t n_gp, uint32_t n_rp, uint32_t n_pp, uint32_t n_spl, uint32_t n_pl, uint32_t n_tp)
	{
		initial[BODY_TYPE_STAR]              = n_s;
		initial[BODY_TYPE_GIANTPLANET]       = n_gp;
		initial[BODY_TYPE_ROCKYPLANET]       = n_rp;
		initial[BODY_TYPE_PROTOPLANET]       = n_pp;
		initial[BODY_TYPE_SUPERPLANETESIMAL] = n_spl;
		initial[BODY_TYPE_PLANETESIMAL]      = n_pl;
		initial[BODY_TYPE_TESTPARTICLE]      = n_tp;

		memcpy(playing, initial, sizeof(playing));

		memset(inactive, 0, sizeof(inactive));
		memset(removed,  0, sizeof(removed));

		n_removed = 0;

		sink.x   = sink.y   = 0;
		source.x = source.y = 0;
	}

	void update()
	{
		n_removed = 0;
		for (uint32_t i = 0; i < BODY_TYPE_N; i++)
		{
			playing[i] -= inactive[i];
			removed[i] += inactive[i];
			n_removed  += inactive[i];
			inactive[i] = 0;
		}
	}

	uint32_t get_n_SI() 
	{
		return (playing[BODY_TYPE_STAR] + playing[BODY_TYPE_GIANTPLANET] + playing[BODY_TYPE_ROCKYPLANET] + playing[BODY_TYPE_PROTOPLANET]);
	}

	uint32_t get_n_NSI()
	{
		return (playing[BODY_TYPE_SUPERPLANETESIMAL] + playing[BODY_TYPE_PLANETESIMAL]);
	}

	uint32_t get_n_NI()
	{
		return playing[BODY_TYPE_TESTPARTICLE];
	}

	uint32_t get_n_total_initial()
	{
		uint32_t n = 0;
		for (uint32_t i = 0; i < BODY_TYPE_N; i++)
		{
			n += initial[i];
		}
		return n; 
	}

	uint32_t get_n_total_playing()
	{
		uint32_t n = 0;
		for (uint32_t i = 0; i < BODY_TYPE_N; i++)
		{
			n += playing[i];
		}
		return n; 
	}

	uint32_t get_n_total_active()
	{
		uint32_t n = 0;
		for (uint32_t i = 0; i < BODY_TYPE_N; i++)
		{
			n += playing[i] - inactive[i];
		}
		return n; 
	}

	uint32_t get_n_total_inactive()
	{
		uint32_t n = 0;
		for (uint32_t i = 0; i < BODY_TYPE_N; i++)
		{
			n += inactive[i];
		}
		return n; 
	}

	uint32_t get_n_total_removed()
	{
		uint32_t n = 0;
		for (uint32_t i = 0; i < BODY_TYPE_N; i++)
		{
			n += removed[i];
		}
		return n; 
	}

	uint32_t get_n_GD()
	{
		return (playing[BODY_TYPE_SUPERPLANETESIMAL] + playing[BODY_TYPE_PLANETESIMAL]);
	}

	uint32_t get_n_MT1()
	{
		return (playing[BODY_TYPE_ROCKYPLANET] + playing[BODY_TYPE_PROTOPLANET]);
	}

	uint32_t get_n_MT2()
	{
		return playing[BODY_TYPE_GIANTPLANET];
	}

	uint32_t get_n_massive()
	{
		return (get_n_SI() + get_n_NSI());
	}

	uint32_t get_n_active_by(body_type_t type)
	{
		return (playing[type] - inactive[type]);
	}

	interaction_bound get_bound_SI()
	{
		sink.x   = 0, sink.y   = get_n_SI();
		source.x = 0, source.y = get_n_massive();

		return interaction_bound(sink, source);
	}

	interaction_bound get_bound_NSI()
	{
		sink.x   = get_n_SI(), sink.y   = sink.x + get_n_NSI();
		source.x = 0,		   source.y = get_n_SI();

		return interaction_bound(sink, source);
	}

	interaction_bound get_bound_NI()
	{
		sink.x   = get_n_massive(), sink.y   = sink.x + get_n_NI();
		source.x = 0,   	        source.y = get_n_massive();

		return interaction_bound(sink, source);
	}

	interaction_bound get_bound_GD()
	{
		sink.x   = get_n_SI(), sink.y   = sink.x + get_n_NSI();
		source.x = 0,		   source.y = 0;

		return interaction_bound(sink, source);
	}

	uint32_t initial[ BODY_TYPE_N];   //!< Number of initial bodies
	uint32_t playing[ BODY_TYPE_N];   //!< Number of bodies which are iterated over in the gravitational computation (may have negative id)
	uint32_t inactive[BODY_TYPE_N];   //!< Number of bodies which has negative id (these are part of the playing bodies, and are flaged to be removed in the next call to remove inactive bodies function)
	uint32_t removed[ BODY_TYPE_N];   //!< Number of removed bodies

	uint32_t n_removed;               //!< Number of bodies which were removed during the last update() function call

	uint2_t sink;
	uint2_t source;

} n_objects_t;
