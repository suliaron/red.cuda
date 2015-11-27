#pragma once

// includes system
#include <string>
#include <vector>

// includes project
#include "analytic_gas_disk.h"
#include "fargo_gas_disk.h"
#include "number_of_bodies.h"
#include "red_type.h"

using namespace std;

class pp_disk
{
public:

	pp_disk(number_of_bodies *n_bodies,             gas_disk_model_t g_disk_model, collision_detection_model_t cdm, unsigned int id_dev, computing_device_t comp_dev);
	pp_disk(string& path, bool continue_simulation, gas_disk_model_t g_disk_model, collision_detection_model_t cdm, unsigned int id_dev, computing_device_t comp_dev, const var_t* thrshld, bool pts);
	~pp_disk();

	//! Initialize the members to default values
	void initialize();

	//! Copies ODE parameters and variables from the host to the cuda device
	void copy_to_device();
	void copy_disk_params_to_device();
	//! Copies ODE parameters and variables from the cuda device to the host
	void copy_to_host();
	//! Copies the event data from the cuda device to the host
	void copy_event_data_to_host();

	//! Set the computing device to calculate the accelerations
	/*!
		\param device specifies which device will execute the computations
	*/
	void set_computing_device(computing_device_t device);
	computing_device_t get_computing_device() { return comp_dev; }

	void set_cdm(collision_detection_model_t c) { cdm = c;    }
	collision_detection_model_t get_cdm(void)   { return cdm; }

	void set_n_tpb(unsigned int n)   { n_tpb = n;     }
	unsigned int  get_n_tpb(void)    { return n_tpb;  }

	void set_id_dev(unsigned int id) { id_dev = id;   }
	unsigned int  get_id_dev(void)   { return id_dev; }

	//! Determines the mass of the central star
	/*!
		\return The mass of the central star
	*/
	var_t get_mass_of_star();
	//! Transforms the system to barycentric reference frame
	/*!
		\param pts If true then writes the position and velocity of the barycenter to the standard output.
	*/
	void transform_to_bc(bool pts);
	//! Transform the time using the new time unit: 1/k = 58.13244 ...
	void transform_time();
	//! Transform the velocity using the new time unit: 1/k = 58.13244 ...
	void transform_velocity();

	//! Print the data of all bodies in text format
	/*!
		\param sout   print the data to this stream
		\param repres indicates the data representation of the file, i.e. text or binary
	*/
	void print_result(ofstream& sout, data_representation_t repres);
	//! Print the data of all bodies in text format
	/*   
		\param sout print the data to this stream
	*/
	void print_result_ascii(ofstream& sout);
	//! Print the data of all bodies in binary format
	/*!
		\param sout print the data to this stream
	*/
	void print_result_binary(ofstream& sout);
	//! Print the data of all bodies in text format
	/*!
		\param sout   print the data to this stream
		\param repres indicates the data representation of the file, i.e. text or binary
	*/
	void print_dump(ofstream& sout, data_representation_t repres);
	//! Print the event data
	/*!
		\param sout print the data to this stream
		\param log_f print the data to this stream
	*/
	void print_event_data(ofstream& sout, ofstream& log_f);
	//! Print the classical integrals
	/*!
		\param sout print the data to this stream
	*/
	void print_integral_data(integral_t& I, ofstream& sout);

	//! Returns the number of events during the simulation
	unsigned int get_n_total_event();

	//! Clears the event_counter (sets to 0)
	void clear_event_counter();
	//! Sets all event counter to a specific value
	/*!
		\param field Sets this field
		\param value The value to set the field
	*/
	void set_event_counter(event_counter_name_t field, uint32_t value);

	//! From the events it will create a vector containing one entry for each colliding pair with the earliest collision time
	void populate_sp_events();

	void handle_collision();
	void handle_ejection_hit_centrum();
	void handle_collision_pair(unsigned int i, event_data_t *collision);

	void rebuild_vectors();

	//! Check all bodies against ejection and hit centrum criterium. The number of detected events are stored in the event_counter member variable.
	bool check_for_ejection_hit_centrum();
	//! Check collisin between all bodies. The number of detected events are stored in the event_counter member variable.
	bool check_for_collision();

	//! Check all bodies against ejection and hit centrum criterium. The number of detected events are stored in the event_counter member variable.
	void gpu_check_for_ejection_hit_centrum();
	void cpu_check_for_ejection_hit_centrum();

	void gpu_check_for_collision();
	void cpu_check_for_collision();
	void cpu_check_for_collision(interaction_bound int_bound, bool SI_NSI, bool SI_TP, bool NSI, bool NSI_TP);

	//! Calculates the differentials of variables
	/*!
		This function is called by the integrator when calculation of the differentials is necessary
		\param i Order of the variables to calculate (e.g. 0: velocities, 1: acceleration ...)
		\param rr Number of substep, used with higher order methods, like Runge-Kutta
		\param curr_t Time
		\param r Device vector with position variables
		\param v Device vector with velocity variables
		\param dy Device vector that will hold the differentials
	*/
	void calc_dydx(unsigned int i, unsigned int rr, ttt_t curr_t, const vec_t *r, const vec_t *v, vec_t* dy);

	void calc_integral(integral_t& integrals);
	void calc_integral_CMU(integral_t& integrals);

	//! Swaps the yout with the y variable, i.e. at the end of an integration step the output will be the input of the next step
	void swap();

	ttt_t t;                                   //!< time when the variables are valid
	number_of_bodies* n_bodies;                //!< class containing the number of different bodies
	sim_data_t* sim_data;                      //!< struct containing all the data of the simulation

	gas_disk_model_t g_disk_model;
	analytic_gas_disk* a_gd;
	fargo_gas_disk* f_gd;

	uint32_t n_ejection[   EVENT_COUNTER_NAME_N];   //!< Number of ejection
	uint32_t n_hit_centrum[EVENT_COUNTER_NAME_N];   //!< Number of hit centrum
	uint32_t n_collision[  EVENT_COUNTER_NAME_N];   //!< Number of collision
	uint32_t n_event[      EVENT_COUNTER_NAME_N];   //!< Number of total events

private:
	void increment_event_counter(uint32_t *event_counter);
	//! Loads the initial position and velocity of the bodies
	/*!
		\param path the full path of the data file
		\param repres the representation of the data in the file (i.e. ASCII or BINARY)
	*/
	void load(string& path, data_representation_t repres);
	//! Loads the initial position and velocity of the bodies
	/*!
		\param input the input stream from which to read the data
	*/
	void load_ascii(ifstream& input);
	//! Loads the initial position and velocity of the bodies
	/*!
		\param input the input stream from which to read the data
	*/
	void load_binary(ifstream& input);
	void load_dump(string& path, data_representation_t repres);
	void load_body_record(ifstream& input, unsigned int k, ttt_t* epoch, body_metadata_t* body_md, param_t* p, vec_t* r, vec_t* v);

	//! Loads the number of bodies from the file
	/*!
		\param path the full path of the data file
		\param repres the representation of the data in the file (i.e. ASCII or BINARY)
	*/
	number_of_bodies* load_number_of_bodies(string& path, data_representation_t repres);

	//! Allocates storage for data on the host and device memory
	void allocate_storage();

	void store_event_data(event_name_t name, ttt_t t, var_t d, unsigned int idx1, unsigned int idx2, event_data_t *e);

	//! Sets the grid and block for the kernel launch
	void set_kernel_launch_param(unsigned int n_data);

public:
	//! Run a gravitational benchmark on the GPU and determines the number of threads at which the computation is the fastest.
	/*!
		\return The optimal number of threads per block
	*/
	unsigned int benchmark();
	float benchmark_calc_grav_accel(ttt_t curr_t, unsigned int n_sink, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a);

	void gpu_calc_grav_accel(ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy);
	void gpu_calc_drag_accel(ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy);

	void cpu_calc_grav_accel(ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy);
	void cpu_calc_grav_accel_SI( ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a);
	void cpu_calc_grav_accel_SI( ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a, event_data_t* events, unsigned int *event_counter);
	void cpu_calc_grav_accel_NI( ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a);
	void cpu_calc_grav_accel_NI( ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a, event_data_t* events, unsigned int *event_counter);
	void cpu_calc_grav_accel_NSI(ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a);
	void cpu_calc_grav_accel_NSI(ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a, event_data_t* events, unsigned int *event_counter);

	void cpu_calc_drag_accel(ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy);
	void cpu_calc_drag_accel_NSI(ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a);

	//! Test function: prints the data stored in sim_data 
	/*!
		\param comp_dev If CPU than prints the data stored in sim_data on the HOST if GPU than on the device
	*/
	void print_sim_data(computing_device_t comp_dev);

private:
	
	unsigned int id_dev;            //!< The id of the GPU
	computing_device_t comp_dev;    //!< The computing device to carry out the calculations (cpu or gpu)
	collision_detection_model_t cdm;//! Collision detection model

	unsigned int n_tpb;             //!< The number of thread per block to use for kernel launches
	bool    continue_simulation;    //!< Continues a simulation from its last saved output

	dim3	grid;
	dim3	block;

	var_t threshold[THRESHOLD_N];   //! Contains the threshold values: hit_centrum_dst, ejection_dst, collision_factor

	unsigned int	event_counter;	//! Number of events occured during the last check
	unsigned int* d_event_counter;	//! Number of events occured during the last check (stored on the devive)

	event_data_t* events;			//!< Vector on the host containing data for events (one colliding pair multiple occurances)
	event_data_t* d_events;			//!< Vector on the device containing data for events (one colliding pair multiple occurances)
	vector<event_data_t> sp_events;	//!< Vector on the host containing data for events but  (one colliding pair one occurances)

	vector<string> body_names;
};
