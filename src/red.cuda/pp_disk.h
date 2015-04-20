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

	pp_disk(number_of_bodies *n_bodies, bool use_padded_storage, gas_disk_model_t g_disk_model, computing_device_t comp_dev);
	pp_disk(string& path, int n_tpb, bool use_padded_storage, gas_disk_model_t g_disk_model, computing_device_t comp_dev);
	~pp_disk();

	//! Initialize the members to default values
	void initialize();
	//! Copies ODE parameters and variables from the host to the cuda device
	void copy_to_device();
	//! Copies ODE parameters and variables from the cuda device to the host
	void copy_to_host();
	//! Copies threshold data (either into HOST or DEVICE memory depending on the cpu boolean value)
	void copy_threshold(const var_t* thrshld);
	//! Copies the event data from the cuda device to the host
	void copy_event_data_to_host();
	//! Copies the parameters of the analytic disk to the constant memory of the device
	//void copy_analytic_disk_to_device();
	//! Copies the parameters of the fargo disk to the constant memory of the device
	//void copy_fargo_disk_to_device();

	void copy_disk_params_to_device();

	//! Set the computing device to calculate the accelerations
	/*
		\param device specifies which device will execute the computations
	*/
	void set_computing_device(computing_device_t device);
	computing_device_t get_computing_device() { return comp_dev; }

	void set_n_tpb(int n) { n_tpb = n;    }
	int  get_n_tpb()      { return n_tpb; }

	//! Returns the mass of the central star
	var_t get_mass_of_star();
	//! Transforms the system to barycentric reference frame
	void transform_to_bc(bool verbose);
	//! Transform the time using the new time unit: 1/k = 58.13244 ...
	void transform_time(bool verbose);
	//! Transform the velocity using the new time unit: 1/k = 58.13244 ...
	void transform_velocity(bool verbose);
	//! Print the data of all bodies in text format
	/*   
		\param sout print the data to this stream
	*/
	void print_result_ascii(ostream& sout);
	//! Print the data of all bodies in binary format
	/*   
		\param sout print the data to this stream
	*/
	void print_result_binary(ostream& sout);
	//! Print the event data
	/*   
		\param sout print the data to this stream
		\param log_f print the data to this stream
	*/
	void print_event_data(ostream& sout, ostream& log_f);

	bool get_ups()		{ return use_padded_storage; }

	//! Returns the number of events during the last step
	int get_n_event();

	//! Returns the number of events during the simulation
	int get_n_total_event();

	//! Clears the event_counter (sets to 0)
	void clear_event_counter();
	//! Sets all event counter to a specific value
	/*
		\param field Sets this field
		\param value The value to set the field
	*/
	void set_event_counter(event_counter_name_t field, int value);

	//! From the events it will create a vector containing one entry for each colliding pair with the earliest collision time
	void create_sp_events();

	bool check_for_ejection_hit_centrum();
	bool check_for_collision();
	bool check_for_rebuild_vectors(int n);

	void handle_collision();
	void handle_ejection_hit_centrum();
	void handle_collision_pair(int i, event_data_t *collision);

	//! Check all bodies against ejection and hit centrum criterium
	int call_kernel_check_for_ejection_hit_centrum();
	int cpu_check_for_ejection_hit_centrum();
	//! Test function: print out all the simulation data contained on the device
	void test_call_kernel_print_sim_data();
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
	void calc_dydx(int i, int rr, ttt_t curr_t, const vec_t *r, const vec_t *v, vec_t* dy);

	//! Swaps the yout with the y variable, i.e. at the end of an integration step the output will be the input of the next step
	void swap();

	void remove_inactive_bodies();

	// Input/Output streams
	friend ostream& operator<<(ostream& stream, const number_of_bodies* n_bodies);

	ttt_t t;
	number_of_bodies* n_bodies;
	sim_data_t* sim_data;		/*!< struct containing all the data of the simulation */

	gas_disk_model_t g_disk_model;
	analytic_gas_disk* a_gd;
	fargo_gas_disk* f_gd;

	int n_ejection[   EVENT_COUNTER_NAME_N];   //!< Number of ejection
	int n_hit_centrum[EVENT_COUNTER_NAME_N];   //!< Number of hit centrum
	int n_collision[  EVENT_COUNTER_NAME_N];   //!< Number of collision
	int n_event[      EVENT_COUNTER_NAME_N];   //!< Number of total events

	actual_phase_storage_t aps;                //!< Shows which storage (y or yout) has the actual phases of the bodies

private:
	void increment_event_counter(int *event_counter);
	//! Loads the initial position and velocity of the bodies (second input version).
	/*   
		\param path the full path of the data file
	*/
	void load(string& path);
	void read_body_record(ifstream& input, int k, ttt_t* epoch, body_metadata_t* body_md, param_t* p, vec_t* r, vec_t* v);
	number_of_bodies* get_number_of_bodies(string& path);

	//! Allocates storage for data on the host and device memory
	void allocate_storage();

	void store_event_data(event_name_t name, ttt_t t, var_t d, int idx1, int idx2, event_data_t *e);

	//! Sets the grid and block for the kernel launch
	void set_kernel_launch_param(int n_data);

public:
	float wrapper_kernel_pp_disk_calc_grav_accel(ttt_t curr_t, int n_sink, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a);

	void call_kernel_calc_grav_accel(ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy);
	void call_kernel_calc_drag_accel(ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy);

	void cpu_calc_grav_accel(ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy);
	void cpu_calc_grav_accel_SI( ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a, event_data_t* events, int *event_counter);
	void cpu_calc_grav_accel_NI( ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a, event_data_t* events, int *event_counter);
	void cpu_calc_grav_accel_NSI(ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a, event_data_t* events, int *event_counter);

	void cpu_calc_drag_accel(ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy);
	void cpu_calc_drag_accel_NSI(ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a);
private:
	
	void create_padding_particle(int k, ttt_t* epoch, body_metadata_t* body_md, param_t* p, vec_t* r, vec_t* v);

	computing_device_t comp_dev;    //!< The computing device to carry out the calculations (cpu or gpu)

	int		n_tpb;					//!< The number of thread per block to use for kernel launches
	bool	use_padded_storage;		//!< If true use the padded storage scheme

	dim3	grid;
	dim3	block;

	var_t threshold[THRESHOLD_N];

	int	event_counter;				//! Number of events occured during the last check
	int* d_event_counter;			//! Number of events occured during the last check (stored on the devive)

	event_data_t* events;			//!< Vector on the host containing data for events (one colliding pair multiple occurances)
	event_data_t* d_events;			//!< Vector on the device containing data for events (one colliding pair multiple occurances)
	vector<event_data_t> sp_events;	//!< Vector on the host containing data for events but  (one colliding pair one occurances)

	vector<string> body_names;
};
