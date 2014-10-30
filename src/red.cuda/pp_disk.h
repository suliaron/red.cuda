#pragma once

// includes system
#include <string>
#include <vector>

// includes project
#include "gas_disk.h"
#include "number_of_bodies.h"
#include "red_type.h"

using namespace std;

class pp_disk
{
public:

	typedef struct survivor
		{
			vec_t			r;
			vec_t			v;
			param_t			p;
			body_metadata_t body_md;
		} survivor_t;

	pp_disk(string& path, gas_disk *gd, int n_tpb, bool use_padded_storage);
	~pp_disk();

	//! Copies ODE parameters and variables from the host to the cuda device
	void copy_to_device();
	//! Copies ODE parameters and variables from the cuda device to the host
	void copy_to_host();
	//! Copies threshold data to the device constant memory
	void copy_threshold_to_device(const var_t* threshold);
	//! Copies the event data from the cuda device to the host
	void copy_event_data_to_host();

	//! Returns the mass of the central star
	var_t get_mass_of_star();
	//! Transforms the system to barycentric reference frame
	void transform_to_bc();
	//! Transform the velocity using the new time unit: 1/k = 58.13244 ...
	void transform_time();
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

	//! Returns the number of events during the last step
	int get_n_event();

	//! Returns the number of events during the simulation
	int get_n_total_event();

	//! Clears the event_counter (sets to 0)
	void clear_event_counter();

	//! From the events it will create a vector containing one entry for each colliding pair with the earliest collision time
	void create_sp_events();

	int handle_collision();
	int2_t handle_ejection_hit_centrum();
	void handle_collision_pair(int i, event_data_t *collision);
	void calc_phase_after_collision(var_t m1, var_t m2, const vec_t* r1, const vec_t* v1, const vec_t* r2, const vec_t* v2, vec_t& r0, vec_t& v0);

	//! Check all bodies against ejection and hit centrum criterium
	int call_kernel_check_for_ejection_hit_centrum();
	//! Test function: print out all the simulation data contained on the device
	void test_call_kernel_print_sim_data();
	//! Transforms the system to a reference frame with the reference body in the origin
	/*
		\param refBodyId the id o
	*/
	void call_kernel_transform_to(int refBodyId);
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
	void calc_dy(int i, int rr, ttt_t curr_t, const vec_t *r, const vec_t *v, vec_t* dy);

	void remove_inactive_bodies();

	// Input/Output streams
	friend ostream& operator<<(ostream& stream, const number_of_bodies* n_bodies);

	number_of_bodies	*n_bodies;
	sim_data_t	*sim_data;		/*!< struct containing all the data of the simulation */

	gas_disk	*g_disk;
	gas_disk	*d_g_disk;

	ttt_t		t;

	int	n_ejection;					//! Total number of ejections
	int n_hit_centrum;				//! Total number of hit centrum
	int n_collision;				//! Total number of collisions

private:
	//! Loads the initial position and velocity of the bodies (second input version).
	/*   
		\param path the full path of the data file
	*/
	void load(string& path);
	number_of_bodies* get_number_of_bodies(string& path);

	//! Allocates storage for data on the host and device memory
	void allocate_storage();
	void allocate_host_storage(sim_data_t *sd, int n);
	void allocate_device_storage(sim_data_t *sd, int n);

	void deallocate_host_storage(sim_data_t *sd);
	void deallocate_device_storage(sim_data_t *sd);

	//! Computes the total mass of the system
	var_t get_total_mass();
	//! Compute the position and velocity of the system's barycenter
	/*  
		\param R0 will contain the position of the barycenter
		\param V0 will contain the velocity of the barycenter
	*/
	void compute_bc(vec_t* R0, vec_t* V0);

	//! Sets the grid and block for the kernel launch
	void set_kernel_launch_param(int n_data);
	void call_kernel_calc_grav_accel(ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy);

	//! Calculates the size for the padded storages
	//size_t get_padded_storage_size(size_t size, int n_tpb);

	void create_padding_particle(int k, ttt_t* epoch, body_metadata_t* body_md, param_t* p, vec_t* r, vec_t* v, bool &b);

	int		n_tpb;					//!< The number of thread per block to use for kernel launches
	bool	use_padded_storage;		//!< If true use the padded storage

	dim3	grid;
	dim3	block;

	int	event_counter;				//! Number of events occured during the last check
	int* d_event_counter;			//! Number of events occured during the last check (stored on the devive)

	event_data_t* events;			//!< Vector on the host containing data for events (one colliding pair multiple occurances)
	event_data_t* d_events;			//!< Vector on the device containing data for events (one colliding pair multiple occurances)
	vector<event_data_t> sp_events;	//!< Vector on the host containing data for events but  (one colliding pair one occurances)
	vector<survivor_t>	survivors;	//!< Vector on the host containing data for the survivor body

	vector<string>		body_names;
};
