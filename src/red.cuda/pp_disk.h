#pragma once

#include <string>
#include <vector>

#include "analytic_gas_disk.h"
#include "fargo_gas_disk.h"
#include "red_type.h"

class pp_disk
{
public:
	pp_disk(n_objects_t *n_bodies, gas_disk_model_t g_disk_model, collision_detection_model_t cdm, uint32_t id_dev, computing_device_t comp_dev);
	pp_disk(std::string& path_data, std::string& path_data_info, gas_disk_model_t g_disk_model, collision_detection_model_t cdm, uint32_t id_dev, computing_device_t comp_dev, const var_t* thrshld);
	~pp_disk();

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

	void set_n_tpb(uint32_t n)   { n_tpb = n;     }
	uint32_t  get_n_tpb(void)    { return n_tpb;  }

	void set_id_dev(uint32_t id) { id_dev = id;   }
	uint32_t  get_id_dev(void)   { return id_dev; }

	//! Determines the mass of the central star
	/*!
		\return The mass of the central star
	*/
	var_t get_mass_of_star();
	//! Returns the number of events during the simulation
	uint32_t get_n_total_event();
	//! Clears the event_counter (sets to 0)
	void clear_event_counter();
	//! Sets all event counter to a specific value
	/*!
		\param field Sets this field
		\param value The value to set the field
	*/
	void set_event_counter(event_counter_name_t field, uint32_t value);

	//! Print the data of all bodies
	/*!
		\param path   print the data to this file
		\param repres indicates the data representation of the file, i.e. text or binary
	*/
	void print_data(std::string& path, data_rep_t repres);
	//! Print the data of all bodies
	/*!
		\param path   print the data to this file
		\param repres indicates the data representation of the file, i.e. text or binary
	*/
	void print_data_info(std::string& path, data_rep_t repres);

	//! Print the event data
	/*!
		\param path print the data to this file
		\param log_f print the data to this stream
	*/
	void print_event_data(std::string& path, ofstream& log_f);
	//! Print the classical integrals
	/*!
		\param sout print the data to this stream
	*/
	void print_integral_data(std::string& path, pp_disk_t::integral_t& I);

	//! From the events it will create a vector containing one entry for each colliding pair with the earliest collision time
	void populate_sp_events();

	void handle_collision();
	void handle_ejection_hit_centrum();
	void handle_collision_pair(uint32_t i, pp_disk_t::event_data_t *collision);

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
	void calc_dydx(uint32_t i, uint32_t rr, ttt_t curr_t, const var4_t *r, const var4_t *v, var4_t* dy);

	//! Calculates the classical integrals of the N-body system
	/*!
		This function is called by the main while cycle in order to determine the integrals in 
		\param cpy_to_HOST if true than the data is copied first from the DEVICE to the HOST
		\param integrals array that will hold the integrals
	*/
	void calc_integral(bool cpy_to_HOST, pp_disk_t::integral_t& integrals);

	//! Swaps the yout with the y variable, i.e. at the end of an integration step the output will be the input of the next step
	void swap();

public:
	//! Run a gravitational benchmark on the GPU and determines the number of threads at which the computation is the fastest.
	/*!
		\return The optimal number of threads per block
	*/
	uint32_t benchmark();
	float benchmark_calc_grav_accel(ttt_t curr_t, uint32_t n_sink, interaction_bound int_bound, const pp_disk_t::body_metadata_t* body_md, const pp_disk_t::param_t* p, const var4_t* r, const var4_t* v, var4_t* a);

	void gpu_calc_grav_accel(ttt_t curr_t, const var4_t* r, const var4_t* v, var4_t* dy);
	void gpu_calc_drag_accel(ttt_t curr_t, const var4_t* r, const var4_t* v, var4_t* dy);

	void cpu_calc_grav_accel(ttt_t curr_t, const var4_t* r, const var4_t* v, var4_t* dy);
	void cpu_calc_grav_accel_SI( ttt_t curr_t, interaction_bound int_bound, const pp_disk_t::body_metadata_t* body_md, const pp_disk_t::param_t* p, const var4_t* r, const var4_t* v, var4_t* a);
	void cpu_calc_grav_accel_SI( ttt_t curr_t, interaction_bound int_bound, const pp_disk_t::body_metadata_t* body_md, const pp_disk_t::param_t* p, const var4_t* r, const var4_t* v, var4_t* a, pp_disk_t::event_data_t* events, uint32_t *event_counter);
	void cpu_calc_grav_accel_NI( ttt_t curr_t, interaction_bound int_bound, const pp_disk_t::body_metadata_t* body_md, const pp_disk_t::param_t* p, const var4_t* r, const var4_t* v, var4_t* a);
	void cpu_calc_grav_accel_NI( ttt_t curr_t, interaction_bound int_bound, const pp_disk_t::body_metadata_t* body_md, const pp_disk_t::param_t* p, const var4_t* r, const var4_t* v, var4_t* a, pp_disk_t::event_data_t* events, uint32_t *event_counter);
	void cpu_calc_grav_accel_NSI(ttt_t curr_t, interaction_bound int_bound, const pp_disk_t::body_metadata_t* body_md, const pp_disk_t::param_t* p, const var4_t* r, const var4_t* v, var4_t* a);
	void cpu_calc_grav_accel_NSI(ttt_t curr_t, interaction_bound int_bound, const pp_disk_t::body_metadata_t* body_md, const pp_disk_t::param_t* p, const var4_t* r, const var4_t* v, var4_t* a, pp_disk_t::event_data_t* events, uint32_t *event_counter);

	void cpu_calc_drag_accel(ttt_t curr_t, const var4_t* r, const var4_t* v, var4_t* dy);
	void cpu_calc_drag_accel_NSI(ttt_t curr_t, interaction_bound int_bound, const pp_disk_t::body_metadata_t* body_md, const pp_disk_t::param_t* p, const var4_t* r, const var4_t* v, var4_t* a);

	//! Test function: prints the data stored in sim_data 
	/*!
		\param comp_dev If CPU than prints the data stored in sim_data on the HOST if GPU than on the device
	*/
	void print_sim_data(computing_device_t comp_dev);

	ttt_t t;                                   //!< time when the variables are valid
	ttt_t dt;                                  //!< timestep for the integrator
	n_objects_t* n_bodies;                     //!< struct containing the number of different bodies
	pp_disk_t::sim_data_t* sim_data;           //!< struct containing all the data of the simulation

	gas_disk_model_t g_disk_model;
	analytic_gas_disk* a_gd;
	fargo_gas_disk* f_gd;

	uint32_t n_ejection[   EVENT_COUNTER_NAME_N];   //!< Number of ejection
	uint32_t n_hit_centrum[EVENT_COUNTER_NAME_N];   //!< Number of hit centrum
	uint32_t n_collision[  EVENT_COUNTER_NAME_N];   //!< Number of collision
	uint32_t n_event[      EVENT_COUNTER_NAME_N];   //!< Number of total events

private:
	//! Initialize the members to default values
	void initialize();

	void increment_event_counter(uint32_t *event_counter);

	void load_data_info(std::string& path, data_rep_t repres);
	void load_data(std::string& path_data, data_rep_t repres);
	
	//! Allocates storage for data on the host and device memory
	void allocate_storage();

	//! Sets the grid and block for the kernel launch
	void set_kernel_launch_param(uint32_t n_data);
	//! Transforms the system to barycentric reference frame
	void transform_to_bc();
	//! Transform the time using the new time unit: 1/k = 58.13244 ...
	void transform_time();
	//! Transform the velocity using the new time unit: 1/k = 58.13244 ...
	void transform_velocity();

	uint32_t id_dev;                      //!< The id of the GPU
	computing_device_t comp_dev;          //!< The computing device to carry out the calculations (cpu or gpu)
	collision_detection_model_t cdm;      //! Collision detection model
									      
	uint32_t n_tpb;                       //!< The number of thread per block to use for kernel launches
									      
	dim3	grid;					      
	dim3	block;					      
									      
	var_t threshold[THRESHOLD_N];         //! Contains the threshold values: hit_centrum_dst, ejection_dst, collision_factor
									      
	uint32_t	event_counter;	          //! Number of events occured during the last check
	uint32_t* d_event_counter;	          //! Number of events occured during the last check (stored on the devive)
									      
	pp_disk_t::event_data_t* events;			      //!< Vector on the host containing data for events (one colliding pair multiple occurances)
	pp_disk_t::event_data_t* d_events;			      //!< Vector on the device containing data for events (one colliding pair multiple occurances)
	vector<pp_disk_t::event_data_t> sp_events;	      //!< Vector on the host containing data for events but  (one colliding pair one occurances)

	vector<std::string> body_names;
};
