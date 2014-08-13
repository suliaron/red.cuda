#pragma once

// includes system
#include <string>
#include <vector>

// includes project
#include "gas_disk.h"
#include "number_of_bodies.h"
#include "red_type.h"

using namespace std;

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

typedef enum orbelem_name
	{
		ORBELEM_SMA,
		ORBELEM_ECC,
		ORBELEM_INC,
		ORBELEM_PERI,
		ORBELEM_NODE,
		ORBELEM_MEAN
	} orbelem_name_t;

typedef enum phys_prop_name
	{
		MASS,
		RADIUS,
		DENSITY,
		DRAG_COEFF
	} phys_prop_name_t;

typedef enum event_name
	{
		EVENT_NAME_NONE,
		EVENT_NAME_HIT_CENTRUM,
		EVENT_NAME_EJECTION,
		EVENT_NAME_CLOSE_ENCOUNTER,
		EVENT_NAME_COLLISION,
		EVENT_NAME_N
	} event_name_t;

class pp_disk
{
public:
	pp_disk(string& path, gas_disk *gd);
	~pp_disk();

	//! Copies ODE parameters and variables from the host to the cuda device
	void copy_to_device();
	//! Copies ODE parameters and variables from the cuda device to the host
	void copy_to_host();
	//! Copies ODE variables from the cuda device to the host
	void copy_variables_to_host();
	//! Returns the mass of the central star
	var_t get_mass_of_star();
	//! Transforms the system to barycentric reference frame
	void transform_to_bc();
	//! Transforms the system to a reference frame with the reference body in the origin
	/*
		\param refBodyId the id o
	*/
	void call_kernel_transform_to(int refBodyId);
	//! Print the data of all bodies
	/*   
		\param sout print the data to this stream
	*/
	void print_body_data(ostream& sout);

	// Input/Output streams
	friend ostream& operator<<(ostream& stream, const number_of_bodies* n_bodies);

	//! Calculates the differentials of variables
	/*!
		This function is called by the integrator when calculation of the differentials is necessary
		\param i Order of the variables to calculate (e.g. 0: velocities, 1: acceleration ...)
		\param r Number of substep, used with higher order methods, like Runge-Kutta
		\param curr_t Time
		\param pos Device vector with position variables
		\param vel Device vector with velocity variables
		\param dy Device vector that will hold the differentials
	*/
	void calculate_dy(int i, int r, ttt_t curr_t, const posm_t *pos, const velR_t *vel, vec_t* dy);

	// Test function: print out all the simulation data contained on the device
	void test_call_kernel_print_sim_data();

	number_of_bodies	*n_bodies;
	sim_data_t	*sim_data;		/*!< aggregate containing all the data of the simulation */

	gas_disk	*g_disk;
	gas_disk	*d_g_disk;

	ttt_t		t;

private:
	//! Loads the initial position and velocity of the bodies (second input version).
	/*   
		\param path the full path of the data file
	*/
	void load(string& path);
	void get_number_of_bodies(string& path);
	void allocate_storage();

	//! Computes the total mass of the system
	var_t get_total_mass();
	//! Compute the position and velocity of the system's barycenter
	/*  
		\param R0 will contain the position of the barycenter
		\param V0 will contain the velocity of the barycenter
	*/
	void compute_bc(posm_t* R0, velR_t* V0);

	//! Sets the grid and block for the kernel launch
	void set_kernel_launch_param(int n_data);
	void call_kernel_calculate_grav_accel(ttt_t curr_t, const posm_t* pos, const velR_t* vel, vec_t* dy);

	dim3	grid;
	dim3	block;

	vector<string>		body_names;
};