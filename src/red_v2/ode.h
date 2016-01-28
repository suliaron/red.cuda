#pragma once

#include <vector>

#include "vector_types.h"

#include "red_type.h"

class ode
{
public:
	ode(uint16_t n_dim, uint16_t n_vpo, uint16_t n_ppo, uint32_t n_obj, computing_device_t comp_dev);
	~ode();

	void copy_vars(copy_direction_t dir);
	void copy_params(copy_direction_t dir);

	void swap();

	virtual void calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy) = 0;

	void initialize();

	void allocate_storage(       uint32_t n_var, uint32_t n_par);
	void allocate_host_storage(  uint32_t n_var, uint32_t n_par);
	void allocate_device_storage(uint32_t n_var, uint32_t n_par);

	void deallocate_storage();
	void deallocate_host_storage();
	void deallocate_device_storage();

	vector<std::string> obj_names;

	ttt_t t;              //! Current time
	ttt_t tout;           //! Time at the end of the integration step

	var_t* h_y;           //! Host vector (size of n_var) of ODE variables at t
	var_t* h_yout;        //! Host vector (size of n_var) of ODE variables at tout
	var_t* d_y;           //! Device vector (size of n_var) of ODE variables at t
	var_t* d_yout;        //! Device vector (size of n_var) of ODE variables at tout
	var_t* y;             //! Alias to either to Host or Device vector of ODE variables at t depeding on the execution device
	var_t* yout;          //! Alias to either to Host or Device vector of ODE variables at tout depeding on the execution device

	var_t* h_p;           //! Host vector (size of n_obj * n_ppo) of parameters
	var_t* d_p;           //! Device vector (size of n_obj * n_ppo) of parameters
	var_t* p;             //! Alias to either to Host or Device vector of parameters depeding on the execution device

	uint32_t n_obj;       //! The total number of objets in the problem
	uint16_t n_dim;       //! The space dimension of the problem 

	uint16_t n_vpo;       //! The number of variables per object (vpo)
	uint16_t n_ppo;       //! The number of parameters per object (ppo)

	uint32_t n_var;       //! The total number of variables of the problem
	uint32_t n_par;       //! The total number of parameters of the problem

	computing_device_t comp_dev;

	dim3 grid;            //! Defines the grid of the blocks of the current execution
	dim3 block;           //! Defines the block of the threads of the current execution
	uint16_t n_tpb;       //! Holds the number of threads per block
};
