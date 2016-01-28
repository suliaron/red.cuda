#pragma once

#include <string>
#include <vector>

#include "red_type.h"

class ode;

class integrator
{
public:
	integrator(ode& f, ttt_t dt, bool adaptive, var_t tolerance, uint16_t n_stage, computing_device_t comp_dev);
	~integrator();

	//! Set the computing device to calculate the integration step
	/*
		\param device specifies which device will execute the computations
	*/
//	void set_computing_device(computing_device_t device);
//	computing_device_t get_computing_device() { return comp_dev; }

	void update_counters(uint16_t iter);

	uint64_t get_n_tried_step()    { return n_tried_step;  }
	uint64_t get_n_passed_step()   { return n_passed_step; }
	uint64_t get_n_failed_step()   { return n_failed_step; }

	void set_max_iter(uint16_t n)  { max_iter = n;         }
	uint16_t get_max_iter()        { return max_iter;      }

	void set_dt_min(var_t dt)      { dt_min = dt;          }
	var_t get_dt_min()             { return dt_min;        }

	virtual ttt_t step() = 0;

	ode& f;

	bool error_check_for_tp;	//!< Check the error also for the test particles
	std::string name;

protected:
//	void calc_grid(int nData, int threads_per_block);
	var_t get_max_error(uint32_t n_var);

	computing_device_t comp_dev;        //!< The computing device to carry out the calculations (cpu or gpu)

	dim3 grid;
	dim3 block;

	ttt_t t;					        //!< Actual time of the integrator
	ttt_t dt_try;                       //!< The size of the step to try (based on the previous successfull step dt_did)
	ttt_t dt_did;                       //!< The size of the previous successfull step

	uint64_t n_tried_step;
	uint64_t n_passed_step;
	uint64_t n_failed_step;                                                                                                                                                                                                

	bool adaptive;                      //!< True if the method estimates the error and accordingly adjusts the step-size	
	var_t tolerance;                    //!< The maximum of the allowed local truncation error
	uint16_t n_order;                   //!< The order of the embedded RK formulae
	uint16_t n_stage;                   //!< The number of the method's stages

	uint32_t n_var;                     //! The total number of variables of the problem
 
	std::vector<var_t *> h_k;           //!< Differentials in the HOST memory
	std::vector<var_t *> d_k;           //!< Differentials in the DEVICE memory

	var_t* h_ytemp;	                    //!< Holds the temporary solution approximation along the step in the HOST memory
	var_t* d_ytemp;	                    //!< Holds the temporary solution approximation along the step in the DEVICE memory

	var_t* h_err;	                    //!< Holds the leading local truncation error for each variable in HOST memory
	var_t* d_err;	                    //!< Holds the leading local truncation error for each variable in DEVICE memory

	uint16_t max_iter;
	var_t dt_min;
private:
	void initialize();
//	void create_aliases();

	//! Allocates storage for data on the host and device memory
	void allocate_storage(       uint32_t n_var);
	void allocate_host_storage(  uint32_t n_var);
	void allocate_device_storage(uint32_t n_var);

	void deallocate_storage();
	void deallocate_host_storage();
	void deallocate_device_storage();
};
