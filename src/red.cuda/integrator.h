#pragma once

#include <string>
#include <vector>

#include "pp_disk.h"

#include "red_type.h"

class integrator
{
public:
	integrator(pp_disk *ppd, ttt_t dt, bool adaptive, var_t tolerance, int n_stage, computing_device_t comp_dev);
	~integrator();

	//! Set the computing device to calculate the integration step
	/*
		\param device specifies which device will execute the computations
	*/
	void set_computing_device(computing_device_t device);
	computing_device_t get_computing_device() { return comp_dev; }

	void update_counters(int iter);
	uint64_t get_n_failed_step();
	uint64_t get_n_passed_step();
	uint64_t get_n_tried_step();

	virtual ttt_t step() = 0;

	bool error_check_for_tp;	//!< Check the error also for the test particles
	std::string name;

protected:
	void calc_grid(int nData, int threads_per_block);
	var_t get_max_error(uint32_t n_var);

	pp_disk* ppd;

	computing_device_t comp_dev;        //!< The computing device to carry out the calculations (cpu or gpu)

	dim3 grid;
	dim3 block;

	ttt_t t;					        //!< Actual time of the integrator
	ttt_t dt_try;                       //!< The size of the step to try (based on the previous successfull step dt_did)
	ttt_t dt_did;                       //!< The size of the previous successfull step
	ttt_t dt_next;                      //!< The size of the next step to try (based on the previous successfull step dt_did)

	uint64_t n_tried_step;
	uint64_t n_passed_step;
	uint64_t n_failed_step;

	int	order;                          //!< The order of the embedded RK formulae
	int	n_stage;                          //!< The maximum number of the force calculation
	bool adaptive;                      //!< True if the method estimates the error and accordingly adjusts the step-size	
	var_t tolerance;                    //!< The maximum of the allowed local truncation error

	std::vector<std::vector <var4_t*> > h_dydx;    //!< Differentials in the HOST memory
	std::vector<std::vector <var4_t*> > d_dydx;    //!< Differentials in the DEVICE memory
	std::vector<std::vector <var4_t*> > dydx;      //!< Alias to the differentials (either in the HOST or the DEVICE memory)

	std::vector<var4_t*> h_ytemp;	            //!< Holds the temporary solution approximation along the step in the HOST memory
	std::vector<var4_t*> d_ytemp;	            //!< Holds the temporary solution approximation along the step in the DEVICE memory
	std::vector<var4_t*> ytemp;	            //!< Alias either to h_ytemp or d_ytemp depending on the executing processing unit

	std::vector<var_t*> h_err;	            //!< Holds the leading local truncation error for each variable in HOST memory
	std::vector<var_t*> d_err;	            //!< Holds the leading local truncation error for each variable in DEVICE memory
	std::vector<var_t*> err;                 //!< Alias to the leading local truncation error (either in the HOST or the DEVICE memory)

private:
	void initialize();
	void create_aliases();

	//! Allocates storage for data on the host and device memory
	void allocate_storage();
	void allocate_host_storage(int n_body);
	void allocate_device_storage(int n_body);

	void deallocate_host_storage();
	void deallocate_device_storage();
};
