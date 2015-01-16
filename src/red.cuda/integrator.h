#pragma once

// includes system
#include <string>
#include <vector>

// includes CUDA

// includes Thrust

// includes project
#include "pp_disk.h"
#include "red_type.h"

using namespace std;

class integrator
{
public:
	integrator(pp_disk *ppd, ttt_t dt, bool cpu);
	~integrator();

	void update_counters(int iter);
	int get_n_failed_step();
	int get_n_passed_step();
	int get_n_tried_step();

	virtual ttt_t step() = 0;

	bool_t error_check_for_tp;	//!< Check the error also for the test particles

	string name;
	string short_name;
protected:
	void	calc_grid(int nData, int threads_per_block);

	pp_disk*	ppd;

	bool	cpu;				//!< If true than execute the code on the cpu

	dim3	grid;
	dim3	block;

	ttt_t t;					//!< Actual time of the integrator
	ttt_t dt_try;
	ttt_t dt_did;
	ttt_t dt_next;

	int n_failed_step;
	int n_passed_step;
	int n_tried_step;

	vector<vec_t*>	d_ytemp;	//!< On the DEVICE holds the temporary solution approximation along the step
	vector<vec_t*>	h_ytemp;	//!< On the HOST holds the temporary solution approximation along the step
	
	vector<vec_t*>	ytemp;	    //!< Holds the temporary solution approximation along the step (either in the host or device memory)
};
