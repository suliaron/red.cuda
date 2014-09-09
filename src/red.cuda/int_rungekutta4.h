#pragma once

// includes system
#include <string>
#include <vector>

// includes project
#include "pp_disk.h"
#include "red_type.h"

using namespace std;

namespace integrator
{

class rungekutta4
{
public:
	static var_t a[];
	static var_t b[];
	static var_t bh[];
	static ttt_t c[];

public:
	rungekutta4(ttt_t t0, ttt_t dt, bool adaptive, var_t tolerance, pp_disk *ppd);
	~rungekutta4();
	ttt_t step();

	ttt_t dt_try;
	ttt_t dt_did;
	ttt_t dt_next;
	string name;
private:
	void calc_grid(int nData, int threads_per_block);
	void call_kernel_calc_ytemp_for_fr(int r);
	void call_kernel_calc_yHat();

	//! The order of the embedded RK formulae
	int	RKOrder;
	//! The maximum number of the force calculation
	int	r_max;
	//! True if the method estimates the error and accordingly adjusts the step-size
	bool adaptive;
	//! The maximum of the allowed local truncation error
	var_t tolerance;

	//! Holds the derivatives for the differential equations
	vector<vector <vec_t*> >	d_f;
	//! Holds the temporary solution approximation along the step
	vector<vec_t*>				d_ytemp;
	//! Holds the leading local truncation error for each variable
	vector<vec_t*>				d_err;

	dim3	grid;
	dim3	block;

	ttt_t	t;					/*!< Actual time of the integrator */
	pp_disk*	ppd;			/*!< simulation data */					
};

} /* namespace integrator */
