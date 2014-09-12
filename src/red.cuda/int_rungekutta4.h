#pragma once
// includes system
#include <string>
#include <vector>

// includes project
#include "integrator.h"
#include "pp_disk.h"
#include "red_type.h"

using namespace std;

class rungekutta4 : public integrator
{
public:
	static var_t a[];
	static var_t b[];
	static var_t bh[];
	static ttt_t c[];

	rungekutta4(pp_disk *ppd, ttt_t dt, bool adaptive, var_t tolerance);
	~rungekutta4();

	ttt_t step();

private:
	void call_kernel_calc_ytemp_for_fr(int r);
	void call_kernel_calc_yHat();
	void call_kernel_calc_error();

	var_t get_max_error(int n_var);

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
	//! Holds the leading local truncation error for each variable
	vector<var_t*>				d_err;
};
