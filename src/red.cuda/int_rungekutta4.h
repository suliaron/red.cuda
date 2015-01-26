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

	rungekutta4(pp_disk *ppd, ttt_t dt, bool adaptive, var_t tolerance, bool cpu);
	~rungekutta4();

	ttt_t step();

private:
	void cpu_sum_vector(int n, const var_t* a, const var_t* b, var_t b_factor, var_t* result);
	void cpu_calc_y_np1(int n, const var_t *y_n, const var_t *f1, const var_t *f2, const var_t *f3, const var_t *f4, var_t b0, var_t b1, var_t *y_np1);
	void cpu_calc_error(int n, const var_t *f4, const var_t* f5, var_t *result);

	//void call_kernel_calc_ytemp_for_fr(int n_var, int r);
	//void call_kernel_calc_y_np1(int n_var);
	//void call_kernel_calc_error(int n_var);

	void calc_ytemp_for_fr(int n_var, int r);
	void calc_y_np1(int n_var);
	void calc_error(int n_var);

	var_t get_max_error(int n_var);

	int	RKOrder;		//!< The order of the embedded RK formulae
	int	r_max;			//!< The maximum number of the force calculation
	bool adaptive;		//!< True if the method estimates the error and accordingly adjusts the step-size	
	var_t tolerance;	//!< The maximum of the allowed local truncation error

	vector<vector <vec_t*> >	dydx;	//!< Holds the derivatives for the differential equations
	vector<var_t*>				err;	//!< Holds the leading local truncation error for each variable
};
