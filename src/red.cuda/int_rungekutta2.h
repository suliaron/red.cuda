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

class rungekutta2
{
public:
	static var_t a[];
	static var_t b[];
	static ttt_t c[];

public:
	rungekutta2(pp_disk *ppd, ttt_t dt);
	~rungekutta2();
	ttt_t step();

	ttt_t dt_try;
	ttt_t dt_did;
	ttt_t dt_next;
	string name;
private:
	void calc_grid(int nData, int threads_per_block);
	void allocate_device_vector(void **d_ptr, size_t size);

	void call_kernel_calc_ytemp_for_fr(int r);
	void call_kernel_calc_y_np1();

	//! The order of the embedded RK formulae
	int	RKOrder;

	//! Holds the derivatives for the differential equations
	vector<vector <vec_t*> >	d_f;
	//! Holds the temporary solution approximation along the step
	vector<vec_t*>				d_ytemp;

	dim3	grid;
	dim3	block;

	ttt_t	t;					/*!< Actual time of the integrator */
	pp_disk*	ppd;			/*!< simulation data */					
};

} /* namespace integrator */
