#pragma once
// includes system
#include <string>
#include <vector>

// includes project
#include "integrator.h"
#include "pp_disk.h"
#include "red_type.h"

using namespace std;

class rungekutta2 : public integrator
{
public:
	static var_t a[];
	static var_t b[];
	static ttt_t c[];

	rungekutta2(pp_disk *ppd, ttt_t dt);
	~rungekutta2();

	ttt_t step();

private:
	void call_kernel_calc_ytemp_for_fr(int r);
	void call_kernel_calc_y_np1();

	//! The order of the embedded RK formulae
	int	RKOrder;

	//! Holds the derivatives for the differential equations
	vector<vector <vec_t*> >	d_f;
};
