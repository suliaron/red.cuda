#pragma once

#include "integrator.h"
#include "pp_disk.h"

#include "red_type.h"

class rungekutta2 : public integrator
{
public:
	static var_t a[];
	static var_t b[];
	static ttt_t c[];

	rungekutta2(pp_disk *ppd, ttt_t dt, computing_device_t comp_dev);
	~rungekutta2();

	ttt_t step();

private:
	void cpu_sum_vector(int n, const var_t* a, const var_t* b, var_t b_factor, var_t* result);
	void calc_ytemp_for_fr(uint32_t n_var, int r);
	void calc_y_np1(uint32_t n_var);
};
