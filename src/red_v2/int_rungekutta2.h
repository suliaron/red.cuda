#pragma once

#include "integrator.h"

#include "red_type.h"

class ode;

class int_rungekutta2 : public integrator
{
public:
	static var_t a[];
	static var_t b[];
	static ttt_t c[];

	int_rungekutta2(ode& f, ttt_t dt, computing_device_t comp_dev);
	~int_rungekutta2();

	ttt_t step();

private:
	void cpu_sum_vector(var_t* a, const var_t* b, var_t F, const var_t* c, uint32_t n);
	void calc_ytemp(uint16_t stage);
	void calc_y_np1();
};
