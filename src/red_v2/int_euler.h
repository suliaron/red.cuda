#pragma once

#include "integrator.h"

#include "red_type.h"

class ode;

class euler : public integrator
{
public:
	euler(ode& f, ttt_t dt, computing_device_t comp_dev);
	~euler();

	ttt_t step();

private:
	void cpu_sum_vector(var_t* a, const var_t* b, var_t F, const var_t* c, uint32_t n);
	void calc_y_np1();
};
