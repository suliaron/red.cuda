#pragma once

#include "integrator.h"

#include "red_type.h"

class ode;

class int_rungekutta4 : public integrator
{
public:
	static var_t a[];
	static var_t b[];
	static var_t bh[];
	static ttt_t c[];
	static uint16_t a_idx[];

	int_rungekutta4(ode& f, ttt_t dt, bool adaptive, var_t tolerance, computing_device_t comp_dev);
	~int_rungekutta4();

	ttt_t step();

private:
	void calc_lin_comb(var_t* y, const var_t* y_n, const var_t* coeff, uint16_t n_coeff, uint32_t n_var);
	//void calc_ytemp(uint16_t stage);
	//void calc_y_np1();
	void calc_error(uint32_t n);

	//void cpu_sum_vector(var_t* a, const var_t* b, var_t F, const var_t* c, uint32_t n);
	void cpu_calc_lin_comb(var_t* y, const var_t* y_n, const var_t* coeff, uint16_t n_coeff, uint32_t n_var);
	//void cpu_calc_ytemp(uint16_t stage);
	//void cpu_calc_y_np1();
	void cpu_calc_error(uint32_t n);
};
