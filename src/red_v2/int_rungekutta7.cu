#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ode.h"
#include "int_rungekutta7.h"

#include "red_macro.h"
#include "redutilcu.h"

#define	LAMBDA	41.0/840.0

using namespace std;

// The Runge-Kutta matrix
var_t int_rungekutta7::a[] = 
{ 
/* 1 */     2.0/27.0,
/* 2 */     1.0/36.0,   1.0/12.0,
/* 3 */     1.0/24.0,        0.0,   1.0/8.0,
/* 4 */     5.0/12.0,        0.0, -25.0/16.0,   25.0/16.0,
/* 5 */     1.0/20.0,        0.0,        0.0,    1.0/4.0,      1.0/5.0,
/* 6 */   -25.0/108.0,       0.0,        0.0,  125.0/108.0,  -65.0/27.0,    125.0/54.0,
/* 7 */    31.0/300.0,       0.0,        0.0,          0.0,   61.0/225.0,    -2.0/9.0,    13.0/900.0,
/* 8 */     2.0,             0.0,        0.0,  -53.0/6.0,    704.0/45.0,   -107.0/9.0,    67.0/90.0,    3.0,
/* 9 */   -91.0/108.0,       0.0,        0.0,   23.0/108.0, -976.0/135.0,   311.0/54.0,  -19.0/60.0,   17.0/6.0,  -1.0/12.0,
/*10 */  2383.0/4100.0,      0.0,        0.0, -341.0/164.0, 4496.0/1025.0, -301.0/82.0, 2133.0/4100.0, 45.0/82.0, 45.0/164.0, 18.0/41.0,
/*11 */     3.0/205.0,       0.0,        0.0,          0.0,           0.0,   -6.0/41.0,   -3.0/205.0,  -3.0/41.0,  3.0/41.0,   6.0/41.0, 0.0,
/*12 */ -1777.0/4100.0,      0.0,        0.0, -341.0/164.0, 4496.0/1025.0, -289.0/82.0, 2193.0/4100.0, 51.0/82.0, 33.0/164.0, 12.0/41.0, 0.0, 1.0
};
// weights
var_t int_rungekutta7::b[]  = { 41.0/840.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0, 9.0/35.0, 9.0/280.0, 9.0/280.0, 41.0/840.0  };
var_t int_rungekutta7::bh[] = {        0.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0, 9.0/35.0, 9.0/280.0, 9.0/280.0, 41.0/840.0, 41.0/840.0, 41.0/840.0 };
// nodes
ttt_t int_rungekutta7::c[]  = { 0.0, 2.0/27.0, 1.0/9.0, 1.0/6.0, 5.0/12.0, 1.0/2.0, 5.0/6.0, 1.0/6.0, 2.0/3.0, 1.0/3.0, 1.0, 0.0, 1.0 };
// The starting index of the RK matrix for the stages
uint16_t int_rungekutta7::a_idx[] = {0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66};

namespace rk8_kernel
{
// a_i = b_i + F * c_i
static __global__
	void sum_vector(var_t* a, const var_t* b, var_t F, const var_t* c, uint32_t n)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		a[tid] = b[tid] + F * c[tid];
		tid += stride;
	}
}
} /* namespace rk8_kernel */

int_rungekutta7::int_rungekutta7(ode& f, ttt_t dt, bool adaptive, var_t tolerance, computing_device_t comp_dev) :
	integrator(f, dt, adaptive, tolerance, (adaptive ? 13 : 11), comp_dev)
{
	name    = "Runge-Kutta7";
	n_order = 7;
}

int_rungekutta7::~int_rungekutta7()
{}

void int_rungekutta7::calc_lin_comb(var_t* y, const var_t* y_n, const var_t* coeff, uint16_t n_coeff, uint32_t n_var)
{
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		// rk4_kernel::calc_lin_comb
		CUDA_CHECK_ERROR();
	}
	else
	{
		cpu_calc_lin_comb(y, y_n, coeff, n_coeff, n_var);
	}
}

void int_rungekutta7::calc_error(uint32_t n)
{
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		// rk4_kernel::calc_error
		CUDA_CHECK_ERROR();
	}
	else
	{
		cpu_calc_error(n);
	}
}

void int_rungekutta7::cpu_calc_lin_comb(var_t* y, const var_t* y_n, const var_t* coeff, uint16_t n_coeff, uint32_t n_var)
{
	for (uint32_t i = 0; i < n_var; i++)
	{
		var_t dy = 0.0;
		for (uint16_t j = 0; j < n_coeff; j++)
		{
			if (0.0 == coeff[j])
			{
				continue;
			}
			dy += coeff[j] * h_k[j][i];
		}
		y[i] = y_n[i] + dy;
	}
}

void int_rungekutta7::cpu_calc_error(uint32_t n)
{
	for (uint32_t i = 0; i < n; i++)
	{
		h_err[i] = fabs(h_k[0][i] + h_k[10][i] - h_k[11][i] - h_k[12][i]);
	}
}

ttt_t int_rungekutta7::step()
{
	static string err_msg1 = "The integrator could not provide the approximation of the solution with the specified tolerance.";

	static const uint16_t n_a = sizeof(int_rungekutta7::a) / sizeof(int_rungekutta7::a[0]);
	static const uint16_t n_b = sizeof(int_rungekutta7::b) / sizeof(int_rungekutta7::b[0]);
	static var_t aa[n_a];
	static var_t bb[n_b];

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		redutilcu::set_kernel_launch_param(f.n_var, THREADS_PER_BLOCK, grid, block);
	}

	uint16_t stage = 0;
	t = f.t;
	// Calculate initial differentials and store them into h_k
	f.calc_dy(stage, t, f.h_y, h_k[stage]);

	var_t max_err = 0.0;
	uint16_t iter = 0;
	do
	{
		dt_did = dt_try;
		// Compute in advance the dt_try * coefficients to save n_var multiplication per stage
		for (uint16_t i = 0; i < n_a; i++)
		{
			aa[i] = dt_try * a[i];
		}
		for (uint16_t i = 0; i < n_b; i++)
		{
			bb[i] = dt_try * b[i];
		}

		for (stage = 1; stage < 11; stage++)
		{
			t = f.t + c[stage] * dt_try;
			// Calculate the y_temp for the next f evaulation
			cpu_calc_lin_comb(h_ytemp, f.h_y, &aa[a_idx[stage-1]], stage, f.n_var);
			f.calc_dy(stage, t, h_ytemp, h_k[stage]);
		}
		// So far we have stage (=11) number of k vectors
		cpu_calc_lin_comb(f.h_yout, f.h_y, bb, stage, f.n_var);

		if (adaptive)
		{
			// Here stage = 11
			for ( ; stage < n_stage; stage++)
			{
				t = f.t + c[stage] * dt_try;
				// Calculate the y_temp for the next f evaulation
				cpu_calc_lin_comb(h_ytemp, f.h_y, &aa[a_idx[stage-1]], stage, f.n_var);
				f.calc_dy(stage, t, h_ytemp, h_k[stage]);
			}
			// calculate: err = abs(k0 + k10 - k11 - k12)
			calc_error(f.n_var);
			max_err = dt_try * LAMBDA * get_max_error(f.n_var);
			dt_try *= 0.9 * pow(tolerance / max_err, 1.0/(n_order));
		}
		iter++;
	} while (adaptive && max_iter > iter && dt_min < dt_try && max_err > tolerance);

	if (max_iter <= iter)
	{
		throw string(err_msg1 + " The number of iteration exceeded the limit.");
	}
	if (dt_min >= dt_try)
	{
		throw string(err_msg1 + " The stepsize is smaller than the limit.");
	}

	update_counters(iter);

	t = f.t + dt_did;
	f.tout = t;
	f.swap();

	return dt_did;
}

#undef LAMBDA
