#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ode.h"
#include "int_rungekutta4.h"

#include "red_macro.h"
#include "redutilcu.h"

#define	LAMBDA	1.0/10.0

// The Runge-Kutta matrix
var_t int_rungekutta4::a[] = 
{ 
	1.0/2.0,
	0.0,     1.0/2.0,
	0.0,     0.0,     1.0,
    1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0
};
// weights
var_t int_rungekutta4::b[]  = { 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0 - LAMBDA, LAMBDA };
var_t int_rungekutta4::bh[] = { 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0         ,    0.0 };
// nodes
ttt_t int_rungekutta4::c[]  = {     0.0, 1.0/2.0, 1.0/2.0, 1.0, 1.0                 };
// The starting index of the RK matrix for the stages
uint16_t int_rungekutta4::a_idx[] = {0, 1, 3, 6};

namespace rk4_kernel
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
} /* namespace rk4_kernel */

int_rungekutta4::int_rungekutta4(ode& f, ttt_t dt, bool adaptive, var_t tolerance, computing_device_t comp_dev) :
	integrator(f, dt, adaptive, tolerance, (adaptive ? 5 : 4), comp_dev)
{
	name    = "Runge-Kutta4";
	n_order = 4;
}

int_rungekutta4::~int_rungekutta4()
{}

void int_rungekutta4::calc_lin_comb(var_t* y, const var_t* y_n, const var_t* coeff, uint16_t n_coeff, uint32_t n_var)
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

//void int_rungekutta4::calc_ytemp(uint16_t stage)
//{
//	if (COMPUTING_DEVICE_GPU == comp_dev)
//	{
//		// rk4_kernel::calc_ytemp
//		CUDA_CHECK_ERROR();
//	}
//	else
//	{
//		cpu_calc_ytemp(stage);
//	}
//}

void int_rungekutta4::calc_error(uint32_t n)
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

//void int_rungekutta4::calc_y_np1()
//{
//	if (COMPUTING_DEVICE_GPU == comp_dev)
//	{
//		// rk4_kernel::calc_y_np1
//		CUDA_CHECK_ERROR();
//	}
//	else
//	{
//		cpu_calc_y_np1();
//	}
//}

// a_i = b_i + F * c_i
//void int_rungekutta4::cpu_sum_vector(var_t* a, const var_t* b, var_t F, const var_t* c, uint32_t n)
//{
//	for (uint32_t tid = 0; tid < n; tid++)
//	{
//		a[tid] = b[tid] + F * c[tid];
//	}
//}

void int_rungekutta4::cpu_calc_lin_comb(var_t* y, const var_t* y_n, const var_t* coeff, uint16_t n_coeff, uint32_t n_var)
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

//void int_rungekutta4::cpu_calc_ytemp(uint16_t stage)
//{
//	for (uint32_t i = 0; i < f.n_var; i++)
//	{
//		var_t dy = 0.0;
//		for (uint16_t j = 0; j < stage; j++)
//		{
//			if (0.0 == a[a_idx[stage-1] + j])
//			{
//				continue;
//			}
//			dy += a[a_idx[stage-1] + j] * h_k[j][i];
//		}
//
//		h_ytemp[i] = f.h_y[i] + dt_try * dy;
//	}
//}
//
//void int_rungekutta4::cpu_calc_y_np1()
//{
//	for (uint32_t i = 0; i < f.n_var; i++)
//	{
//		var_t dy = 0.0;
//		for (uint16_t j = 0; j < n_order; j++)
//		{
//			if (0.0 == b[j])
//			{
//				continue;
//			}
//			dy += b[j] * h_k[j][i];
//		}
//		f.h_yout[i] = f.h_y[i] + dt_try * dy;
//	}
//}

void int_rungekutta4::cpu_calc_error(uint32_t n)
{
	for (uint32_t i = 0; i < n; i++)
	{
		h_err[i] = fabs(h_k[3][i] - h_k[4][i]);
	}
}

ttt_t int_rungekutta4::step()
{
	static string err_msg1 = "The integrator could not provide the approximation of the solution with the specified tolerance.";

	static const uint16_t n_a = sizeof(int_rungekutta4::a) / sizeof(int_rungekutta4::a[0]);
	static const uint16_t n_b = sizeof(int_rungekutta4::b) / sizeof(int_rungekutta4::b[0]);
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
			bb[i] = dt_try * bh[i];
		}

		for (stage = 1; stage < 4; stage++)
		{
			t = f.t + c[stage] * dt_try;
			cpu_calc_lin_comb(h_ytemp, f.h_y, &aa[a_idx[stage-1]], stage, f.n_var);
			//calc_ytemp(stage);
			f.calc_dy(stage, t, h_ytemp, h_k[stage]);
		}
		// y_(n+1) = yn + dt*(1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4) + O(dt^5)
		// So far we have stage (=4) number of k vectors
		cpu_calc_lin_comb(f.h_yout, f.h_y, bb, stage, f.n_var);
		//calc_y_np1();

		if (adaptive)
		{
			// Here stage = 4
			t = f.t + c[stage] * dt_try;
			f.calc_dy(stage, t, f.h_yout, h_k[stage]);

			// calculate: err = abs(k4 - k5)
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
