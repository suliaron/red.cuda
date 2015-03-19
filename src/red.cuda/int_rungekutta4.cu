// includes system
#include <algorithm>    // std::min_element, std::max_element

// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes Thrust
#ifdef __GNUC__
#include "thrust/device_ptr.h"
#include "thrust/fill.h"
#include "thrust/extrema.h"
#else
#include "thrust\device_ptr.h"
#include "thrust\fill.h"
#include "thrust\extrema.h"
#endif

// includes project
#include "int_rungekutta4.h"
#include "number_of_bodies.h"
#include "nbody_exception.h"
#include "red_macro.h"
#include "red_constants.h"
#include "util.h"

#define	LAMBDA	1.0/10.0

ttt_t rungekutta4::c[] =  {0.0, 1.0/2.0, 1.0/2.0, 1.0, 1.0};
var_t rungekutta4::a[] =  {0.0, 1.0/2.0, 1.0/2.0, 1.0, 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0};
var_t rungekutta4::bh[] = {1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0, 0.0};
var_t rungekutta4::b[] =  {1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0 -LAMBDA, LAMBDA};

namespace rk4_kernel
{
// result = a + b_factor * b
static __global__
	void sum_vector(int n, const var_t* a, const var_t* b, var_t b_factor, var_t* result)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		result[tid] = a[tid] + b_factor * b[tid];
		tid += stride;
	}
}

static __global__
	void calc_y_np1(int n, const var_t *y_n, const var_t *f1, const var_t *f2, const var_t *f3, const var_t *f4, var_t b0, var_t b1, var_t *y_np1)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		y_np1[tid] = y_n[tid] + b0 * (f1[tid] + f4[tid]) + b1 * (f2[tid] + f3[tid]);
		tid += stride;
	}
}

static __global__
	void calc_error(int n, const var_t *f4, const var_t* f5, var_t *err)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		err[tid] = f4[tid] - f5[tid];
		tid += stride;
	}
}
} /* rk4_kernel */

void rungekutta4::cpu_sum_vector(int n, const var_t* a, const var_t* b, var_t b_factor, var_t* result)
{
	for (int i = 0; i < n; i++)
	{
		result[i] = a[i] + b_factor * b[i];
	}
}

void rungekutta4::cpu_calc_y_np1(int n, const var_t *y_n, const var_t *f1, const var_t *f2, const var_t *f3, const var_t *f4, var_t b0, var_t b1, var_t *y_np1)
{
	for (int i = 0; i < n; i++)
	{
		y_np1[i] = y_n[i] + b0 * (f1[i] + f4[i]) + b1 * (f2[i] + f3[i]);
	}
}

void rungekutta4::cpu_calc_error(int n, const var_t *f4, const var_t* f5, var_t *err)
{
	for (int i = 0; i < n; i++)
	{
		err[i] = f4[i] - f5[i];
	}
}


rungekutta4::rungekutta4(pp_disk *ppd, ttt_t dt, bool adaptive, var_t tolerance, computing_device_t comp_dev) :
	integrator(ppd, dt, adaptive, tolerance, (adaptive ? 5 : 4), comp_dev)
{
	name = "Runge-Kutta4";
	short_name = "RK4";

	order = 4;
}

rungekutta4::~rungekutta4()
{
}

void rungekutta4::calc_ytemp_for_fr(int n_var, int r)
{
	for (int i = 0; i < 2; i++)
	{
		var_t *y_n  = (var_t*)ppd->sim_data->y[i];
		var_t *fr   = (var_t*)dydx[i][r-1];
		var_t* ytmp = (var_t*)ytemp[i];

		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			rk4_kernel::sum_vector<<<grid, block>>>(n_var, y_n, fr, a[r] * dt_try, ytmp);
			cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
			if (cudaSuccess != cudaStatus)
			{
				throw string("rk4_kernel::sum_vector failed");
			}
		}
		else
		{
			cpu_sum_vector(n_var, y_n, fr, a[r] * dt_try, ytmp);
		}
	}
}

void rungekutta4::calc_y_np1(int n_var)
{
	for (int i = 0; i < 2; i++)
	{
		var_t *y_n	 = (var_t*)ppd->sim_data->y[i];
		var_t *y_np1 = (var_t*)ppd->sim_data->yout[i];
		var_t *f1	 = (var_t*)dydx[i][0];
		var_t *f2	 = (var_t*)dydx[i][1];
		var_t *f3	 = (var_t*)dydx[i][2];
		var_t *f4	 = (var_t*)dydx[i][3];

		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			rk4_kernel::calc_y_np1<<<grid, block>>>(n_var, y_n, f1, f2, f3, f4, b[0] * dt_try, b[1] * dt_try, y_np1);
			cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
			if (cudaSuccess != cudaStatus)
			{
				throw string("rk4_kernel::calc_y_np1 failed");
			}
		}
		else
		{
			cpu_calc_y_np1(n_var, y_n, f1, f2, f3, f4, b[0] * dt_try, b[1] * dt_try, y_np1);
		}
	}
}

void rungekutta4::calc_error(int n_var)
{
	for (int i = 0; i < 2; i++)
	{
		var_t *f4  = (var_t*)dydx[i][3];
		var_t *f5  = (var_t*)dydx[i][4];

		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			rk4_kernel::calc_error<<<grid, block>>>(n_var, f4, f5, err[i]);
			cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
			if (cudaSuccess != cudaStatus)
			{
				throw string("rk4_kernel::calc_error failed");
			}
		}
		else
		{
			cpu_calc_error(n_var, f4, f5, err[i]);
		}
	}
}

ttt_t rungekutta4::step()
{
	const int n_body_total = ppd->get_ups() ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_total();
	const int n_var_total = NDIM * n_body_total;

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		// Set the kernel launch parameters
		calc_grid(n_var_total, THREADS_PER_BLOCK);
	}

	// Calculate initial differentials and store them into d_f[][0]
	int r = 0;
	ttt_t ttemp = ppd->t + c[r] * dt_try;
	// Calculate f1 = f(tn, yn) = d_f[][0]
	const vec_t *coor = ppd->sim_data->y[0];
	const vec_t *velo = ppd->sim_data->y[1];
	for (int i = 0; i < 2; i++)
	{
		ppd->calc_dydx(i, r, ttemp, coor, velo, dydx[i][r]);
	}

	var_t max_err = 0.0;
	int iter = 0;
	do
	{
		dt_did = dt_try;
		// Calculate f2 = f(tn + c2 * dt, yn + a21 * dt * f1) = dydx[][1]
		// Calculate f3 = f(tn + c3 * dt, yn + a31 * dt * f2) = dydx[][2]
		// Calculate f4 = f(tn + c4 * dt, yn + a41 * dt * f3) = dydx[][3]
		for (r = 1; r < order; r++)
		{
			ttemp = ppd->t + c[r] * dt_try;
			calc_ytemp_for_fr(n_var_total, r);
			for (int i = 0; i < 2; i++)
			{
				ppd->calc_dydx(i, r, ttemp, ytemp[0], ytemp[1], dydx[i][r]);
			}
		}
		// y_(n+1) = yn + dt*(1/6*f1 + 1/3*f2 + 1/3*f3 + 1/6*f4) + O(dt^5)
		calc_y_np1(n_var_total);

		if (adaptive)
		{
			r = 4;
			ttemp = ppd->t + c[r] * dt_try;
			// Calculate f5 = f(tn + c5 * dt,  yn + dt*(1/6*f1 + 1/3*f2 + 1/3*f3 + 1/6*f4)) = dydx[][4]
			for (int i = 0; i < 2; i++)
			{
				ppd->calc_dydx(i, r, ttemp, ppd->sim_data->yout[0], ppd->sim_data->yout[1], dydx[i][r]);
			}

			int n_var = 0;
			if (ppd->get_ups())
			{
				n_var = NDIM * (error_check_for_tp ? n_body_total : ppd->n_bodies->get_n_prime_massive());
			}
			else
			{
				n_var = NDIM * (error_check_for_tp ? n_body_total : ppd->n_bodies->get_n_massive());
			}
			// calculate: err = (f4 - f5)
			calc_error(n_var);
			max_err = get_max_error(n_var, LAMBDA);
			dt_try *= 0.9 * pow(tolerance / max_err, 1.0/order);
		}
		iter++;
	} while (adaptive && max_err > tolerance);

	update_counters(iter);

	ppd->t += dt_did;
	ppd->swap();

	return dt_did;
}

#undef LAMBDA
