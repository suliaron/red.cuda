// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes project
#include "int_rungekutta2.h"
#include "number_of_bodies.h"
#include "nbody_exception.h"
#include "red_macro.h"
#include "red_constants.h"
#include "util.h"

var_t rungekutta2::a[] = {0.0, 1.0/2.0};
var_t rungekutta2::b[] = {0.0, 1.0    };
ttt_t rungekutta2::c[] = {0.0, 1.0/2.0};

namespace rk2_kernel
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
} /* namespace rk2_kernel */

void rungekutta2::cpu_sum_vector(int n, const var_t* a, const var_t* b, var_t b_factor, var_t* result)
{
	for (int i = 0; i < n; i++)
	{
		result[i] = a[i] + b_factor * b[i];
	}
}

rungekutta2::rungekutta2(pp_disk *ppd, ttt_t dt, computing_device_t comp_dev) :
	integrator(ppd, dt, false, 0.0, 2, comp_dev)
{
	name = "Runge-Kutta2";
	short_name = "RK2";

	order = 2;
}

rungekutta2::~rungekutta2()
{
}

void rungekutta2::calc_ytemp_for_fr(int n_var, int r)
{
	for (int i = 0; i < 2; i++)
	{
		var_t *y_n	  = (var_t*)ppd->sim_data->y[i];
		var_t *fr	  = (var_t*)dydx[i][r-1];
		var_t* result = (var_t*)ytemp[i];

		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			rk2_kernel::sum_vector<<<grid, block>>>(n_var, y_n, fr, a[r] * dt_try, result);
			cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
			if (cudaSuccess != cudaStatus)
			{
				throw string("rk2_kernel::sum_vector failed");
			}
		}
		else
		{
			cpu_sum_vector(n_var, y_n, fr, a[r] * dt_try, result);
		}
	}
}

void rungekutta2::calc_y_np1(int n_var)
{
	for (int i = 0; i < 2; i++)
	{
		var_t *y_n	 = (var_t*)ppd->sim_data->y[i];
		var_t *f2	 = (var_t*)dydx[i][1];
		var_t *y_np1 = (var_t*)ppd->sim_data->yout[i];

		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			rk2_kernel::sum_vector<<<grid, block>>>(n_var, y_n, f2, b[1] * dt_try, y_np1);
			cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
			if (cudaSuccess != cudaStatus)
			{
				throw string("rk2_kernel::sum_vector failed");
			}
		}
		else
		{
			cpu_sum_vector(n_var, y_n, f2, b[1] * dt_try, y_np1);
		}
	}
}

ttt_t rungekutta2::step()
{
	const int n_body_total = ppd->get_ups() ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_total();
	const int n_var_total = NDIM * n_body_total;

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		// Set the kernel launch parameters
		calc_grid(n_var_total, THREADS_PER_BLOCK);
	}

	int r = 0;
	ttt_t ttemp = ppd->t + c[r] * dt_try;
	// Calculate initial differentials f1 = f(tn, yn) and store them into dydx[][0]
	const vec_t *coor = ppd->sim_data->y[0];
	const vec_t *velo = ppd->sim_data->y[1];
	for (int i = 0; i < 2; i++)
	{
		ppd->calc_dydx(i, r, ttemp, coor, velo, dydx[i][r]);
	}

	r = 1;
	ttemp = ppd->t + c[r] * dt_try;
	calc_ytemp_for_fr(n_var_total, r);
	// Calculate f2 = f(tn + 1/2*h, yn + 1/2*h*f1) = d_f[][1]
	for (int i = 0; i < 2; i++)
	{
		ppd->calc_dydx(i, r, ttemp, ytemp[0], ytemp[1], dydx[i][r]);
	}
	calc_y_np1(n_var_total);

	dt_did = dt_try;
	dt_next = dt_try;

	update_counters(1);

	ppd->t += dt_did;
	ppd->swap();

	return dt_did;
}
