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
var_t rungekutta2::b[] = {0.0, 1.0};
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
} /* rk2_kernel */

void rungekutta2::cpu_sum_vector(int n, const var_t* a, const var_t* b, var_t b_factor, var_t* result)
{
	for (int i = 0; i < n; i++)
	{
		result[i] = a[i] + b_factor * b[i];
	}
}


rungekutta2::rungekutta2(pp_disk *ppd, ttt_t dt, bool cpu) :
	integrator(ppd, dt, cpu),
	RKOrder(2),
	dydx(2)
{
	name = "Runge-Kutta2";
	short_name = "RK2";

	const int n_total = ppd->get_ups() ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_total();

	t = ppd->t;
	for (int i = 0; i < 2; i++)
	{
		dydx[i].resize(RKOrder);
		for (int r = 0; r < RKOrder; r++) 
		{
			ALLOCATE_VECTOR((void**)&(dydx[i][r]), n_total*sizeof(vec_t), cpu);
		}
	}
}

rungekutta2::~rungekutta2()
{
	for (int i = 0; i < 2; i++)
	{
		for (int r = 0; r < RKOrder; r++) 
		{
			FREE_VECTOR(dydx[i][r], cpu);
		}
	}
}

void rungekutta2::call_kernel_calc_ytemp_for_fr(int n_var, int r)
{
	for (int i = 0; i < 2; i++)
	{
		var_t *y_n	  = (var_t*)ppd->sim_data->d_y[i];
		var_t *fr	  = (var_t*)dydx[i][r-1];
		var_t* result = (var_t*)ytemp[i];

		rk2_kernel::sum_vector<<<grid, block>>>(n_var, y_n, fr, a[r] * dt_try, result);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus)
		{
			throw string("kernel_sum_vector failed");
		}
	}
}

void rungekutta2::call_kernel_calc_y_np1(int n_var)
{
	for (int i = 0; i < 2; i++)
	{
		var_t *y_n	 = (var_t*)ppd->sim_data->d_y[i];
		var_t *y_np1 = (var_t*)ppd->sim_data->d_yout[i];
		var_t *f2	 = (var_t*)dydx[i][1];

		rk2_kernel::sum_vector<<<grid, block>>>(n_var, y_n, f2, b[1] * dt_try, y_np1);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus)
		{
			throw string("kernel_sum_vector failed");
		}
	}
}

void rungekutta2::cpu_calc_ytemp_for_fr(int n_var, int r)
{
	for (int i = 0; i < 2; i++)
	{
		var_t *y_n	  = (var_t*)ppd->sim_data->y[i];
		var_t *fr	  = (var_t*)dydx[i][r-1];
		var_t* result = (var_t*)ytemp[i];

		cpu_sum_vector(n_var, y_n, fr, a[r] * dt_try, result);
	}
}

void rungekutta2::cpu_calc_y_np1(int n_var)
{
	for (int i = 0; i < 2; i++)
	{
		var_t *y_n	 = (var_t*)ppd->sim_data->y[i];
		var_t *y_np1 = (var_t*)ppd->sim_data->yout[i];
		var_t *f2	 = (var_t*)dydx[i][1];

		cpu_sum_vector(n_var, y_n, f2, b[1] * dt_try, y_np1);
	}
}

void rungekutta2::calc_ytemp_for_fr(int n_var, int r)
{
	if (!cpu)
	{
		call_kernel_calc_ytemp_for_fr(n_var, r);
	}
	else
	{
		cpu_calc_ytemp_for_fr(n_var, r);
	}
}

void rungekutta2::calc_y_np1(int n_var)
{
	if (!cpu)
	{
		call_kernel_calc_y_np1(n_var);
	}
	else
	{
		cpu_calc_y_np1(n_var);
	}
}

ttt_t rungekutta2::step()
{
	// Set the kernel launch parameters
	const int n_body_total = ppd->get_ups() ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_total();
	const int n_var_total = NDIM * n_body_total;
	if (!cpu)
	{
		calc_grid(n_var_total, THREADS_PER_BLOCK);
	}

	int r = 0;
	ttt_t ttemp = ppd->t + c[r] * dt_try;
	// Calculate initial differentials f1 = f(tn, yn) and store them into dydx[][0]
	for (int i = 0; i < 2; i++)
	{
		const vec_t *coor = cpu ? ppd->sim_data->y[0] : ppd->sim_data->d_y[0];
		const vec_t *velo = cpu ? ppd->sim_data->y[1] : ppd->sim_data->d_y[1];

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
