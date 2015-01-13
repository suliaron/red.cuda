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

rungekutta2::rungekutta2(pp_disk *ppd, ttt_t dt) :
	integrator(ppd, dt),
	RKOrder(2),
	d_f(2)
{
	name = "Runge-Kutta2";
	short_name = "RK2";

	const int n_total = ppd->get_ups() ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_total();

	t = ppd->t;
	for (int i = 0; i < 2; i++)
	{
		ALLOCATE_DEVICE_VECTOR((void**) &(d_ytemp[i]), n_total*sizeof(vec_t));
		d_f[i].resize(RKOrder);
		for (int r = 0; r < RKOrder; r++) 
		{
			ALLOCATE_DEVICE_VECTOR((void**) &(d_f[i][r]), n_total * sizeof(vec_t));
		}
	}
}

rungekutta2::~rungekutta2()
{
	for (int i = 0; i < 2; i++)
	{
		for (int r = 0; r < RKOrder; r++) 
		{
			cudaFree(d_f[i][r]);
		}
	}
}

void rungekutta2::call_kernel_calc_ytemp_for_fr(int n_var, int r)
{
	for (int i = 0; i < 2; i++)
	{
		var_t *y_n	  = (var_t*)ppd->sim_data->d_y[i];
		var_t *fr	  = (var_t*)d_f[i][r-1];
		var_t* result = (var_t*)d_ytemp[i];

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
		var_t *f2	 = (var_t*)d_f[i][1];

		rk2_kernel::sum_vector<<<grid, block>>>(n_var, y_n, f2, b[1] * dt_try, y_np1);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus)
		{
			throw string("kernel_sum_vector failed");
		}
	}
}

ttt_t rungekutta2::step()
{
	// Set the kernel launch parameters
	const int n_body_total = ppd->get_ups() ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_total();
	const int n_var_total = NDIM * n_body_total;
	calc_grid(n_var_total, THREADS_PER_BLOCK);

	int r = 0;
	ttt_t ttemp = ppd->t + c[r] * dt_try;
	// Calculate initial differentials f1 = f(tn, yn) and store them into d_f[][0]
	for (int i = 0; i < 2; i++)
	{
		ppd->calc_dy(i, r, ttemp, ppd->sim_data->d_y[0], ppd->sim_data->d_y[1], d_f[i][r]);
	}

	r = 1;
	ttemp = ppd->t + c[r] * dt_try;
	call_kernel_calc_ytemp_for_fr(n_var_total, r);

	// Calculate f2 = f(tn + 1/2*h, yn + 1/2*h*f1) = d_f[][1]
	for (int i = 0; i < 2; i++)
	{
		ppd->calc_dy(i, r, ttemp, d_ytemp[0], d_ytemp[1], d_f[i][r]);
	}

	dt_did = dt_try;
	call_kernel_calc_y_np1(n_var_total);

	update_counters(1);

	ppd->t += dt_did;
	for (int i = 0; i < 2; i++)
	{
		swap(ppd->sim_data->d_yout[i], ppd->sim_data->d_y[i]);
	}

	return dt_did;
}
