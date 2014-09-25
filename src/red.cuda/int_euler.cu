// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes project
#include "int_euler.h"
#include "number_of_bodies.h"
#include "nbody_exception.h"
#include "red_macro.h"
#include "red_constants.h"
#include "util.h"

// result = a + b_factor * b
static __global__
	void kernel_sum_vector(int n, const var_t* a, const var_t* b, var_t b_factor, var_t* result)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		result[tid] = a[tid] + b_factor * b[tid];
		tid += stride;
	}
}


euler::euler(pp_disk *ppd, ttt_t dt) :
	integrator(ppd, dt),
	d_df(2)
{
	const int n = ppd->n_bodies->total;
	name = "Euler";

	t = ppd->t;
	for (int i = 0; i < 2; i++)
	{
		ALLOCATE_DEVICE_VECTOR((void**)&(d_df[i]), n*sizeof(vec_t));
	}
}

euler::~euler()
{
	cudaFree(d_df[0]);
	cudaFree(d_df[1]);
}

void euler::call_kernel_calc_y_np1()
{
	const int n_var = NDIM * ppd->n_bodies->total;

	calc_grid(n_var, THREADS_PER_BLOCK);
	for (int i = 0; i < 2; i++)
	{	
		var_t *y_n	 = (var_t*)ppd->sim_data->d_y[i];
		var_t *y_np1 = (var_t*)ppd->sim_data->d_yout[i];
		var_t *f0	 = (var_t*)d_df[i];

		kernel_sum_vector<<<grid, block>>>(n_var, y_n, f0, dt_try, y_np1);
		cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) 
		{
			throw nbody_exception("kernel_sum_vector failed", cudaStatus);
		}
	}
}

ttt_t euler::step()
{
	t = ppd->t;
	// Calculate initial differentials and store them into d_dy
	for (int i = 0; i < 2; i++)
	{
		ppd->calc_dy(i, 0, t, ppd->sim_data->d_y[0], ppd->sim_data->d_y[1], d_df[i]);
	}
	call_kernel_calc_y_np1();

	dt_did = dt_try;
	dt_next = dt_try;

	update_counters(1);

	ppd->t += dt_did;
	for (int i = 0; i < 2; i++)
	{
		swap(ppd->sim_data->d_yout[i], ppd->sim_data->d_y[i]);
	}

	return dt_did;
}
