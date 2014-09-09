// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes project
#include "int_euler.h"
#include "number_of_bodies.h"
#include "nbody_exception.h"
#include "red_macro.h"
#include "red_constants.h"


static __global__
	void kernel_print_vector(int n, const vec_t* v)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		printf("v[%4d].x : %20.16lf\n", i, v[i].x);
		printf("v[%4d].y : %20.16lf\n", i, v[i].y);
		printf("v[%4d].z : %20.16lf\n", i, v[i].z);
		printf("v[%4d].w : %20.16lf\n", i, v[i].w);
	}
}

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


namespace integrator
{
euler::euler(pp_disk *ppd, ttt_t dt) :
	name("Euler"),
	d_dy(2),
	dt_try(dt),
	dt_did(0.0),
	dt_next(0.0),
	ppd(ppd)
{
	const int n = ppd->n_bodies->total;
	t = ppd->t;

	for (int i = 0; i < 2; i++)
	{
		allocate_device_vector((void**)&(d_dy[i]), n*sizeof(vec_t));
	}
}

euler::~euler()
{
	cudaFree(d_dy[0]);
	cudaFree(d_dy[1]);
}

void euler::allocate_device_vector(void **d_ptr, size_t size)
{
	cudaMalloc(d_ptr, size);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw nbody_exception("cudaMalloc failed", cudaStatus);
	}
}

void euler::calculate_grid(int nData, int threads_per_block)
{
	int	nThread = std::min(threads_per_block, nData);
	int	nBlock = (nData + nThread - 1)/nThread;
	grid.x  = nBlock;
	block.x = nThread;
}

ttt_t euler::step()
{
	t = ppd->t;
	// Calculate initial differentials and store them into d_dy
	for (int i = 0; i < 2; i++)
	{
		ppd->calculate_dy(i, 0, t, ppd->sim_data->d_y[0], ppd->sim_data->d_y[1], d_dy[i]);
	}

	const int n_var = NDIM * ppd->n_bodies->total;
	calculate_grid(n_var, THREADS_PER_BLOCK);

	for (int i = 0; i < 2; i++)
	{	
		kernel_sum_vector<<<grid, block>>>(
			n_var, (var_t*)ppd->sim_data->d_y[i], (var_t*)d_dy[i], dt_try, (var_t*)ppd->sim_data->d_yout[i]);
		cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) 
		{
			throw nbody_exception("kernel_sum_vector failed", cudaStatus);
		}
	}

	dt_did = dt_try;
	dt_next = dt_try;

	ppd->t += dt_did;
	for (int i = 0; i < 2; i++)
	{
		swap(ppd->sim_data->d_yout[i], ppd->sim_data->d_y[i]);
	}

	return dt_did;
}

} /* namespace integrator */
