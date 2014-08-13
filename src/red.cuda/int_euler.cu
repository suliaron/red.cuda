// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes project
#include "int_euler.h"
#include "number_of_bodies.h"
#include "nbody_exception.h"
#include "red_macro.h"
#include "red_constants.h"

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
euler::euler(ttt_t t0, ttt_t dt, pp_disk *ppd) :
	name("Euler"),
	d_dy(2),
	t(t0),
	dt_try(dt),
	dt_did(0.0),
	dt_next(0.0),
	ppd(ppd)
{
	const int n = ppd->n_bodies->total;

	// Allocate device pointer.
	for (int i = 0; i < 2; i++)
	{
		cudaMalloc((void**) &(d_dy[i]), n*sizeof(vec_t));
		cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("cudaMalloc failed", cudaStatus);
		}
	}
}

euler::~euler()
{
	cudaFree(d_dy[0]);
	cudaFree(d_dy[1]);
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
	// Calculate initial differentials and store them into d_dy
	for (int i = 0; i < 2; i++)
	{
		ppd->calculate_dy(i, 0, t, ppd->sim_data->d_pos, ppd->sim_data->d_vel, d_dy[i]);
	}

	const int n = ppd->n_bodies->total * sizeof(vec_t);
	calculate_grid(n, THREADS_PER_BLOCK);

	// Create aliases
	var_t* d_pos_out = (var_t*)ppd->sim_data->d_pos_out;
	var_t* d_pos	 = (var_t*)ppd->sim_data->d_pos;
	var_t* d_vel_out = (var_t*)ppd->sim_data->d_vel_out;
	var_t* d_vel	 = (var_t*)ppd->sim_data->d_vel;
	for (int i = 0; i < 2; i++)
	{	
		if (i == 0)	{ // Propagate position
			kernel_sum_vector<<<grid, block>>>(n, d_pos, (var_t*)d_dy[i], dt_try, d_pos_out);
		}
		else {		  // Propagate velocity
			kernel_sum_vector<<<grid, block>>>(n, d_vel, (var_t*)d_dy[i], dt_try, d_vel_out);
		}
		cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("kernel_sum_vector failed", cudaStatus);
		}
	}

	dt_did = dt_try;
	dt_next = dt_try;

	ppd->t += dt_did;
	swap(ppd->sim_data->d_pos_out, ppd->sim_data->d_pos);
	swap(ppd->sim_data->d_vel_out, ppd->sim_data->d_vel);

	return dt_did;
}

} /* namespace integrator */
