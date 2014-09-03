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
void kernel_sum_vector(int n, const vec_t* a, const vec_t* b, var_t b_factor, vec_t* result)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		result[tid].x = a[tid].x + b_factor * b[tid].x;
		result[tid].y = a[tid].y + b_factor * b[tid].y;
		result[tid].z = a[tid].z + b_factor * b[tid].z;
		result[tid].w = a[tid].w;
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
	// Calculate initial differentials and store them into d_dy
	for (int i = 0; i < 2; i++)
	{
		ppd->calculate_dy(i, 0, t, ppd->sim_data->d_pos, ppd->sim_data->d_vel, d_dy[i]);
	}

	const int n = ppd->n_bodies->total;
	calculate_grid(n, THREADS_PER_BLOCK);

	// Create aliases
	vec_t* d_pos	 = (vec_t*)ppd->sim_data->d_pos;
	vec_t* d_pos_out = (vec_t*)ppd->sim_data->d_pos_out;
	vec_t* d_vel	 = (vec_t*)ppd->sim_data->d_vel;
	vec_t* d_vel_out = (vec_t*)ppd->sim_data->d_vel_out;
	for (int i = 0; i < 2; i++)
	{	
		if (i == 0)	{ // Propagate position
			//kernel_print_vector<<<grid, block>>>(n, d_pos_out);
			//cudaDeviceSynchronize();
			kernel_sum_vector<<<grid, block>>>(n, d_pos, d_dy[i], dt_try, d_pos_out);
			//kernel_print_vector<<<grid, block>>>(n, d_pos_out);
			//cudaDeviceSynchronize();
		}
		else {		  // Propagate velocity
			kernel_sum_vector<<<grid, block>>>(n, d_vel, d_dy[i], dt_try, d_vel_out);
		}
		cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) 
		{
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
