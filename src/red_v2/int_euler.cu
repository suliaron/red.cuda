#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ode.h"
#include "int_euler.h"

#include "redutilcu.h"
#include "red_macro.h"

namespace euler_kernel
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
} /* namespace euler_kernel */

void euler::cpu_sum_vector(var_t* a, const var_t* b, var_t F, const var_t* c, uint32_t n)
{
	for (uint32_t tid = 0; tid < n; tid++)
	{
		a[tid] = b[tid] + F * c[tid];
	}
}

euler::euler(ode& f, ttt_t dt, computing_device_t comp_dev) :
	integrator(f, dt, false, 0.0, 1, comp_dev)
{
	name    = "Euler";
	n_order = 1;
}

euler::~euler()
{}

void euler::calc_y_np1()
{
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		euler_kernel::sum_vector<<<grid, block>>>(f.d_yout, f.d_y, dt_try, d_k[0], f.n_var);
		CUDA_CHECK_ERROR();
	}
	else
	{
		cpu_sum_vector(f.h_yout, f.h_y, dt_try, h_k[0], f.n_var);
	}
}

ttt_t euler::step()
{
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		redutilcu::set_kernel_launch_param(f.n_var, THREADS_PER_BLOCK, grid, block);
	}

	uint16_t stage = 0;
	t = f.t;
	// Calculate initial differentials and store them into h_k
	f.calc_dy(stage, t, f.h_y, h_k[stage]);

	calc_y_np1();

	dt_did = dt_try;

	update_counters(1);

	t = f.t + dt_did;
	f.tout = t;
	f.swap();

	return dt_did;
}
