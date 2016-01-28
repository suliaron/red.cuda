#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ode.h"
#include "int_rungekutta2.h"

#include "redutilcu.h"
#include "red_macro.h"

// The Runge-Kutta matrix
var_t int_rungekutta2::a[] = 
{ 
	0.0,     0.0, 
	1.0/2.0, 0.0
};
// weights
var_t int_rungekutta2::b[] = { 0.0, 1.0     };
// nodes
ttt_t int_rungekutta2::c[] = { 0.0, 1.0/2.0 };

namespace rk2_kernel
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
} /* namespace rk2_kernel */

void int_rungekutta2::cpu_sum_vector(var_t* a, const var_t* b, var_t F, const var_t* c, uint32_t n)
{
	for (uint32_t tid = 0; tid < n; tid++)
	{
		a[tid] = b[tid] + F * c[tid];
	}
}

int_rungekutta2::int_rungekutta2(ode& f, ttt_t dt, computing_device_t comp_dev) :
	integrator(f, dt, false, 0.0, 2, comp_dev)
{
	name    = "Runge-Kutta2";
	n_order = 2;
}

int_rungekutta2::~int_rungekutta2()
{}

void int_rungekutta2::calc_y_np1()
{
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		rk2_kernel::sum_vector<<<grid, block>>>(f.d_yout, f.d_y, dt_try, d_k[1], f.n_var);
		CUDA_CHECK_ERROR();
	}
	else
	{
		cpu_sum_vector(f.h_yout, f.h_y, dt_try, h_k[1], f.n_var);
	}
}

void int_rungekutta2::calc_ytemp(uint16_t stage)
{
	for (uint32_t i = 0; i < f.n_var; i++)
	{
		var_t dy = 0.0;
		for (uint16_t j = 0; j < stage; j++)
		{
			dy += a[stage * n_stage + j] * h_k[j][i];
		}
		h_ytemp[i] = f.h_y[i] + dt_try * dy;
	}
}

ttt_t int_rungekutta2::step()
{
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		redutilcu::set_kernel_launch_param(f.n_var, THREADS_PER_BLOCK, grid, block);
	}

	uint16_t stage = 0;
	t = f.t;
	// Calculate initial differentials and store them into h_k
	f.calc_dy(stage, t, f.h_y, h_k[stage]);

	stage = 1;
	t = f.t + c[stage] * dt_try;
	calc_ytemp(stage);
	f.calc_dy(stage, t, h_ytemp, h_k[stage]);

	calc_y_np1();

	dt_did = dt_try;

	update_counters(1);

	t = f.t + dt_did;
	f.tout = t;
	f.swap();

	return dt_did;
}
