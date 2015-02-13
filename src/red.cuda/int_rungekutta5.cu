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
#include "int_rungekutta5.h"
#include "number_of_bodies.h"
#include "nbody_exception.h"
#include "red_macro.h"
#include "red_constants.h"
#include "util.h"

ttt_t rungekutta5::c[] =  { 0.0, 1.0/5.0, 3.0/10.0, 3.0/5.0, 1.0, 7.0/8.0 };
var_t rungekutta5::a[] =  {            0.0,  
                                   1.0/5.0, 
                                  3.0/40.0,    9.0/40.0,
							      3.0/10.0,   -9.0/10.0,       6.0/5.0, 
							    -11.0/54.0,     5.0/2.0,    -70.0/27.0,        35.0/27.0, 
							1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0 };
var_t rungekutta5::bh[] = {    37.0/378.0, 0.0,     250.0/621.0,     125.0/594.0,           0.0, 512.0/1771.0};
var_t rungekutta5::b[] =  {2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0,      1.0/4.0};

__constant__ var_t dc_a[ sizeof(rungekutta5::a)  / sizeof(var_t)];
__constant__ var_t dc_b[ sizeof(rungekutta5::b)  / sizeof(var_t)];
__constant__ var_t dc_bh[sizeof(rungekutta5::bh) / sizeof(var_t)];
__constant__ var_t dc_c[ sizeof(rungekutta5::c)  / sizeof(ttt_t)];

namespace rk5_kernel
{
static __global__
	void calc_ytemp(int n, int r, int idx, int offset, ttt_t dt, const var_t *y_n, var_t** dydt, var_t *ytemp)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n)
	{
		ytemp[tid] = y_n[tid];
		for (int i = 0; i < r; i++)
		{
			if (0.0 == dc_a[idx + i])
			{
				continue;
			}
			ytemp[tid] += dt * dc_a[idx + i] * dydt[offset + i][tid];
		}
	}
}

static __global__
	void calc_y_np1(int n, int offset, ttt_t dt, const var_t *y_n, var_t** dydt, var_t *y_np1)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n)
	{
		y_np1[tid] = y_n[tid];
		for (int i = 0; i < 6; i++)
		{
			if (0.0 == dc_b[i])
			{
				continue;
			}
			y_np1[tid] += dt * dc_b[i] * dydt[offset + i][tid];
		}
	}
}

static __global__
	void calc_error(int n, int offset, var_t** dydt, var_t *err)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n)
	{
		err[tid] = (dc_bh[0] - dc_b[0]) * dydt[offset + 0][tid];
		for (int i = 1; i < 6; i++)
		{
			err[tid] += (dc_bh[i] - dc_b[i]) * dydt[offset + i][tid];
		}
	}
}
} /* rk5_kernel */

rungekutta5::rungekutta5(pp_disk *ppd, ttt_t dt, bool adaptive, var_t tolerance, computing_device_t comp_dev) :
	integrator(ppd, dt, comp_dev),
	adaptive(adaptive),
	tolerance(tolerance),
	d_f(2),
	d_dydt(0x0),
	d_err(2)
{
	name = "Runge-Kutta5";
	short_name = "RK5";

	const int n_total = ppd->get_ups() ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_total();

	t = ppd->t;
	RKOrder = 5;
	r_max = adaptive ? RKOrder + 1 : RKOrder;

	ALLOCATE_DEVICE_VECTOR((void**)&d_dydt, 2*r_max*sizeof(vec_t*));

	for (int i = 0; i < 2; i++)
	{
		d_f[i].resize(r_max);
		for (int r = 0; r < r_max; r++) 
		{
			ALLOCATE_DEVICE_VECTOR((void**) &(d_f[i][r]), n_total * sizeof(vec_t));
			copy_vector_to_device((void*)&d_dydt[i*r_max + r], &d_f[i][r], sizeof(var_t*));
		}
		if (adaptive)
		{
			static const int n_var = NDIM * n_total;
			ALLOCATE_DEVICE_VECTOR((void**) &(d_err[i]), n_var * sizeof(var_t));
		}
	}

	copy_constant_to_device(dc_a, a, sizeof(a));
	copy_constant_to_device(dc_b, b, sizeof(b));
	copy_constant_to_device(dc_bh, bh, sizeof(bh));
	copy_constant_to_device(dc_c, c, sizeof(c));
}

rungekutta5::~rungekutta5()
{
	for (int i = 0; i < 2; i++)
	{
		for (int r = 0; r < r_max; r++) 
		{
			FREE_DEVICE_VECTOR(&d_f[i][r]);
		}
		if (adaptive)
		{
			FREE_DEVICE_VECTOR(&d_err[i]);
		}
	}
	FREE_DEVICE_VECTOR(&d_dydt);
}

void rungekutta5::call_kernel_calc_ytemp(int n_var, int r)
{
	static int idx_array[] = {0, 1, 2, 4, 7, 11};

	for (int i = 0; i < 2; i++)
	{
		var_t* y_n   = (var_t*)ppd->sim_data->d_y[i];
		var_t** dydt = (var_t**)d_dydt;
		var_t* ytmp = (var_t*)ytemp[i];

		rk5_kernel::calc_ytemp<<<grid, block>>>(n_var, r, idx_array[r], i*r_max, dt_try, y_n, dydt, ytmp);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus)
		{
			throw string("rk5_kernel::calc_ytemp failed");
		}
	}
}

void rungekutta5::call_kernel_calc_y_np1(int n_var)
{
	for (int i = 0; i < 2; i++)
	{
		var_t* y_n   = (var_t*)ppd->sim_data->d_y[i];
		var_t** dydt = (var_t**)d_dydt;
		var_t* y_np1 = (var_t*)ppd->sim_data->d_yout[i];

		rk5_kernel::calc_y_np1<<<grid, block>>>(n_var, i*r_max, dt_try, y_n, dydt, y_np1);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus)
		{
			throw string("rk5_kernel::calc_y_np1 failed");
		}
	}
}

void rungekutta5::call_kernel_calc_error(int n_var)
{
	for (int i = 0; i < 2; i++)
	{
		var_t** dydt = (var_t**)d_dydt;

		rk5_kernel::calc_error<<<grid, block>>>(n_var, i*r_max, dydt, d_err[i]);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus)
		{
			throw string("rk5_kernel::calc_error failed");
		}
	}
}

ttt_t rungekutta5::step()
{
	// Set the kernel launch parameters
	const int n_body_total = ppd->get_ups() ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_total();
	const int n_var_total = NDIM * n_body_total;
	calc_grid(n_var_total, THREADS_PER_BLOCK);

	// Calculate initial differentials and store them into d_f[][0] = f1(tn, yn)
	int r = 0;
	ttt_t ttemp = ppd->t + c[r] * dt_try;
	for (int i = 0; i < 2; i++)
	{
		ppd->calc_dydx(i, r, ttemp, ppd->sim_data->d_y[0], ppd->sim_data->d_y[1], d_f[i][r]);
	}

	var_t max_err = 0.0;
	int iter = 0;
	do
	{
		dt_did = dt_try;
		// Calculate f2 = f(tn + c2 * dt, yn + a21 * dt * f1) = d_f[][1]
		// ...
		// Calculate f5 = f(tn + c5 * dt, yn + a51 * dt * f1 + ...) = d_f[][4]
		for (r = 1; r < RKOrder; r++)
		{
			ttemp = ppd->t + c[r] * dt_try;
			call_kernel_calc_ytemp(n_var_total, r);
			for (int i = 0; i < 2; i++)
			{
				ppd->calc_dydx(i, r, ttemp, ytemp[0], ytemp[1], d_f[i][r]);
			}
		}
		call_kernel_calc_y_np1(n_var_total);

		if (adaptive)
		{
			for (r = RKOrder; r < r_max; r++)
			{
				ttemp = ppd->t + c[r] * dt_try;
				call_kernel_calc_ytemp(n_var_total, r);
				for (int i = 0; i < 2; i++)
				{
					ppd->calc_dydx(i, r, ttemp, d_ytemp[0], d_ytemp[1], d_f[i][r]);
				}
			}

			int n_var = 0;
			if (ppd->get_ups())
			{
				n_var = NDIM * (error_check_for_tp ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_prime_massive());
			}
			else
			{
				n_var = NDIM * (error_check_for_tp ? ppd->n_bodies->get_n_total() : ppd->n_bodies->get_n_massive());
			}
			call_kernel_calc_error(n_var);
			max_err = get_max_error(n_var);
			dt_try *= 0.9 * pow(tolerance / max_err, 1.0/RKOrder);

			if (ppd->get_n_event() > 0)
			{
				if (dt_try < dt_did)
				{
					dt_try = dt_did;
				}
				break;
			}
		}
		iter++;
	} while (adaptive && max_err > tolerance);

	update_counters(iter);

	ppd->t += dt_did;
	for (int i = 0; i < 2; i++)
	{
		swap(ppd->sim_data->d_yout[i], ppd->sim_data->d_y[i]);
	}

	return dt_did;
}

var_t rungekutta5::get_max_error(int n_var)
{
	// Wrap raw pointer with a device_ptr
	thrust::device_ptr<var_t> d_ptr_r(d_err[0]);
	thrust::device_ptr<var_t> d_ptr_v(d_err[1]);

	// Use thrust to find the maximum element
	thrust::device_ptr<var_t> d_ptr_max_r = thrust::max_element(d_ptr_r, d_ptr_r + n_var);
	thrust::device_ptr<var_t> d_ptr_max_v = thrust::max_element(d_ptr_v, d_ptr_v + n_var);

	// Get the index of the maximum element
	int64_t idx_max_err_r = d_ptr_max_r.get() - d_ptr_r.get();
	int64_t idx_max_err_v = d_ptr_max_v.get() - d_ptr_v.get();

	var_t max_err_r = 0.0;
	var_t max_err_v = 0.0;
	// Copy the max element from device memory to host memory
	cudaMemcpy((void*)&max_err_r, (void*)d_ptr_max_r.get(), sizeof(var_t), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)&max_err_v, (void*)d_ptr_max_v.get(), sizeof(var_t), cudaMemcpyDeviceToHost);

	return fabs(dt_try * std::max(max_err_r, max_err_v));
}
