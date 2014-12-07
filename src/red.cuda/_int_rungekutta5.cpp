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

__constant__ var_t dc_a[sizeof(rungekutta5::a) / sizeof(var_t)];
__constant__ var_t dc_b[sizeof(rungekutta5::b) / sizeof(var_t)];
__constant__ var_t dc_bh[sizeof(rungekutta5::bh) / sizeof(var_t)];
__constant__ var_t dc_c[sizeof(rungekutta5::c) / sizeof(ttt_t)];

static __global__
	void kernel_print_constant_memory(int n, var_t *src)
{
	for (int i = 0; i < n; i++)
	{
		printf("const[%d]: %20.16lf\n", i, src[i]);
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

static __global__
	void kernel_calc_yHat(int n, const var_t *y_n, const var_t *f1, const var_t *f2, const var_t *f3, const var_t *f4, var_t b0, var_t b1, var_t *y_hat)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		y_hat[tid] = y_n[tid] + b0 * (f1[tid] + f4[tid]) + b1 * (f2[tid] + f3[tid]);
		tid += stride;
	}
}

static __global__
	void kernel_calc_f4_sub_f5(int n, const var_t *f4, const var_t* f5, var_t *result)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		result[tid] = f4[tid] - f5[tid];
		tid += stride;
	}
}


rungekutta5::rungekutta5(pp_disk *ppd, ttt_t dt, bool adaptive, var_t tolerance) :
	integrator(ppd, dt),
	adaptive(adaptive),
	tolerance(tolerance),
	d_f(2),
	d_err(2)
{
	name = "Runge-Kutta5";

	const int n_total = ppd->get_ups() ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_total();

	t = ppd->t;
	RKOrder = 5;
	r_max = adaptive ? RKOrder + 1 : RKOrder;
	for (int i = 0; i < 2; i++)
	{
		ALLOCATE_DEVICE_VECTOR((void**) &(d_ytemp[i]), n_total * sizeof(vec_t));
		d_f[i].resize(r_max);
		for (int r = 0; r < r_max; r++) 
		{
			ALLOCATE_DEVICE_VECTOR((void**) &(d_f[i][r]), n_total * sizeof(vec_t));
		}
		if (adaptive)
		{
			static const int n_var = NDIM * n_total;
			ALLOCATE_DEVICE_VECTOR((void**) &(d_err[i]), n_var * sizeof(var_t));
		}
	}

	size_t n = sizeof(a);
	copy_constant_to_device(dc_a,  a, n);

	kernel_print_constant_memory<<<1,1>>>(n, dc_a);
	cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("call_kernel_calc_ytemp_for_fr failed");
	}
	cudaDeviceSynchronize();

	n = sizeof(b);
	copy_constant_to_device(dc_b,  b,  n);

	n = sizeof(bh);
	copy_constant_to_device(dc_bh, bh, n);

	n = sizeof(c);
	copy_constant_to_device(dc_c,  c,  n);
}

rungekutta5::~rungekutta5()
{
	for (int i = 0; i < 2; i++)
	{
		for (int r = 0; r < r_max; r++) 
		{
			cudaFree(d_f[i][r]);
		}
		if (adaptive)
		{
			cudaFree(d_err[i]);
		}
	}
}

void rungekutta5::call_kernel_calc_ytemp(int r)
{
	static int idx_array[] = {0, 1, 2, 4, 7, 11};

	const int n_total = ppd->get_ups() ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_total();
	const int n_var = NDIM * n_total;

	calc_grid(n_var, THREADS_PER_BLOCK);

	int idx = idx_array[r];
	var_t a0 = a[idx];
	var_t a0 = a[idx];
	var_t a0 = a[idx];
	var_t a0 = a[idx];
	var_t a0 = a[idx];
	for (int i = 0; i < 2; i++)
	{
		var_t *y_n = (var_t*)ppd->sim_data->d_y[i];
		var_t *f1  = (var_t*)d_f[i][0];
		var_t *f2  = (var_t*)d_f[i][1];
		var_t *f3  = (var_t*)d_f[i][2];
		var_t *f4  = (var_t*)d_f[i][3];
		var_t *f5  = (var_t*)d_f[i][4];
		var_t *f6  = (var_t*)d_f[i][5];
		var_t* ytemp = (var_t*)d_ytemp[i];

		kernel_sum_vector<<<grid, block>>>(n_var, y_n, r, f0, f1, f2, f3, f4, f5, a[idx] * dt_try, a[idx] * dt_try, a[idx] * dt_try, a[idx] * dt_try, a[idx] * dt_try, ytemp);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus)
		{
			throw string("kernel_calc_ytemp_for_fr failed");
		}
	}
}

void rungekutta5::call_kernel_calc_yHat()
{
	const int n_total = ppd->get_ups() ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_total();
	const int n_var = NDIM * n_total;

	for (int i = 0; i < 2; i++) {
		var_t *y_n	 = (var_t*)ppd->sim_data->d_y[i];
		var_t *y_Hat = (var_t*)ppd->sim_data->d_yout[i];
		var_t *f1	 = (var_t*)d_f[i][0];
		var_t *f2	 = (var_t*)d_f[i][1];
		var_t *f3	 = (var_t*)d_f[i][2];
		var_t *f4	 = (var_t*)d_f[i][3];

		calc_grid(n_var, THREADS_PER_BLOCK);
		kernel_calc_yHat<<<grid, block>>>(n_var, y_n, f1, f2, f3, f4, b[0] * dt_try, b[1] * dt_try, y_Hat);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw string("calc_yHat_kernel failed");
		}
	}
}

void rungekutta5::call_kernel_calc_error()
{
	const int n_total = ppd->get_ups() ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_total();
	const int n_var = NDIM * n_total;

	for (int i = 0; i < 2; i++) {
		var_t *f4  = (var_t*)d_f[i][3];
		var_t *f5  = (var_t*)d_f[i][4];

		calc_grid(n_var, THREADS_PER_BLOCK);
		kernel_calc_f4_sub_f5<<<grid, block>>>(n_var, f4, f5, d_err[i]);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw string("kernel_calc_f4_sub_f5 failed");
		}
	}
}

ttt_t rungekutta5::step()
{
	// Calculate initial differentials and store them into d_f[][0]
	int r = 0;
	ttt_t ttemp = ppd->t + c[r] * dt_try;
	// Calculate f1 = f(tn, yn) = d_f[][0]
	for (int i = 0; i < 2; i++)
	{
		ppd->calc_dy(i, r, ttemp, ppd->sim_data->d_y[0], ppd->sim_data->d_y[1], d_f[i][r]);
	}

	var_t max_err = 0.0;
	int iter = 0;
	do
	{
		dt_did = dt_try;
		// Calculate f2 = f(tn + c2 * dt, yn + a21 * dt * f1) = d_f[][1]
		// ...
		// Calculate f5 = f(tn + c4 * dt, yn + a41 * dt * f3) = d_f[][4]
		for (r = 1; r < RKOrder; r++)
		{
			ttemp = ppd->t + c[r] * dt_try;
			call_kernel_calc_ytemp_for_fr(r);
			for (int i = 0; i < 2; i++)
			{
				ppd->calc_dy(i, r, ttemp, d_ytemp[0], d_ytemp[1], d_f[i][r]);
			}
		}
		// yHat_(n+1) = yn + dt*(1/6*f1 + 1/3*f2 + 1/3*f3 + 1/6*f4) + O(dt^5)
		call_kernel_calc_yHat();

		if (adaptive)
		{
			r = 4;
			ttemp = ppd->t + c[r] * dt_try;
			// Calculate f5 = f(tn + c5 * dt,  yn + dt*(1/6*f1 + 1/3*f2 + 1/3*f3 + 1/6*f4)) = d_f[][4]
			for (int i = 0; i < 2; i++) {
				ppd->calc_dy(i, r, ttemp, ppd->sim_data->d_yout[0], ppd->sim_data->d_yout[1], d_f[i][r]);
			}
			// calculate: d_err = h(f4 - f5)
			call_kernel_calc_error();

			int n_var = 0;
			if (ppd->get_ups())
			{
				n_var = NDIM * (error_check_for_tp ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_prime_massive());
			}
			else
			{
				n_var = NDIM * (error_check_for_tp ? ppd->n_bodies->get_n_total() : ppd->n_bodies->get_n_massive());
			}
			max_err = get_max_error(n_var);
			dt_try *= 0.9 * pow(tolerance / max_err, 1.0/4.0);

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

	return fabs(dt_try * LAMBDA * std::max(max_err_r, max_err_v));
}
