// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes project
#include "int_rungekutta4.h"
#include "number_of_bodies.h"
#include "nbody_exception.h"
#include "red_macro.h"
#include "red_constants.h"

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

// ytemp = y_n + a*fr, r = 2, 3, 4
static __global__
	void kernel_calc_ytemp_for_fr(int_t n, var_t *ytemp, const var_t *y_n, const var_t *fr, var_t a)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + a * fr[tid];
		tid += stride;
	}
}

static __global__
	void kernel_calc_yHat(int_t n, var_t *y_hat, const var_t *y_n, const var_t *f1, const var_t *f2, const var_t *f3, const var_t *f4, var_t b0, var_t b1)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		y_hat[tid] = y_n[tid] + b0 * (f1[tid] + f4[tid]) + b1 * (f2[tid] + f3[tid]);
		tid += stride;
	}
}

namespace integrator
{
#define	LAMBDA	1.0/10.0

ttt_t rungekutta4::c[] =  {0.0, 1.0/2.0, 1.0/2.0, 1.0, 1.0};
var_t rungekutta4::a[] =  {0.0, 1.0/2.0, 1.0/2.0, 1.0, 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0};
var_t rungekutta4::bh[] = {1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0, 0.0};
var_t rungekutta4::b[] =  {1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0 -LAMBDA, LAMBDA};

rungekutta4::rungekutta4(ttt_t t0, ttt_t dt, bool adaptive, var_t tolerance, pp_disk *ppd) :
	name("Runge-Kutta4"),
	d_f(2),
	d_ytemp(2),
	d_err(2),
	t(t0),
	adaptive(adaptive),
	tolerance(tolerance),
	dt_try(dt),
	dt_did(0.0),
	dt_next(0.0),
	ppd(ppd)
{
	const int n = ppd->n_bodies->total;
	cudaError_t cudaStatus;

	RKOrder = 4;
	r_max = adaptive ? RKOrder + 1 : RKOrder;

	// Allocate device pointer.
	for (int i = 0; i < 2; i++)
	{
		cudaMalloc((void**) &(d_ytemp[i]), n * sizeof(vec_t));
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("cudaMalloc failed", cudaStatus);
		}

		d_f[i].resize(r_max);
		for (int r = 0; r < r_max; r++) 
		{
			cudaMalloc((void**) &(d_f[i][r]), n * sizeof(vec_t));
			cudaStatus = HANDLE_ERROR(cudaGetLastError());
			if (cudaSuccess != cudaStatus) {
				throw nbody_exception("cudaMalloc failed", cudaStatus);
			}
		}

		if (adaptive)
		{
			cudaMalloc((void**) &(d_err[i]), n * sizeof(vec_t));
			cudaStatus = HANDLE_ERROR(cudaGetLastError());
			if (cudaSuccess != cudaStatus) {
				throw nbody_exception("cudaMalloc failed", cudaStatus);
			}
		}
	}
}

rungekutta4::~rungekutta4()
{
	for (int i = 0; i < 2; i++)
	{
		cudaFree(d_ytemp[i]);
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

void rungekutta4::calc_grid(int nData, int threads_per_block)
{
	int	nThread = std::min(threads_per_block, nData);
	int	nBlock = (nData + nThread - 1)/nThread;
	grid.x  = nBlock;
	block.x = nThread;
}

void rungekutta4::call_kernel_calc_ytemp_for_fr(int r)
{
	const int n_var = 4 * ppd->n_bodies->total;
	for (int i = 0; i < 2; i++) {
		var_t *y_n = (i == 0 ? (var_t*)ppd->sim_data->d_y[0] : (var_t*)ppd->sim_data->d_y[1]);
		var_t *fr	= (var_t*)d_f[i][r-1];

		calc_grid(n_var, THREADS_PER_BLOCK);
		kernel_calc_ytemp_for_fr<<<grid, block>>>(n_var, (var_t*)d_ytemp[i], y_n, fr, a[r] * dt_try);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw string("kernel_calc_ytemp_for_fr failed");
		}
	}
}

void rungekutta4::call_kernel_calc_yHat()
{
	const int n_var = 4 * ppd->n_bodies->total;

	for (int i = 0; i < 2; i++) {

		var_t *y_n	 = (i == 0 ? (var_t*)ppd->sim_data->d_y[0] : (var_t*)ppd->sim_data->d_y[1]);
		var_t *y_Hat = (i == 0 ? (var_t*)ppd->sim_data->d_yout[0] : (var_t*)ppd->sim_data->d_yout[1]);
		var_t *f1	 = (var_t*)d_f[i][0];
		var_t *f2	 = (var_t*)d_f[i][1];
		var_t *f3	 = (var_t*)d_f[i][2];
		var_t *f4	 = (var_t*)d_f[i][3];

		calc_grid(n_var, THREADS_PER_BLOCK);
		kernel_calc_yHat<<<grid, block>>>(n_var, y_Hat, y_n, f1, f2, f3, f4, b[0] * dt_try, b[1] * dt_try);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw string("calc_yHat_kernel failed");
		}
	}
}

ttt_t rungekutta4::step()
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
		// Calculate f3 = f(tn + c3 * dt, yn + a31 * dt * f2) = d_f[][2]
		// Calculate f4 = f(tn + c4 * dt, yn + a41 * dt * f3) = d_f[][3]
		for (r = 1; r < RKOrder; r++) {
			ttemp = ppd->t + c[r] * dt_try;
			call_kernel_calc_ytemp_for_fr(r);
			for (int i = 0; i < 2; i++) {
				ppd->calc_dy(i, r, ttemp, d_ytemp[0], d_ytemp[1], d_f[i][r]);
			}
		}
		// yHat_(n+1) = yn + dt*(1/6*f1 + 1/3*f2 + 1/3*f3 + 1/6*f4) + O(dt^5)
		call_kernel_calc_yHat();

		iter++;
	} while (adaptive && max_err > tolerance);


	ppd->t += dt_did;
	swap(ppd->sim_data->d_yout[0], ppd->sim_data->d_y[0]);
	swap(ppd->sim_data->d_yout[1], ppd->sim_data->d_y[1]);

	return dt_did;
}

#undef LAMBDA

} /* namespace integrator */