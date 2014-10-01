// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes Thrust
#include "thrust\device_ptr.h"
#include "thrust\fill.h"
#include "thrust\extrema.h"

// includes project
#include "int_rungekutta8.h"
#include "number_of_bodies.h"
#include "nbody_exception.h"
#include "red_macro.h"
#include "red_constants.h"
#include "util.h"




ttt_t rungekutta8::c[] =  { 0.0, 2.0/27.0, 1.0/9.0, 1.0/6.0, 5.0/12.0, 1.0/2.0, 5.0/6.0, 1.0/6.0, 2.0/3.0, 1.0/3.0, 1.0, 0.0, 1.0 };

var_t rungekutta8::a[] =  {	         0.0, 
					    2.0/27.0,
					    1.0/36.0,    1.0/12.0,
					    1.0/24.0,         0.0,   1.0/8.0,
					    5.0/12.0,         0.0, -25.0/16.0,   25.0/16.0,
					    1.0/20.0,         0.0,        0.0,    1.0/4.0,      1.0/5.0,
					  -25.0/108.0,        0.0,        0.0,  125.0/108.0,  -65.0/27.0,    125.0/54.0,
					   31.0/300.0,        0.0,        0.0,          0.0,   61.0/225.0,    -2.0/9.0,    13.0/900.0,
					          2.0,        0.0,        0.0,  -53.0/6.0,    704.0/45.0,   -107.0/9.0,    67.0/90.0,    3.0,
					  -91.0/108.0,        0.0,        0.0,   23.0/108.0, -976.0/135.0,   311.0/54.0,  -19.0/60.0,   17.0/6.0,  -1.0/12.0,
					 2383.0/4100.0,       0.0,        0.0, -341.0/164.0, 4496.0/1025.0, -301.0/82.0, 2133.0/4100.0, 45.0/82.0, 45.0/164.0, 18.0/41.0,
					    3.0/205.0,        0.0,        0.0,          0.0,           0.0,   -6.0/41.0,   -3.0/205.0,  -3.0/41.0,  3.0/41.0,   6.0/41.0, 0.0,
					-1777.0/4100.0,       0.0,        0.0, -341.0/164.0, 4496.0/1025.0, -289.0/82.0, 2193.0/4100.0, 51.0/82.0, 33.0/164.0, 12.0/41.0, 0.0, 1.0 };

var_t rungekutta8::b[]  = { 41.0/840.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0, 9.0/35.0, 9.0/280.0, 9.0/280.0, 41.0/840.0 };
var_t rungekutta8::bh[] = { 41.0/840.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0, 9.0/35.0, 9.0/280.0, 9.0/280.0, 41.0/840.0, 41.0/840.0, 41.0/840.0 };

// ytemp = yn + dt*(a10*f0)
static __global__
	void kernel_calc_ytemp_for_f1(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, var_t a10)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a10*f0[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a20*f0 + a21*f1)
static __global__
	void kernel_calc_ytemp_for_f2(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f1, var_t a20, var_t a21)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a20*f0[tid] + a21*f0[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a30*f0 + a32*f2)
static __global__
	void kernel_calc_ytemp_for_f3(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f2, var_t a30, var_t a32)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a30*f0[tid] + a32*f2[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a40*f0 + a42*f2 + a43*f3)
static __global__
	void kernel_calc_ytemp_for_f4(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f2, const var_t *f3, var_t a40, var_t a42, var_t a43)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a40*f0[tid] + a42*f2[tid] + a43*f3[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a50*f0 + a53*f3 + a54*f4)
static __global__
	void kernel_calc_ytemp_for_f5(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, var_t a50, var_t a53, var_t a54)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a50*f0[tid] + a53*f3[tid] + a54*f4[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a60*f0 + a63*f3 + a64*f4 + a65*f5)
static __global__
	void kernel_calc_ytemp_for_f6(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, var_t a60, var_t a63, var_t a64, var_t a65)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a60*f0[tid] + a63*f3[tid] + a64*f4[tid] + a65*f5[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a70*f0 + a74*f4 + a75*f5 + a76*f6)
static __global__
	void kernel_calc_ytemp_for_f7(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f4, const var_t *f5, const var_t *f6, var_t a70, var_t a74, var_t a75, var_t a76)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a70*f0[tid] + a74*f4[tid] + a75*f5[tid] + a76*f6[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a80*f0 + a83*f3 + a84*f4 + a85*f5 + a86*f6 + a87*f7)
static __global__
	void kernel_calc_ytemp_for_f8(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, var_t a80, var_t a83, var_t a84, var_t a85, var_t a86, var_t a87)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a80*f0[tid] + a83*f3[tid] + a84*f4[tid] + a85*f5[tid] + a86*f6[tid] + a87*f7[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a90*f0 + a93*f3 + a94*f4 + a95*f5 + a96*f6 + a97*f7 + a98*f8)
static __global__
	void kernel_calc_ytemp_for_f9(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, var_t a90, var_t a93, var_t a94, var_t a95, var_t a96, var_t a97, var_t a98)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a90*f0[tid] + a93*f3[tid] + a94*f4[tid] + a95*f5[tid] + a96*f6[tid] + a97*f7[tid] + a98*f8[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a100*f0 + a103*f3 + a104*f4 + a105*f5 + a106*f6 + a107*f7 + a108*f8 + a109*f9)
static __global__
	void kernel_calc_ytemp_for_f10(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, var_t a100, var_t a103, var_t a104, var_t a105, var_t a106, var_t a107, var_t a108, var_t a109)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a100*f0[tid] + a103*f3[tid] + a104*f4[tid] + a105*f5[tid] + a106*f6[tid] + a107*f7[tid] + a108*f8[tid] + a109*f9[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a110*f0 + a115*f5 + a116*f6 + a117*f7 + a118*f8 + a119*f9)
static __global__
	void kernel_calc_ytemp_for_f11(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, var_t a110, var_t a115, var_t a116, var_t a117, var_t a118, var_t a119)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a110*f0[tid] + a115*f5[tid] + a116*f6[tid] + a117*f7[tid] + a118*f8[tid] + a119*f9[tid]);
// DEBUG CODE
		//printf("%2d\n%25.15le\n%25.15le\n%25.15le\n%25.15le\n%25.15le\n%25.15le\n%25.15le\n%25.15le\n%25.15le\n%25.15le\n%25.15le\n%25.15le\n%25.15le\n%25.15le\n%25.15le\n",
		//	tid, y_n[tid], dt, a110, f0[tid], a115, f5[tid], a116, f6[tid], a117, f7[tid], a118, f8[tid], a119, f9[tid], ytemp[tid]);
// END DEBUG CODE
		tid += stride;
	}
}

// ytemp = yn + dt*(a120*f0 + a123*f3 + a124*f4 + a125*f5 + a126*f6 + a127*f7 + a128*f8 + a129*f9 + a1211*f11)
static __global__
	void kernel_calc_ytemp_for_f12(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, const var_t *f11, var_t a120, var_t a123, var_t a124, var_t a125, var_t a126, var_t a127, var_t a128, var_t a129, var_t a1211)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a120*f0[tid] + a123*f3[tid] + a124*f4[tid] + a125*f5[tid] + a126*f6[tid] + a127*f7[tid] + a128*f8[tid] + a129*f9[tid] + f11[tid]);
		tid += stride;
	}
}

// For the scaling used to monitor accuracy
#define TINY	1.0e-30
// yscale = |y_n| + |dt * f0| + TINY
static __global__
	void kernel_calc_yscale(int_t n, var_t *yscale, ttt_t dt, const var_t *y_n, const var_t *f0)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		yscale[tid] = fabs(y_n[tid]) + fabs(dt * f0[tid]) + TINY;
		tid += stride;
	}
}
#undef TINY

// err = f0 + f10 - f11 - f12
static __global__
	void kernel_calc_error(int_t n, var_t *err, const var_t *f0, const var_t *f10, const var_t *f11, const var_t *f12)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		err[tid] = (f0[tid] + f10[tid] - f11[tid] - f12[tid]);
		tid += stride;
	}
}

// err = f0 + f10 - f11 - f12
static __global__
	void kernel_calc_scalederror(int_t n, var_t *err, ttt_t dt, const var_t* yscale, const var_t *f0, const var_t *f10, const var_t *f11, const var_t *f12)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	var_t s = 41.0/840.0 * fabs(dt);
	while (n > tid) {
		err[tid] = (s * fabs(f0[tid] + f10[tid] - f11[tid] - f12[tid])) / yscale[tid];
		tid += stride;
	}
}

// y_n+1 = yn + dt*(b0*f0 + b5*f5 + b6*f6 + b7*f7 + b8*f8 + b9*f9 + b10*f10)
static __global__
	void kernel_calc_y_np1(int_t n, var_t *y_np1, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, const var_t *f10, var_t b0, var_t b5, var_t b6, var_t b7, var_t b8, var_t b9, var_t b10)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		y_np1[tid] = y_n[tid] + dt * (b0*f0[tid] + b5*f5[tid] + b6*(f6[tid] + f7[tid]) + b8*(f8[tid] + f9[tid]) + b10*f10[tid]);
		tid += stride;
	}
}

rungekutta8::rungekutta8(pp_disk *ppd, ttt_t dt, bool adaptive, var_t tolerance) :
	integrator(ppd, dt),
	adaptive(adaptive),
	tolerance(tolerance),
	d_f(2),
	d_err(2),
	d_yscale(2)
{
	const int n = ppd->n_bodies->total;
	const int n_var = NDIM * n;
	name = "Runge-Kutta-Fehlberg8";

	t = ppd->t;
	RKOrder = 7;
	r_max = adaptive ? RKOrder + 6 : RKOrder + 4;
	for (int i = 0; i < 2; i++)
	{
		ALLOCATE_DEVICE_VECTOR((void**) &(d_ytemp[i]), n * sizeof(vec_t));
		d_f[i].resize(r_max);
		for (int r = 0; r < r_max; r++) 
		{
			ALLOCATE_DEVICE_VECTOR((void**) &(d_f[i][r]), n * sizeof(vec_t));
		}
		if (adaptive)
		{
			ALLOCATE_DEVICE_VECTOR((void**) &(d_err[i]), n_var * sizeof(var_t));
			ALLOCATE_DEVICE_VECTOR((void**) &(d_yscale[i]), n_var * sizeof(var_t));
		}
	}
}

rungekutta8::~rungekutta8()
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
			cudaFree(d_yscale[i]);
		}
	}
}

void rungekutta8::call_calc_ytemp_for_fr_kernel(int r)
{
	int idx = 0;

	const int n_var = 4 * ppd->n_bodies->total;
	calc_grid(n_var, THREADS_PER_BLOCK);

	for (int i = 0; i < 2; i++) {

		var_t* y_n   = (var_t*)ppd->sim_data->d_y[i];
		var_t* ytemp = (var_t*)d_ytemp[i];

		var_t* f0 = (var_t*)d_f[i][0];
		var_t* f1 = (var_t*)d_f[i][1];
		var_t* f2 = (var_t*)d_f[i][2];
		var_t* f3 = (var_t*)d_f[i][3];
		var_t* f4 = (var_t*)d_f[i][4];
		var_t* f5 = (var_t*)d_f[i][5];
		var_t* f6 = (var_t*)d_f[i][6];
		var_t* f7 = (var_t*)d_f[i][7];
		var_t* f8 = (var_t*)d_f[i][8];
		var_t* f9 = (var_t*)d_f[i][9];
		var_t* f10= (var_t*)d_f[i][10];
		var_t* f11;

		if (adaptive) 
		{
			f11	= (var_t*)d_f[i][11];
		}

		switch (r) {
		case 1:
			idx = 1;		
			kernel_calc_ytemp_for_f1<<<grid, block>>>(n_var, ytemp, dt_try, y_n, f0, a[idx]);
			break;
		case 2:
			idx = 2;
			kernel_calc_ytemp_for_f2<<<grid, block>>>(n_var, ytemp, dt_try, y_n, f0, f1, a[idx], a[idx+1]);
			break;
		case 3:
			idx = 4;
			kernel_calc_ytemp_for_f3<<<grid, block>>>(n_var, ytemp, dt_try, y_n, f0, f2, a[idx], a[idx+2]);
			break;
		case 4:
			idx = 7;
			kernel_calc_ytemp_for_f4<<<grid, block>>>(n_var, ytemp, dt_try, y_n, f0, f2, f3, a[idx], a[idx+2], a[idx+3]);
			break;
		case 5:
			idx = 11;
			kernel_calc_ytemp_for_f5<<<grid, block>>>(n_var, ytemp, dt_try, y_n, f0, f3, f4, a[idx], a[idx+3], a[idx+4]);
			break;
		case 6:
			idx = 16;
			kernel_calc_ytemp_for_f6<<<grid, block>>>(n_var, ytemp, dt_try, y_n, f0, f3, f4, f5, a[idx], a[idx+3], a[idx+4], a[idx+5]);
			break;
		case 7:
			idx = 22;
			kernel_calc_ytemp_for_f7<<<grid, block>>>(n_var, ytemp, dt_try, y_n, f0, f4, f5, f6, a[idx], a[idx+4], a[idx+5], a[idx+6]);
			break;
		case 8:
			idx = 29;
			kernel_calc_ytemp_for_f8<<<grid, block>>>(n_var, ytemp, dt_try, y_n, f0, f3, f4, f5, f6, f7, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7]);
			break;
		case 9:
			idx = 37;
			kernel_calc_ytemp_for_f9<<<grid, block>>>(n_var, ytemp, dt_try, y_n, f0, f3, f4, f5, f6, f7, f8, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7], a[idx+8]);
			break;
		case 10:
			idx = 46;
			kernel_calc_ytemp_for_f10<<<grid, block>>>(n_var, ytemp, dt_try, y_n, f0, f3, f4, f5, f6, f7, f8, f9, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7], a[idx+8], a[idx+9]);
			break;
		case 11:
			idx = 56;
			kernel_calc_ytemp_for_f11<<<grid, block>>>(n_var, ytemp, dt_try, y_n, f0, f5, f6, f7, f8, f9, a[idx], a[idx+5], a[idx+6], a[idx+7], a[idx+8], a[idx+9]);
// DEBUG CODE
			//cudaDeviceSynchronize();
// END DEBUG CODE
			break;
		case 12:
			idx = 67;
			kernel_calc_ytemp_for_f12<<<grid, block>>>(n_var, ytemp, dt_try, y_n, f0, f3, f4, f5, f6, f7, f8, f9, f11, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7], a[idx+8], a[idx+9], a[idx+11]);
			break;
		default:
			throw string("call_calc_ytemp_for_fr_kernel: parameter out of range.");
		}
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw string("call_calc_ytemp_for_fr_kernel failed");
		}
	}
}

void rungekutta8::call_calc_yscale_kernel()
{
	const int n_var = 4 * ppd->n_bodies->total;
	calc_grid(n_var, THREADS_PER_BLOCK);

	for (int i = 0; i < 2; i++) {
		var_t* yscale = d_yscale[i];
		var_t* y_n    = (var_t*)ppd->sim_data->d_y[i];
		var_t* f0     = (var_t*)d_f[i][0];

		kernel_calc_yscale<<<grid, block>>>(n_var, yscale, dt_try, y_n, f0);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw string("kernel_calc_yscale failed");
		}
	}
}

void rungekutta8::call_calc_error_kernel()
{
	const int n_var = 4 * ppd->n_bodies->total;
	calc_grid(n_var, THREADS_PER_BLOCK);

	for (int i = 0; i < 2; i++) {
		var_t *err = d_err[i];
		var_t *f0  = (var_t*)d_f[i][0];
		var_t *f10 = (var_t*)d_f[i][10];
		var_t *f11 = (var_t*)d_f[i][11];
		var_t *f12 = (var_t*)d_f[i][12];

		kernel_calc_error<<<grid, block>>>(n_var, err, f0, f10, f11, f12);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw string("kernel_calc_error failed");
		}
	}
}

void rungekutta8::call_calc_scalederror_kernel()
{
	const int n_var = 4 * ppd->n_bodies->total;
	calc_grid(n_var, THREADS_PER_BLOCK);

	for (int i = 0; i < 2; i++) {
		var_t *err = d_err[i];
		var_t *yscale= (var_t*)d_yscale[i];
		var_t *f0  = (var_t*)d_f[i][0];
		var_t *f10 = (var_t*)d_f[i][10];
		var_t *f11 = (var_t*)d_f[i][11];
		var_t *f12 = (var_t*)d_f[i][12];

		kernel_calc_scalederror<<<grid, block>>>(n_var, err, dt_try, yscale, f0, f10, f11, f12);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw string("kernel_calc_scalederror failed");
		}
	}
}

void rungekutta8::call_calc_y_np1_kernel()
{
	const int n_var = 4 * ppd->n_bodies->total;
	calc_grid(n_var, THREADS_PER_BLOCK);

	for (int i = 0; i < 2; i++) {
		var_t* y_n   = (var_t*)ppd->sim_data->d_y[i];
		var_t* y_np1 = (var_t*)ppd->sim_data->d_yout[i];

		var_t *f0  = (var_t*)d_f[i][0];
		var_t *f5  = (var_t*)d_f[i][5];
		var_t *f6  = (var_t*)d_f[i][6];
		var_t *f7  = (var_t*)d_f[i][7];
		var_t *f8  = (var_t*)d_f[i][8];
		var_t *f9  = (var_t*)d_f[i][9];
		var_t *f10 = (var_t*)d_f[i][10];

		kernel_calc_y_np1<<<grid, block>>>(n_var, y_np1, dt_try, y_n, f0, f5, f6, f7, f8, f9, f10, b[0], b[5], b[6], b[7], b[8], b[9], b[10]);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw string("calc_y_np1_kernel failed");
		}
	}
}

// DEBUG CODE
void print_vector(int n, const char* name, const var_t *v)
{
	for (int i = 0; i < n; i++)
	{
		printf("%10s[%2d]: %25.15le\n", name, i, v[i]);
	}
}
// END DEBUG CODE


ttt_t rungekutta8::step()
{
	const int n_var = NDIM * ppd->n_bodies->total;

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
		// Calculate f2 = f(tn + c1 * dt, yn + a10 * dt * f0) = d_f[][1]
		// ...
		// Calculate f11 = f(tn + c10 * dt, yn + a10,0 * dt * f0 + ...) = d_f[][10]
		for (r = 1; r <= 10; r++) {
			ttemp = ppd->t + c[r] * dt_try;
			call_calc_ytemp_for_fr_kernel(r);
			for (int i = 0; i < 2; i++) {
				ppd->calc_dy(i, r, ttemp, d_ytemp[0], d_ytemp[1], d_f[i][r]);
			}
		}
	
		// y_(n+1) = yn + dt*(b0*f0 + b5*f5 + b6*f6 + b7*f7 + b8*f8 + b9*f9 + b10*f10) + O(dt^8)
		call_calc_y_np1_kernel();

		if (adaptive) {
			// Calculate f11 = f(tn + c11 * dt, yn + ...) = d_f[][11]
			// Calculate f12 = f(tn + c11 * dt, yn + ...) = d_f[][12]
			for (r = 11; r < r_max; r++) {
				ttemp = ppd->t + c[r] * dt_try;
				call_calc_ytemp_for_fr_kernel(r);
				for (int i = 0; i < 2; i++) {
					ppd->calc_dy(i, r, ttemp, d_ytemp[0], d_ytemp[1], d_f[i][r]);
				}
			}
			call_calc_error_kernel();
			max_err = get_max_error(n_var);
			dt_try *= 0.9 * pow(tolerance / max_err, 1.0/8.0);

			// Were there collisions?
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

var_t rungekutta8::get_max_error(int n_var)
{
	// Wrap raw pointer with a device_ptr
	thrust::device_ptr<var_t> d_ptr_r(d_err[0]);
	thrust::device_ptr<var_t> d_ptr_v(d_err[1]);

	// DEBUG CODE
	var_t* h_ptr_r = (var_t*)malloc(n_var * sizeof(var_t));
	var_t* h_ptr_v = (var_t*)malloc(n_var * sizeof(var_t));

	cudaMemcpy((void*)h_ptr_r, (void*)d_ptr_r.get(), n_var * sizeof(var_t), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)h_ptr_v, (void*)d_ptr_v.get(), n_var * sizeof(var_t), cudaMemcpyDeviceToHost);
	// END DEBUG CODE

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

	return fabs(dt_try * 41.0/840.0 * std::max(max_err_r, max_err_v));
}
