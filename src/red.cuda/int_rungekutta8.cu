// includes system
#include <sstream>		// ostringstream

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
#include "int_rungekutta8.h"
#include "nbody_exception.h"
#include "red_macro.h"
#include "red_constants.h"
#include "util.h"

#define	LAMBDA	41.0/840.0

using namespace std;

ttt_t rungekutta8::c[] =  { 0.0, 2.0/27.0, 1.0/9.0, 1.0/6.0, 5.0/12.0, 1.0/2.0, 5.0/6.0, 1.0/6.0, 2.0/3.0, 1.0/3.0, 1.0, 0.0, 1.0 };

var_t rungekutta8::a[] =  {  0.0, 
					    2.0/27.0,
					    1.0/36.0,   1.0/12.0,
					    1.0/24.0,        0.0,   1.0/8.0,
					    5.0/12.0,        0.0, -25.0/16.0,   25.0/16.0,
					    1.0/20.0,        0.0,        0.0,    1.0/4.0,      1.0/5.0,
					  -25.0/108.0,       0.0,        0.0,  125.0/108.0,  -65.0/27.0,    125.0/54.0,
					   31.0/300.0,       0.0,        0.0,          0.0,   61.0/225.0,    -2.0/9.0,    13.0/900.0,
					          2.0,       0.0,        0.0,  -53.0/6.0,    704.0/45.0,   -107.0/9.0,    67.0/90.0,    3.0,
					  -91.0/108.0,       0.0,        0.0,   23.0/108.0, -976.0/135.0,   311.0/54.0,  -19.0/60.0,   17.0/6.0,  -1.0/12.0,
					 2383.0/4100.0,      0.0,        0.0, -341.0/164.0, 4496.0/1025.0, -301.0/82.0, 2133.0/4100.0, 45.0/82.0, 45.0/164.0, 18.0/41.0,
					    3.0/205.0,       0.0,        0.0,          0.0,           0.0,   -6.0/41.0,   -3.0/205.0,  -3.0/41.0,  3.0/41.0,   6.0/41.0, 0.0,
					-1777.0/4100.0,      0.0,        0.0, -341.0/164.0, 4496.0/1025.0, -289.0/82.0, 2193.0/4100.0, 51.0/82.0, 33.0/164.0, 12.0/41.0, 0.0, 1.0 };

var_t rungekutta8::b[]  = { 41.0/840.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0, 9.0/35.0, 9.0/280.0, 9.0/280.0, 41.0/840.0 };

var_t rungekutta8::bh[] = {        0.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0, 9.0/35.0, 9.0/280.0, 9.0/280.0, 41.0/840.0, 41.0/840.0, 41.0/840.0 };



ttt_t c_rungekutta8::c[] =  { 0.0, 2.0/27.0, 1.0/9.0, 1.0/6.0, 5.0/12.0, 1.0/2.0, 5.0/6.0, 1.0/6.0, 2.0/3.0, 1.0/3.0, 1.0, 0.0, 1.0 };

var_t c_rungekutta8::a[] =  {  0.0, 
					    2.0/27.0,
					    1.0/36.0,   1.0/12.0,
					    1.0/24.0,        0.0,   1.0/8.0,
					    5.0/12.0,        0.0, -25.0/16.0,   25.0/16.0,
					    1.0/20.0,        0.0,        0.0,    1.0/4.0,      1.0/5.0,
					  -25.0/108.0,       0.0,        0.0,  125.0/108.0,  -65.0/27.0,    125.0/54.0,
					   31.0/300.0,       0.0,        0.0,          0.0,   61.0/225.0,    -2.0/9.0,    13.0/900.0,
					          2.0,       0.0,        0.0,  -53.0/6.0,    704.0/45.0,   -107.0/9.0,    67.0/90.0,    3.0,
					  -91.0/108.0,       0.0,        0.0,   23.0/108.0, -976.0/135.0,   311.0/54.0,  -19.0/60.0,   17.0/6.0,  -1.0/12.0,
					 2383.0/4100.0,      0.0,        0.0, -341.0/164.0, 4496.0/1025.0, -301.0/82.0, 2133.0/4100.0, 45.0/82.0, 45.0/164.0, 18.0/41.0,
					    3.0/205.0,       0.0,        0.0,          0.0,           0.0,   -6.0/41.0,   -3.0/205.0,  -3.0/41.0,  3.0/41.0,   6.0/41.0, 0.0,
					-1777.0/4100.0,      0.0,        0.0, -341.0/164.0, 4496.0/1025.0, -289.0/82.0, 2193.0/4100.0, 51.0/82.0, 33.0/164.0, 12.0/41.0, 0.0, 1.0 };

var_t c_rungekutta8::b[]  = { 41.0/840.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0, 9.0/35.0, 9.0/280.0, 9.0/280.0, 41.0/840.0 };

var_t c_rungekutta8::bh[] = {        0.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0, 9.0/35.0, 9.0/280.0, 9.0/280.0, 41.0/840.0, 41.0/840.0, 41.0/840.0 };


__constant__ var_t dc_a[ sizeof(rungekutta8::a ) / sizeof(var_t)];
__constant__ var_t dc_b[ sizeof(rungekutta8::b ) / sizeof(var_t)];
__constant__ var_t dc_bh[sizeof(rungekutta8::bh) / sizeof(var_t)];
__constant__ var_t dc_c[ sizeof(rungekutta8::c ) / sizeof(ttt_t)];

namespace rk8_kernel
{
// ytemp = yn + dt*(a10*f0)
static __global__
	void calc_ytemp_for_f1(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, var_t a10)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) 
	{
		ytemp[tid] = y_n[tid] + dt * (a10*f0[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a20*f0 + a21*f1)
static __global__
	void calc_ytemp_for_f2(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f1, var_t a20, var_t a21)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		ytemp[tid] = y_n[tid] + dt * (a20*f0[tid] + a21*f1[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a30*f0 + a32*f2)
static __global__
	void calc_ytemp_for_f3(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f2, var_t a30, var_t a32)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		ytemp[tid] = y_n[tid] + dt * (a30*f0[tid] + a32*f2[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a40*f0 + a42*f2 + a43*f3)
static __global__
	void calc_ytemp_for_f4(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f2, const var_t *f3, var_t a40, var_t a42, var_t a43)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		ytemp[tid] = y_n[tid] + dt * (a40*f0[tid] + a42*f2[tid] + a43*f3[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a50*f0 + a53*f3 + a54*f4)
static __global__
	void calc_ytemp_for_f5(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, var_t a50, var_t a53, var_t a54)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		ytemp[tid] = y_n[tid] + dt * (a50*f0[tid] + a53*f3[tid] + a54*f4[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a60*f0 + a63*f3 + a64*f4 + a65*f5)
static __global__
	void calc_ytemp_for_f6(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, var_t a60, var_t a63, var_t a64, var_t a65)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		ytemp[tid] = y_n[tid] + dt * (a60*f0[tid] + a63*f3[tid] + a64*f4[tid] + a65*f5[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a70*f0 + a74*f4 + a75*f5 + a76*f6)
static __global__
	void calc_ytemp_for_f7(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f4, const var_t *f5, const var_t *f6, var_t a70, var_t a74, var_t a75, var_t a76)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		ytemp[tid] = y_n[tid] + dt * (a70*f0[tid] + a74*f4[tid] + a75*f5[tid] + a76*f6[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a80*f0 + a83*f3 + a84*f4 + a85*f5 + a86*f6 + a87*f7)
static __global__
	void calc_ytemp_for_f8(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, var_t a80, var_t a83, var_t a84, var_t a85, var_t a86, var_t a87)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		ytemp[tid] = y_n[tid] + dt * (a80*f0[tid] + a83*f3[tid] + a84*f4[tid] + a85*f5[tid] + a86*f6[tid] + a87*f7[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a90*f0 + a93*f3 + a94*f4 + a95*f5 + a96*f6 + a97*f7 + a98*f8)
static __global__
	void calc_ytemp_for_f9(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, var_t a90, var_t a93, var_t a94, var_t a95, var_t a96, var_t a97, var_t a98)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		ytemp[tid] = y_n[tid] + dt * (a90*f0[tid] + a93*f3[tid] + a94*f4[tid] + a95*f5[tid] + a96*f6[tid] + a97*f7[tid] + a98*f8[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a100*f0 + a103*f3 + a104*f4 + a105*f5 + a106*f6 + a107*f7 + a108*f8 + a109*f9)
static __global__
	void calc_ytemp_for_f10(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, var_t a100, var_t a103, var_t a104, var_t a105, var_t a106, var_t a107, var_t a108, var_t a109)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		ytemp[tid] = y_n[tid] + dt * (a100*f0[tid] + a103*f3[tid] + a104*f4[tid] + a105*f5[tid] + a106*f6[tid] + a107*f7[tid] + a108*f8[tid] + a109*f9[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a110*f0 + a115*f5 + a116*f6 + a117*f7 + a118*f8 + a119*f9)
static __global__
	void calc_ytemp_for_f11(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, var_t a110, var_t a115, var_t a116, var_t a117, var_t a118, var_t a119)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		ytemp[tid] = y_n[tid] + dt * (a110*f0[tid] + a115*f5[tid] + a116*f6[tid] + a117*f7[tid] + a118*f8[tid] + a119*f9[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a120*f0 + a123*f3 + a124*f4 + a125*f5 + a126*f6 + a127*f7 + a128*f8 + a129*f9 + a1211*f11)
static __global__
	void calc_ytemp_for_f12(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, const var_t *f11, var_t a120, var_t a123, var_t a124, var_t a125, var_t a126, var_t a127, var_t a128, var_t a129, var_t a1211)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		ytemp[tid] = y_n[tid] + dt * (a120*f0[tid] + a123*f3[tid] + a124*f4[tid] + a125*f5[tid] + a126*f6[tid] + a127*f7[tid] + a128*f8[tid] + a129*f9[tid] + a1211*f11[tid]);
		tid += stride;
	}
}

// err = f0 + f10 - f11 - f12
static __global__
	void calc_error(int_t n, var_t *err, const var_t *f0, const var_t *f10, const var_t *f11, const var_t *f12)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		err[tid] = fabs(f0[tid] + f10[tid] - f11[tid] - f12[tid]);
		tid += stride;
	}
}

// y_n+1 = yn + dt*(b0*f0 + b5*f5 + b6*f6 + b7*f7 + b8*f8 + b9*f9 + b10*f10)
static __global__
	void calc_y_np1(int_t n, var_t *y_np1, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, const var_t *f10, var_t b0, var_t b5, var_t b6, var_t b7, var_t b8, var_t b9, var_t b10)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid)
	{
		y_np1[tid] = y_n[tid] + dt * (b0*f0[tid] + b5*f5[tid] + b6*(f6[tid] + f7[tid]) + b8*(f8[tid] + f9[tid]) + b10*f10[tid]);
		tid += stride;
	}
}
} /* rk8_kernel */

// ytemp = yn + dt*(a10*f0)
void rungekutta8::cpu_calc_ytemp_for_f1(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, var_t a10)
{
	for (int tid = 0; tid < n; tid++)
	{
		ytemp[tid] = y_n[tid] + dt * (a10*f0[tid]);
	}
}

// ytemp = yn + dt*(a10*f0)
void rungekutta8::cpu_calc_ytemp_for_f2(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f1, var_t a20, var_t a21)
{
	for (int tid = 0; tid < n; tid++)
	{
		ytemp[tid] = y_n[tid] + dt * (a20*f0[tid] + a21*f1[tid]);
	}
}

// ytemp = yn + dt*(a30*f0 + a32*f2)
void rungekutta8::cpu_calc_ytemp_for_f3(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f2, var_t a30, var_t a32)
{
	for (int tid = 0; tid < n; tid++)
	{
		ytemp[tid] = y_n[tid] + dt * (a30*f0[tid] + a32*f2[tid]);
	}
}

// ytemp = yn + dt*(a40*f0 + a42*f2 + a43*f3)
void rungekutta8::cpu_calc_ytemp_for_f4(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f2, const var_t *f3, var_t a40, var_t a42, var_t a43)
{
	for (int tid = 0; tid < n; tid++)
	{
		ytemp[tid] = y_n[tid] + dt * (a40*f0[tid] + a42*f2[tid] + a43*f3[tid]);
	}
}

// ytemp = yn + dt*(a50*f0 + a53*f3 + a54*f4)
void rungekutta8::cpu_calc_ytemp_for_f5(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, var_t a50, var_t a53, var_t a54)
{
	for (int tid = 0; tid < n; tid++)
	{
		ytemp[tid] = y_n[tid] + dt * (a50*f0[tid] + a53*f3[tid] + a54*f4[tid]);
	}
}

// ytemp = yn + dt*(a60*f0 + a63*f3 + a64*f4 + a65*f5)
void rungekutta8::cpu_calc_ytemp_for_f6(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, var_t a60, var_t a63, var_t a64, var_t a65)
{
	for (int tid = 0; tid < n; tid++)
	{
		ytemp[tid] = y_n[tid] + dt * (a60*f0[tid] + a63*f3[tid] + a64*f4[tid] + a65*f5[tid]);
	}
}

// ytemp = yn + dt*(a70*f0 + a74*f4 + a75*f5 + a76*f6)
void rungekutta8::cpu_calc_ytemp_for_f7(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f4, const var_t *f5, const var_t *f6, var_t a70, var_t a74, var_t a75, var_t a76)
{
	for (int tid = 0; tid < n; tid++)
	{
		ytemp[tid] = y_n[tid] + dt * (a70*f0[tid] + a74*f4[tid] + a75*f5[tid] + a76*f6[tid]);
	}
}

// ytemp = yn + dt*(a80*f0 + a83*f3 + a84*f4 + a85*f5 + a86*f6 + a87*f7)
void rungekutta8::cpu_calc_ytemp_for_f8(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, var_t a80, var_t a83, var_t a84, var_t a85, var_t a86, var_t a87)
{
	for (int tid = 0; tid < n; tid++)
	{
		ytemp[tid] = y_n[tid] + dt * (a80*f0[tid] + a83*f3[tid] + a84*f4[tid] + a85*f5[tid] + a86*f6[tid] + a87*f7[tid]);
	}
}

// ytemp = yn + dt*(a90*f0 + a93*f3 + a94*f4 + a95*f5 + a96*f6 + a97*f7 + a98*f8)
void rungekutta8::cpu_calc_ytemp_for_f9(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, var_t a90, var_t a93, var_t a94, var_t a95, var_t a96, var_t a97, var_t a98)
{
	for (int tid = 0; tid < n; tid++)
	{
		ytemp[tid] = y_n[tid] + dt * (a90*f0[tid] + a93*f3[tid] + a94*f4[tid] + a95*f5[tid] + a96*f6[tid] + a97*f7[tid] + a98*f8[tid]);
	}
}

// ytemp = yn + dt*(a100*f0 + a103*f3 + a104*f4 + a105*f5 + a106*f6 + a107*f7 + a108*f8 + a109*f9)
void rungekutta8::cpu_calc_ytemp_for_f10(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, var_t a100, var_t a103, var_t a104, var_t a105, var_t a106, var_t a107, var_t a108, var_t a109)
{
	for (int tid = 0; tid < n; tid++)
	{
		ytemp[tid] = y_n[tid] + dt * (a100*f0[tid] + a103*f3[tid] + a104*f4[tid] + a105*f5[tid] + a106*f6[tid] + a107*f7[tid] + a108*f8[tid] + a109*f9[tid]);
	}
}

// ytemp = yn + dt*(a110*f0 + a115*f5 + a116*f6 + a117*f7 + a118*f8 + a119*f9)
void rungekutta8::cpu_calc_ytemp_for_f11(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, var_t a110, var_t a115, var_t a116, var_t a117, var_t a118, var_t a119)
{
	for (int tid = 0; tid < n; tid++)
	{
		ytemp[tid] = y_n[tid] + dt * (a110*f0[tid] + a115*f5[tid] + a116*f6[tid] + a117*f7[tid] + a118*f8[tid] + a119*f9[tid]);
	}
}

// ytemp = yn + dt*(a120*f0 + a123*f3 + a124*f4 + a125*f5 + a126*f6 + a127*f7 + a128*f8 + a129*f9 + a1211*f11)
void rungekutta8::cpu_calc_ytemp_for_f12(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, const var_t *f11, var_t a120, var_t a123, var_t a124, var_t a125, var_t a126, var_t a127, var_t a128, var_t a129, var_t a1211)
{
	for (int tid = 0; tid < n; tid++)
	{
		ytemp[tid] = y_n[tid] + dt * (a120*f0[tid] + a123*f3[tid] + a124*f4[tid] + a125*f5[tid] + a126*f6[tid] + a127*f7[tid] + a128*f8[tid] + a129*f9[tid] + a1211*f11[tid]);
	}
}

// err = abs(f0 + f10 - f11 - f12)
void rungekutta8::cpu_calc_error(int_t n, var_t *err, const var_t *f0, const var_t *f10, const var_t *f11, const var_t *f12)
{
	for (int tid = 0; tid < n; tid++)
	{
		err[tid] = fabs(f0[tid] + f10[tid] - f11[tid] - f12[tid]);
	}
}

// y_n+1 = yn + dt*(b0*f0 + b5*f5 + b6*f6 + b7*f7 + b8*f8 + b9*f9 + b10*f10)
void rungekutta8::cpu_calc_y_np1(int_t n, var_t *y_np1, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, const var_t *f10, var_t b0, var_t b5, var_t b6, var_t b7, var_t b8, var_t b9, var_t b10)
{
	for (int tid = 0; tid < n; tid++)
	{
		y_np1[tid] = y_n[tid] + dt * (b0*f0[tid] + b5*f5[tid] + b6*(f6[tid] + f7[tid]) + b8*(f8[tid] + f9[tid]) + b10*f10[tid]);
	}
}


rungekutta8::rungekutta8(pp_disk *ppd, ttt_t dt, bool adaptive, var_t tolerance, computing_device_t comp_dev) :
	integrator(ppd, dt, adaptive, tolerance, (adaptive ? 13 : 11), comp_dev)
{
	name  = "Runge-Kutta-Fehlberg7";
	order = 7;    
}

rungekutta8::~rungekutta8()
{}

void rungekutta8::calc_ytemp_for_fr(uint32_t n_var, int r)
{
	int idx = 0;
	for (int i = 0; i < 2; i++)
	{
		var_t* y_n  = (var_t*)ppd->sim_data->y[i];
		var_t* ytmp = (var_t*)ytemp[i];

		var_t* f0 = (var_t*)dydx[i][0];
		var_t* f1 = (var_t*)dydx[i][1];
		var_t* f2 = (var_t*)dydx[i][2];
		var_t* f3 = (var_t*)dydx[i][3];
		var_t* f4 = (var_t*)dydx[i][4];
		var_t* f5 = (var_t*)dydx[i][5];
		var_t* f6 = (var_t*)dydx[i][6];
		var_t* f7 = (var_t*)dydx[i][7];
		var_t* f8 = (var_t*)dydx[i][8];
		var_t* f9 = (var_t*)dydx[i][9];
		//var_t* f10= (var_t*)dydx[i][10];
		var_t* f11= adaptive ? (var_t*)dydx[i][11] : 0x0;

		switch (r)
		{
		case 1:
			idx = 1;
			if (COMPUTING_DEVICE_CPU == comp_dev)
			{
				cpu_calc_ytemp_for_f1(n_var, ytmp, dt_try, y_n, f0, a[idx]);
			}
			else
			{
				rk8_kernel::calc_ytemp_for_f1<<<grid, block>>>(n_var, ytmp, dt_try, y_n, f0, a[idx]);
				CUDA_CHECK_ERROR();
			}
			break;
		case 2:
			idx = 2;
			if (COMPUTING_DEVICE_CPU == comp_dev)
			{
				cpu_calc_ytemp_for_f2(n_var, ytmp, dt_try, y_n, f0, f1, a[idx], a[idx+1]);
			}
			else
			{
				rk8_kernel::calc_ytemp_for_f2<<<grid, block>>>(n_var, ytmp, dt_try, y_n, f0, f1, a[idx], a[idx+1]);
				CUDA_CHECK_ERROR();
			}
			break;
		case 3:
			idx = 4;
			if (COMPUTING_DEVICE_CPU == comp_dev)
			{
				cpu_calc_ytemp_for_f3(n_var, ytmp, dt_try, y_n, f0, f2, a[idx], a[idx+2]);
			}
			else
			{
				rk8_kernel::calc_ytemp_for_f3<<<grid, block>>>(n_var, ytmp, dt_try, y_n, f0, f2, a[idx], a[idx+2]);
				CUDA_CHECK_ERROR();
			}
			break;
		case 4:
			idx = 7;
			if (COMPUTING_DEVICE_CPU == comp_dev)
			{
				cpu_calc_ytemp_for_f4(n_var, ytmp, dt_try, y_n, f0, f2, f3, a[idx], a[idx+2], a[idx+3]);
			}
			else
			{
				rk8_kernel::calc_ytemp_for_f4<<<grid, block>>>(n_var, ytmp, dt_try, y_n, f0, f2, f3, a[idx], a[idx+2], a[idx+3]);
				CUDA_CHECK_ERROR();
			}
			break;
		case 5:
			idx = 11;
			if (COMPUTING_DEVICE_CPU == comp_dev)
			{
				cpu_calc_ytemp_for_f5(n_var, ytmp, dt_try, y_n, f0, f3, f4, a[idx], a[idx+3], a[idx+4]);
			}
			else
			{
				rk8_kernel::calc_ytemp_for_f5<<<grid, block>>>(n_var, ytmp, dt_try, y_n, f0, f3, f4, a[idx], a[idx+3], a[idx+4]);
				CUDA_CHECK_ERROR();
			}
			break;
		case 6:
			idx = 16;
			if (COMPUTING_DEVICE_CPU == comp_dev)
			{
				cpu_calc_ytemp_for_f6(n_var, ytmp, dt_try, y_n, f0, f3, f4, f5, a[idx], a[idx+3], a[idx+4], a[idx+5]);
			}
			else
			{
				rk8_kernel::calc_ytemp_for_f6<<<grid, block>>>(n_var, ytmp, dt_try, y_n, f0, f3, f4, f5, a[idx], a[idx+3], a[idx+4], a[idx+5]);
				CUDA_CHECK_ERROR();
			}
			break;
		case 7:
			idx = 22;
			if (COMPUTING_DEVICE_CPU == comp_dev)
			{
				cpu_calc_ytemp_for_f7(n_var, ytmp, dt_try, y_n, f0, f4, f5, f6, a[idx], a[idx+4], a[idx+5], a[idx+6]);
			}
			else
			{
				rk8_kernel::calc_ytemp_for_f7<<<grid, block>>>(n_var, ytmp, dt_try, y_n, f0, f4, f5, f6, a[idx], a[idx+4], a[idx+5], a[idx+6]);
				CUDA_CHECK_ERROR();
			}
			break;
		case 8:
			idx = 29;
			if (COMPUTING_DEVICE_CPU == comp_dev)
			{
				cpu_calc_ytemp_for_f8(n_var, ytmp, dt_try, y_n, f0, f3, f4, f5, f6, f7, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7]);
			}
			else
			{
				rk8_kernel::calc_ytemp_for_f8<<<grid, block>>>(n_var, ytmp, dt_try, y_n, f0, f3, f4, f5, f6, f7, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7]);
				CUDA_CHECK_ERROR();
			}
			break;
		case 9:
			idx = 37;
			if (COMPUTING_DEVICE_CPU == comp_dev)
			{
				cpu_calc_ytemp_for_f9(n_var, ytmp, dt_try, y_n, f0, f3, f4, f5, f6, f7, f8, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7], a[idx+8]);
			}
			else
			{
				rk8_kernel::calc_ytemp_for_f9<<<grid, block>>>(n_var, ytmp, dt_try, y_n, f0, f3, f4, f5, f6, f7, f8, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7], a[idx+8]);
				CUDA_CHECK_ERROR();
			}
			break;
		case 10:
			idx = 46;
			if (COMPUTING_DEVICE_CPU == comp_dev)
			{
				cpu_calc_ytemp_for_f10(n_var, ytmp, dt_try, y_n, f0, f3, f4, f5, f6, f7, f8, f9, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7], a[idx+8], a[idx+9]);
			}
			else
			{
				rk8_kernel::calc_ytemp_for_f10<<<grid, block>>>(n_var, ytmp, dt_try, y_n, f0, f3, f4, f5, f6, f7, f8, f9, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7], a[idx+8], a[idx+9]);
				CUDA_CHECK_ERROR();
			}
			break;
		case 11:
			idx = 56;
			if (COMPUTING_DEVICE_CPU == comp_dev)
			{
				cpu_calc_ytemp_for_f11(n_var, ytmp, dt_try, y_n, f0, f5, f6, f7, f8, f9, a[idx], a[idx+5], a[idx+6], a[idx+7], a[idx+8], a[idx+9]);
			}
			else
			{
				rk8_kernel::calc_ytemp_for_f11<<<grid, block>>>(n_var, ytmp, dt_try, y_n, f0, f5, f6, f7, f8, f9, a[idx], a[idx+5], a[idx+6], a[idx+7], a[idx+8], a[idx+9]);
				CUDA_CHECK_ERROR();
			}
			break;
		case 12:
			idx = 67;
			if (COMPUTING_DEVICE_CPU == comp_dev)
			{
				cpu_calc_ytemp_for_f12(n_var, ytmp, dt_try, y_n, f0, f3, f4, f5, f6, f7, f8, f9, f11, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7], a[idx+8], a[idx+9], a[idx+11]);
			}
			else
			{
				rk8_kernel::calc_ytemp_for_f12<<<grid, block>>>(n_var, ytmp, dt_try, y_n, f0, f3, f4, f5, f6, f7, f8, f9, f11, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7], a[idx+8], a[idx+9], a[idx+11]);
				CUDA_CHECK_ERROR();
			}
			break;
		default:
			throw string("rungekutta8::calc_ytemp_for_fr: parameter out of range.");
		} /* switch */
	} /* for */
}

void rungekutta8::calc_error(uint32_t n_var)
{
	for (int i = 0; i < 2; i++)
	{
		var_t *f0  = (var_t*)dydx[i][0];
		var_t *f10 = (var_t*)dydx[i][10];
		var_t *f11 = (var_t*)dydx[i][11];
		var_t *f12 = (var_t*)dydx[i][12];

		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			rk8_kernel::calc_error<<<grid, block>>>(n_var, err[i], f0, f10, f11, f12);
			CUDA_CHECK_ERROR();
		}
		else
		{
			cpu_calc_error(n_var, err[i], f0, f10, f11, f12);
		}
	}
}

void rungekutta8::calc_y_np1(uint32_t n_var)
{
	for (int i = 0; i < 2; i++)
	{
		var_t* y_n   = (var_t*)ppd->sim_data->y[i];
		var_t* y_np1 = (var_t*)ppd->sim_data->yout[i];

		var_t *f0  = (var_t*)dydx[i][0];
		var_t *f5  = (var_t*)dydx[i][5];
		var_t *f6  = (var_t*)dydx[i][6];
		var_t *f7  = (var_t*)dydx[i][7];
		var_t *f8  = (var_t*)dydx[i][8];
		var_t *f9  = (var_t*)dydx[i][9];
		var_t *f10 = (var_t*)dydx[i][10];

		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			rk8_kernel::calc_y_np1<<<grid, block>>>(n_var, y_np1, dt_try, y_n, f0, f5, f6, f7, f8, f9, f10, b[0], b[5], b[6], b[7], b[8], b[9], b[10]);
			CUDA_CHECK_ERROR();
		}
		else
		{
			cpu_calc_y_np1(n_var, y_np1, dt_try, y_n, f0, f5, f6, f7, f8, f9, f10, b[0], b[5], b[6], b[7], b[8], b[9], b[10]);
		}
	}
}

ttt_t rungekutta8::step()
{
	const uint32_t n_body_total = ppd->n_bodies->get_n_total_playing();
	const uint32_t n_var_total = NDIM * n_body_total;

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		// Set the kernel launch parameters
		calc_grid(n_var_total, THREADS_PER_BLOCK);
	}

    dt_try = dt_next;
	// Calculate initial differentials and store them into dydx[][0]
	int r = 0;
	ttt_t ttemp = ppd->t + c[r] * dt_try;
	// Calculate f1 = f(tn, yn) = dydx[][0]
	const var4_t *coor = ppd->sim_data->y[0];
	const var4_t *velo = ppd->sim_data->y[1];
	for (int i = 0; i < 2; i++)
	{
		ppd->calc_dydx(i, r, ttemp, coor, velo, dydx[i][r]);
	}

	var_t max_err = 0.0;
	int iter = 0;
	do
	{
		dt_did = dt_try;
		// Calculate f2 = f(tn + c1 * dt, yn + a10 * dt * f0) = dydx[][1]
		// ...
		// Calculate f11 = f(tn + c10 * dt, yn + a10,0 * dt * f0 + ...) = dydx[][10]
		for (r = 1; r <= 10; r++)
		{
			ttemp = ppd->t + c[r] * dt_try;
			calc_ytemp_for_fr(n_var_total, r);
			for (int i = 0; i < 2; i++)
			{
				ppd->calc_dydx(i, r, ttemp, ytemp[0], ytemp[1], dydx[i][r]);
			}
		}
	
		// y_(n+1) = yn + dt*(b0*f0 + b5*f5 + b6*f6 + b7*f7 + b8*f8 + b9*f9 + b10*f10) + O(dt^8)
		calc_y_np1(n_var_total);

		if (adaptive)
		{
			// Calculate f11 = f(tn + c11 * dt, yn + ...) = dydx[][11]
			// Calculate f12 = f(tn + c11 * dt, yn + ...) = dydx[][12]
			for (r = 11; r < n_stage; r++)
			{
				ttemp = ppd->t + c[r] * dt_try;
				calc_ytemp_for_fr(n_var_total, r);
				for (int i = 0; i < 2; i++)
				{
					ppd->calc_dydx(i, r, ttemp, ytemp[0], ytemp[1], dydx[i][r]);
				}
			}

			uint32_t n_var = NDIM * (error_check_for_tp ? n_body_total : ppd->n_bodies->get_n_massive());
			// Calculate the absolute values of the errors
			calc_error(n_var);
            // In case of backward integration (when dt < 0) one needs abs of the maximum error
			max_err = fabs(LAMBDA * dt_try * get_max_error(n_var));
			if (0.0 < max_err)
			{
				dt_next = dt_try *= 0.9 * pow(tolerance / max_err, 1.0/(order));
			}
			else
			{	
				dt_next = dt_try *= 2.0;
			}
		}
		iter++;
	} while (adaptive && max_err > tolerance);

	update_counters(iter);

	ppd->t += dt_did;
	ppd->swap();

	return dt_did;
}




namespace c_rk8_kernel
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
			//TODO: the dt factor can be used at the end of the loop
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
		for (int i = 0; i < 11; i++)
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
	void calc_error(int n, const var_t *f0, const var_t *f10, const var_t *f11, const var_t *f12, var_t *err)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n)
	{
		err[tid] = fabs(f0[tid] + f10[tid] - f11[tid] - f12[tid]);
	}
}
} /* c_rk8_kernel */


c_rungekutta8::c_rungekutta8(pp_disk *ppd, ttt_t dt, bool adaptive, var_t tolerance, computing_device_t comp_dev) :
	integrator(ppd, dt, adaptive, tolerance, (adaptive ? 13 : 11), comp_dev)
{
	name  = "c_Runge-Kutta-Fehlberg8";
	order = 7;

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		redutilcu::copy_constant_to_device(dc_a, a,   sizeof(a));
		redutilcu::copy_constant_to_device(dc_b, b,   sizeof(b));
		redutilcu::copy_constant_to_device(dc_bh, bh, sizeof(bh));
		redutilcu::copy_constant_to_device(dc_c, c,   sizeof(c));
	}
}

c_rungekutta8::~c_rungekutta8()
{}

void c_rungekutta8::call_kernel_calc_ytemp(uint32_t n_var, int r)
{
	static int idx_array[] = {0, 1, 2, 4, 7, 11, 16, 22, 29, 37, 46, 56, 67};

	for (int i = 0; i < 2; i++)
	{
		var_t* y_n   = (var_t*)ppd->sim_data->y[i];
		var_t** dydt = (var_t**)d_dydt;
		var_t* ytmp = (var_t*)ytemp[i];

		c_rk8_kernel::calc_ytemp<<<grid, block>>>(n_var, r, idx_array[r], i*n_stage, dt_try, y_n, dydt, ytmp);
		CUDA_CHECK_ERROR();
	}
}

void c_rungekutta8::call_kernel_calc_error(uint32_t n_var)
{
	for (int i = 0; i < 2; i++)
	{
		var_t* f0  = (var_t*)dydx[i][0];
		var_t* f10 = (var_t*)dydx[i][10];
		var_t* f11 = (var_t*)dydx[i][11];
		var_t* f12 = (var_t*)dydx[i][12];

		c_rk8_kernel::calc_error<<<grid, block>>>(n_var, f0, f10, f11, f12, err[i]);
		CUDA_CHECK_ERROR();
	}
}

void c_rungekutta8::call_kernel_calc_y_np1(uint32_t n_var)
{
	for (int i = 0; i < 2; i++)
	{
		var_t* y_n   = (var_t*)ppd->sim_data->y[i];
		var_t** dydt = (var_t**)d_dydt;
		var_t* y_np1 = (var_t*)ppd->sim_data->yout[i];

		c_rk8_kernel::calc_y_np1<<<grid, block>>>(n_var, i*n_stage, dt_try, y_n, dydt, y_np1);
		CUDA_CHECK_ERROR();
	}
}

ttt_t c_rungekutta8::step()
{
	const uint32_t n_body_total = ppd->n_bodies->get_n_total_playing();
	const uint32_t n_var_total = NDIM * n_body_total;

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		// Set the kernel launch parameters
		calc_grid(n_var_total, THREADS_PER_BLOCK);
	}

    dt_try = dt_next;
	// Calculate initial differentials and store them into dydx[][0]
	int stage = 0;
	ttt_t ttemp = ppd->t + c[stage] * dt_try;
	// Calculate f1 = f(tn, yn) = dydx[][0]
	const var4_t *coor = ppd->sim_data->y[0];
	const var4_t *velo = ppd->sim_data->y[1];
	for (int i = 0; i < 2; i++)
	{
		ppd->calc_dydx(i, stage, ttemp, coor, velo, dydx[i][stage]);
	}

	var_t max_err = 0.0;
	int iter = 0;
	do
	{
		dt_did = dt_try;
		// Calculate f2 = f(tn + c1 * dt, yn + a10 * dt * f0) = dydx[][1]
		// ...
		// Calculate f11 = f(tn + c10 * dt, yn + a10,0 * dt * f0 + ...) = dydx[][10]
		for (stage = 1; stage <= 10; stage++)
		{
			ttemp = ppd->t + c[stage] * dt_try;
			call_kernel_calc_ytemp(n_var_total, stage);

			for (int i = 0; i < 2; i++)
			{
				ppd->calc_dydx(i, stage, ttemp, ytemp[0], ytemp[1], dydx[i][stage]);
			}
		}
	
		// y_(n+1) = yn + dt*(b0*f0 + b5*f5 + b6*f6 + b7*f7 + b8*f8 + b9*f9 + b10*f10) + O(dt^8)
		call_kernel_calc_y_np1(n_var_total);

		if (adaptive)
		{
			// Calculate f11 = f(tn + c11 * dt, yn + ...) = dydx[][11]
			// Calculate f12 = f(tn + c11 * dt, yn + ...) = dydx[][12]
			for (stage = 11; stage < n_stage; stage++)
			{
				ttemp = ppd->t + c[stage] * dt_try;
				call_kernel_calc_ytemp(n_var_total, stage);
				for (int i = 0; i < 2; i++)
				{
					ppd->calc_dydx(i, stage, ttemp, ytemp[0], ytemp[1], dydx[i][stage]);
				}
			}

			uint32_t n_var = NDIM * (error_check_for_tp ? ppd->n_bodies->get_n_total_playing() : ppd->n_bodies->get_n_massive());
			// Calculate the absolute values of the errors
			call_kernel_calc_error(n_var);
            // In case of backward integration (when dt < 0) one needs abs of the maximum error
			max_err = fabs(LAMBDA * dt_try * get_max_error(n_var));
			if (0.0 < max_err)
			{
				dt_next = dt_try *= 0.9 * pow(tolerance / max_err, 1.0/(order));
			}
			else
			{	
				dt_next = dt_try *= 2.0;
			}
		}
		iter++;
	} while (adaptive && max_err > tolerance);

	update_counters(iter);

	ppd->t += dt_did;
	ppd->swap();

	return dt_did;
}

#undef LAMBDA
