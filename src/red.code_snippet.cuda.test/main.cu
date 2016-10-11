// includes system
#include <algorithm>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <stdio.h>

// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes Thrust
#include "thrust/device_ptr.h"
#include "thrust/fill.h"
#include "thrust/extrema.h"

// includes project
#include "red_constants.h"
#include "red_type.h"
#include "red_macro.h"
#include "redutilcu.h"
#include "red_test.h"

using namespace std;
using namespace redutilcu;


__constant__ var_t dc_threshold[THRESHOLD_N];
__constant__ analytic_gas_disk_params_t dc_anal_gd_params;

#define N    256

namespace kernel
{
static __global__
	void populate(int n, var_t value, var_t* dst)
{
	for (int i = 0; i < n; i++)
	{
		dst[i] = value;
	}
}

static __global__
	void set_element_of_array(int n, int idx, var_t* v, var_t value)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n && idx == i)
	{
		v[idx] = value;
	}
}

static __global__
	void print_memory_address(int n, var_t* adr)
{
	for (int i = 0; i < n; i++)
	{
		printf("adr[%2d]: %p\n", i, adr[i]);
	}
}

static __global__
	void print_vector(int n, const var4_t* v)
{
	int tid = 0;
	while (tid < n)
	{
		printf("%5d %20.16lf, %20.16lf, %20.16lf, %20.16lf\n", tid, v[tid].x, v[tid].y, v[tid].z, v[tid].w);
		tid++;
	}
}

static __global__
	void print_array(int n, const var_t* v)
{
	printf("v: %p\n", v);

	for (int i = 0; i < n; i++)
	{
		printf("v[%4d] : %20.16lf\n", i, v[i]);
	}
}

static __global__
	void print_array(int n, int k, var_t** v)
{
	printf("k: %2d\n", k);
	printf("v: %p\n", v);
	printf("v[%2d]: %p\n", k, v[k]);

	for (int i = 0; i < n; i++)
	{
		printf("v[%2d][%2d] : %20.16lf\n", k, i, v[k][i]);
	}
}

static __global__
	void print_constant_memory()
{
	printf("dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE        ] : %lf\n", dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE]);
	printf("dc_threshold[THRESHOLD_EJECTION_DISTANCE           ] : %lf\n", dc_threshold[THRESHOLD_EJECTION_DISTANCE]);
	printf("dc_threshold[THRESHOLD_RADII_ENHANCE_FACTOR        ] : %lf\n", dc_threshold[THRESHOLD_RADII_ENHANCE_FACTOR]);
}

static __global__
	void print_analytic_gas_disk_params()
{
	printf(" eta: %25.lf,  eta.y: %25.lf\n", dc_anal_gd_params.eta.x, dc_anal_gd_params.eta.y);
	printf(" sch: %25.lf,  sch.y: %25.lf\n", dc_anal_gd_params.sch.x, dc_anal_gd_params.sch.y);
	printf(" tau: %25.lf,  tau.y: %25.lf\n", dc_anal_gd_params.tau.x, dc_anal_gd_params.tau.y);
	printf(" rho: %25.lf,  rho.y: %25.lf\n", dc_anal_gd_params.rho.x, dc_anal_gd_params.rho.y);

	printf(" mfp: %25.lf,  mfp.y: %25.lf\n", dc_anal_gd_params.mfp.x,  dc_anal_gd_params.mfp.y);
	printf("temp: %25.lf, temp.y: %25.lf\n", dc_anal_gd_params.temp.x, dc_anal_gd_params.temp.y);

	printf("  gas_decrease: %d\n", dc_anal_gd_params.gas_decrease);
	printf("            t0: %25.lf\n", dc_anal_gd_params.t0);
	printf("            t1: %25.lf\n", dc_anal_gd_params.t1);
	printf("e_folding_time: %25.lf\n", dc_anal_gd_params.e_folding_time);

	printf(" c_vth: %25.lf\n", dc_anal_gd_params.c_vth);
	printf(" alpha: %25.lf\n", dc_anal_gd_params.alpha);
	printf("mean_molecular_weight: %25.lf\n", dc_anal_gd_params.mean_molecular_weight);
	printf("    particle_diameter: %25.lf\n", dc_anal_gd_params.particle_diameter);
}

static __global__
	void calc_grav_accel_int_mul_of_thread_per_block(interaction_bound int_bound, const pp_disk_t::param_t* p, const var4_t* r, var4_t* a)
{
	const int i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	var4_t dVec;
	// This line (beyond my depth) speeds up the kernel
	a[i].x = a[i].y = a[i].z = a[i].w = 0.0;
	for (int j = int_bound.source.x; j < int_bound.source.y; j++) 
	{
		/* Skip the body with the same index */
		if (i == j)
		{
			continue;
		}
		// 3 FLOP
		dVec.x = r[j].x - r[i].x;
		dVec.y = r[j].y - r[i].y;
		dVec.z = r[j].z - r[i].z;
		// 5 FLOP
		dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2
		// 20 FLOP
		var_t d = sqrt(dVec.w);								// = r
		// 2 FLOP
		dVec.w = p[j].mass / (d*dVec.w);					// = m / r^3
		// 6 FLOP
		a[i].x += dVec.w * dVec.x;
		a[i].y += dVec.w * dVec.y;
		a[i].z += dVec.w * dVec.z;
	}
}

static __global__
	void calc_grav_accel(ttt_t t, interaction_bound int_bound, const pp_disk_t::body_metadata_t* body_md, const pp_disk_t::param_t* p, const var4_t* r, const var4_t* v, var4_t* a, pp_disk_t::event_data_t* events, int *event_counter)
{
	const int i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	if (i < int_bound.sink.y)
	{
		// This line (beyond my depth) speeds up the kernel
		a[i].x = a[i].y = a[i].z = a[i].w = 0.0;
		if (body_md[i].id > 0)
		{
			var4_t dVec = {0.0, 0.0, 0.0, 0.0};
			for (int j = int_bound.source.x; j < int_bound.source.y; j++) 
			{
				/* Skip the body with the same index and those which are inactive ie. id < 0 */
				if (i == j || body_md[j].id < 0)
				{
					continue;
				}
				// 3 FLOP
				dVec.x = r[j].x - r[i].x;
				dVec.y = r[j].y - r[i].y;
				dVec.z = r[j].z - r[i].z;
				// 5 FLOP
				dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2
				// 20 FLOP
				var_t d = sqrt(dVec.w);								// = r
				// 2 FLOP
				dVec.w = p[j].mass / (d*dVec.w);
				// 6 FLOP
				a[i].x += dVec.w * dVec.x;
				a[i].y += dVec.w * dVec.y;
				a[i].z += dVec.w * dVec.z;

				// Check for collision - ignore the star (i > 0 criterium)
				// The data of the collision will be stored for the body with the greater index (test particles can collide with massive bodies)
				// If i < j is the condition than test particles can not collide with massive bodies
				if (i > 0 && i > j && d < /* dc_threshold[THRESHOLD_RADII_ENHANCE_FACTOR] */ 5.0 * (p[i].radius + p[j].radius))
				{
					uint32_t k = atomicAdd(event_counter, 1);

					int survivIdx = i;
					int mergerIdx = j;
					if (p[mergerIdx].mass > p[survivIdx].mass)
					{
						int t = survivIdx;
						survivIdx = mergerIdx;
						mergerIdx = t;
					}
					//printf("t = %20.10le d = %20.10le %d. COLLISION detected: id: %5d id: %5d\n", t, d, k+1, body_md[survivIdx].id, body_md[mergerIdx].id);

					events[k].event_name = EVENT_NAME_COLLISION;
					events[k].d = d;
					events[k].t = t;
					events[k].id1 = body_md[survivIdx].id;
					events[k].id2 = body_md[mergerIdx].id;
					events[k].idx1 = survivIdx;
					events[k].idx2 = mergerIdx;
					events[k].r1 = r[survivIdx];
					events[k].v1 = v[survivIdx];
					events[k].r2 = r[mergerIdx];
					events[k].v2 = v[mergerIdx];
				}
			}
			// 36 FLOP
			// With the used time unit k = 1
			//a[i].x *= K2;
			//a[i].y *= K2;
			//a[i].z *= K2;
		}
	}
}

inline __host__ __device__
	var4_t body_body_interaction(var4_t riVec, var4_t rjVec, var_t mj, var4_t aiVec)
{
	var4_t dVec = {0.0, 0.0, 0.0, 0.0};

	// compute d = r_i - r_j [3 FLOPS] [6 read, 3 write]
	dVec.x = rjVec.x - riVec.x;
	dVec.y = rjVec.y - riVec.y;
	dVec.z = rjVec.z - riVec.z;

	// compute norm square of d vector [5 FLOPS] [3 read, 1 write]
	dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);
	// compute norm of d vector [1 FLOPS] [1 read, 1 write] TODO: how long does it take to compute sqrt ???
	var_t s = sqrt(dVec.w);
	// compute m_j / d^3 []
	s = mj * 1.0 / (s * dVec.w);

	aiVec.x += s * dVec.x;
	aiVec.y += s * dVec.y;
	aiVec.z += s * dVec.z;

	return aiVec;
}

/*
 * riVec = {x, y, z,  }
 * rjVec = {x, y, z, m}
 */
inline __host__ __device__
	var4_t body_body_interaction(var4_t riVec, var4_t rjVec, var4_t aiVec)
{
	var4_t dVec;

	// compute d = r_i - r_j [3 FLOPS] [6 read, 3 write]
	dVec.x = rjVec.x - riVec.x;
	dVec.y = rjVec.y - riVec.y;
	dVec.z = rjVec.z - riVec.z;

	// compute norm square of d vector [5 FLOPS] [3 read, 1 write]
	dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);
	// compute norm of d vector [1 FLOPS] [1 read, 1 write] TODO: how long does it take to compute sqrt ???
	var_t d = sqrt(dVec.w);
	// compute m_j / d^3 []
	dVec.w = rjVec.w * 1.0 / (d * dVec.w);

	aiVec.x += dVec.w * dVec.x;
	aiVec.y += dVec.w * dVec.y;
	aiVec.z += dVec.w * dVec.z;

	return aiVec;
}

__global__
	void calc_gravity_accel_tile_verbose(int tile_size, const var4_t* global_x, const var_t* mass, var4_t* global_a)
{
	extern __shared__ var4_t sh_pos[];

	var4_t my_pos = {0.0, 0.0, 0.0, 0.0};
	var4_t acc    = {0.0, 0.0, 0.0, 0.0};

	const int gtid = blockIdx.x * blockDim.x + threadIdx.x;

	if (0 == gtid)
	{
		printf("[0 == gtid]:  gridDim = [%3d, %3d, %3d]\n", gridDim.x, gridDim.y, gridDim.z);
		printf("[0 == gtid]: blockDim = [%3d, %3d, %3d]\n", blockDim.x, blockDim.y, blockDim.z);
		printf("[0 == gtid]: total number of threads = gridDim.x * blockDim.x = %3d\n", gridDim.x * blockDim.x);
		//printf("[0 == gtid]: sizeof(sh_pos) = %5d [byte]\n", sizeof(sh_pos));
	}

	if (0 == threadIdx.x)
	{
		printf("[0 == threadIdx.x]: blockIdx.x = %3d\n", blockIdx.x);
	}
	if (0 == blockIdx.x)
	{
		printf("[0 == blockIdx.x]: threadIdx.x = %3d\n", threadIdx.x);
	}

	// To avoid overruning the global_x buffer
	if (N > gtid)
	{
		my_pos = global_x[gtid];
	}
	printf("gtid = %3d my_pos = [%10.6le, %10.6le, %10.6le]\n", gtid, my_pos.x, my_pos.y, my_pos.z);
	if (0 == blockIdx.x)
	{
		printf("[0 == blockIdx.x]: gtid = %3d my_pos = [%10.6le, %10.6le, %10.6le]\n", gtid, my_pos.x, my_pos.y, my_pos.z);
	}

	for (int tile = 0; (tile * tile_size) < N; tile++)
	{
		int idx = tile * blockDim.x + threadIdx.x;
		// To avoid overruning the global_x buffer
		if (N > idx)
		{
			sh_pos[threadIdx.x] = global_x[idx];
		}
		if (0 == blockIdx.x)
		{
			printf("[0 == blockIdx.x]: tile = %3d threadIdx.x = %3d idx = %3d sh_pos[threadIdx.x] = [%10.6le, %10.6le, %10.6le]\n", tile, threadIdx.x, idx, sh_pos[threadIdx.x].x, sh_pos[threadIdx.x].y, sh_pos[threadIdx.x].z);
		}
		__syncthreads();
		if (0 == blockIdx.x && 0 == threadIdx.x)
		{
			printf("After 1st __syncthreads()\n");
		}

		for (int j = 0; j < blockDim.x; j++)
		{
			if (0 == blockIdx.x)
			{
				// To avoid overrun the mass buffer
				if (N <= (tile * tile_size) + j)
				{
					printf("Warning: N (%3d) <= tile * tile_size + j (%3d)\n", N, (tile * tile_size) + j);
				}
				// To avoid self-interaction or mathematically division by zero
				if (gtid == (tile * tile_size)+j)
				{
					printf("Warning: gtid (%3d) == (tile * tile_size)+j (%3d)\n", gtid, (tile * tile_size)+j);
				}
				printf("[0 == blockIdx.x]: tile = %3d j = %3d threadIdx.x = %3d idx = %3d my_pos = [%10.6le, %10.6le, %10.6le] sh_pos[j] = [%10.6le, %10.6le, %10.6le]\n", tile, j, threadIdx.x, idx, my_pos.x, my_pos.y, my_pos.z, sh_pos[j].x, sh_pos[j].y, sh_pos[j].z);
			}
			// To avoid overrun the mass buffer
			if (N <= (tile * tile_size) + j)
			{
				break;
			}
			// To avoid self-interaction or mathematically division by zero
			if (gtid != (tile * tile_size)+j)
			{
				acc = body_body_interaction(my_pos, sh_pos[j], mass[idx], acc);
			}
		}

		__syncthreads();
		if (0 == blockIdx.x && 0 == threadIdx.x)
		{
			printf("After 2nd __syncthreads()\n");
		}
	}

	// To avoid overruning the global_a buffer
	if (N > gtid)
	{
		printf("gtid = %3d acc = [%14.6le, %14.6le, %14.6le]\n", gtid, acc.x, acc.y, acc.z);
		global_a[gtid] = acc;
	}
}

__global__
	void calc_gravity_accel_tile(int tile_size, const var4_t* global_x, const var_t* mass, var4_t* global_a)
{
	extern __shared__ var4_t sh_pos[];

	var4_t my_pos = {0.0, 0.0, 0.0, 0.0};
	var4_t acc    = {0.0, 0.0, 0.0, 0.0};

	const int gtid = blockIdx.x * blockDim.x + threadIdx.x;

	// To avoid overruning the global_x buffer
	if (N > gtid)
	{
		my_pos = global_x[gtid];
	}
	for (int tile = 0; (tile * tile_size) < N; tile++)
	{
		int idx = tile * blockDim.x + threadIdx.x;
		// To avoid overruning the global_x buffer
		if (N > idx)
		{
			sh_pos[threadIdx.x] = global_x[idx];
		}
		__syncthreads();
		for (int j = 0; j < blockDim.x; j++)
		{
			// To avoid overrun the mass buffer
			if (N <= (tile * tile_size) + j)
			{
				break;
			}
			// To avoid self-interaction or mathematically division by zero
			if (gtid != (tile * tile_size)+j)
			{
				acc = body_body_interaction(my_pos, sh_pos[j], mass[idx], acc);
				//var4_t dVec = {0.0, 0.0, 0.0, 0.0};

				//// compute d = r_i - r_j [3 FLOPS] [6 read, 3 write]
				//dVec.x = sh_pos[j].x - my_pos.x;
				//dVec.y = sh_pos[j].y - my_pos.y;
				//dVec.z = sh_pos[j].z - my_pos.z;

				//// compute norm square of d vector [5 FLOPS] [3 read, 1 write]
				//dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);
				//// compute norm of d vector [1 FLOPS] [1 read, 1 write] TODO: how long does it take to compute sqrt ???
				//var_t s = sqrt(dVec.w);
				//// compute m_j / d^3 []
				//s = mass[j] * 1.0 / (s * dVec.w);

				//acc.x += s * dVec.x;
				//acc.y += s * dVec.y;
				//acc.z += s * dVec.z;
			}
		}
		__syncthreads();
	}
	// To avoid overruning the global_a buffer
	if (N > gtid)
	{
		global_a[gtid] = acc;
	}
}

#define GTID     16
#define BLKIDX    4
/*
 * The same as above but the blockDim.x is used instead of tile_size (both variable have the same value)
 */
__global__
	void calc_gravity_accel_tile_verbose(const var4_t* global_x, const var_t* mass, var4_t* global_a)
{
	extern __shared__ var4_t sh_pos[];
	// extern __shared__ var_t sh_mass[512]; // This ruins the code !!!
	//__shared__ var_t sh_mass[512];         // This is OK but it is a static allocation, NOT NICE

	const int gtid = blockIdx.x * blockDim.x + threadIdx.x;

	if (0 == gtid)
	{
		//printf("[0 == gtid]:  gridDim = [%3d, %3d, %3d] [block]\n", gridDim.x, gridDim.y, gridDim.z);
		//printf("[0 == gtid]: blockDim = [%3d, %3d, %3d] [thread]\n", blockDim.x, blockDim.y, blockDim.z);
		//printf("[0 == gtid]: total size of the grid = gridDim.x * blockDim.x = %3d [thread]\n", gridDim.x * blockDim.x);
		//printf("[0 == gtid]: sizeof(sh_pos) = %5d [byte]\n", sizeof(sh_pos));
	}

	var4_t acc = {0.0, 0.0, 0.0, 0.0};
	var4_t my_pos;

	// To avoid overruning the global_x buffer
	if (N > gtid)
	{
		my_pos = global_x[gtid];
		if (GTID == gtid)
		{
			printf("[%2d == i] rVec = [%16.8le, %16.8le, %16.8le]\n", GTID, my_pos.x, my_pos.y, my_pos.z);
			printf("[%2d == i], blockIdx.x = %2d\n", GTID, blockIdx.x);
		}
	}
	
	for (int tile = 0; (tile * blockDim.x) < N; tile++)
	{
		const int idx = tile * blockDim.x + threadIdx.x;
		if (BLKIDX == blockIdx.x)
		{
			printf("[%2d == blockIdx.x]: tile = %3d threadIdx.x = %3d idx = %3d\n", BLKIDX, tile, threadIdx.x, idx);
		}
		// To avoid overruning the global_x and mass buffer
		if (N > idx)
		{
			if (GTID == gtid)
			{
				//printf("[Ln: %3d] [%2d == i] global_x = [%16.8le, %16.8le, %16.8le]\n", __LINE__, GTID, global_x[idx].x, global_x[idx].y, global_x[idx].z);
				printf("[Ln: %3d] [%2d == i] idx = [%5d]\n", __LINE__, GTID, idx);
			}
			sh_pos[ threadIdx.x]   = global_x[idx];
			sh_pos[ threadIdx.x].w = mass[idx];
			//sh_mass[threadIdx.x] = mass[idx];
			// This thread block executes the GTID thread
			if (BLKIDX == blockIdx.x)
			{
				printf("[Ln: %3d] [%2d == i] threadIdx.x = %2d sh_pos = [%16.8le, %16.8le, %16.8le]\n", __LINE__, GTID, threadIdx.x, sh_pos[threadIdx.x].x, sh_pos[threadIdx.x].y, sh_pos[threadIdx.x].z);
			}
		}
		__syncthreads();

		for (int j = 0; j < blockDim.x; j++)
		{
			// To avoid overrun the mass buffer
			if (N <= (tile * blockDim.x) + j)
			{
				break;
			}
			// To avoid self-interaction or mathematically division by zero
			if (gtid != (tile * blockDim.x) + j)
			{
				//acc = body_body_interaction(my_pos, sh_pos[j], mass[j], acc);
				// WARNING: not mass[j] BUT mass[idx] !! Check the other functions!!
				//acc = body_body_interaction(my_pos, sh_pos[j], mass[idx], acc);
				var4_t dVec = {0.0, 0.0, 0.0, 0.0};

				// compute d = r_i - r_j [3 FLOPS] [6 read, 3 write]
				dVec.x = sh_pos[j].x - my_pos.x;
				dVec.y = sh_pos[j].y - my_pos.y;
				dVec.z = sh_pos[j].z - my_pos.z;

				// compute norm square of d vector [5 FLOPS] [3 read, 1 write]
				dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);
				// compute norm of d vector [1 FLOPS] [1 read, 1 write] TODO: how long does it take to compute sqrt ???
				var_t d = sqrt(dVec.w);
				// compute m_j / d^3 []
				//dVec.w = mass[(tile * blockDim.x) + j] * 1.0 / (d * dVec.w);
				//dVec.w = sh_mass[j] * 1.0 / (d * dVec.w);
				dVec.w = sh_pos[j].w * 1.0 / (d * dVec.w);

				acc.x += dVec.w * dVec.x;
				acc.y += dVec.w * dVec.y;
				acc.z += dVec.w * dVec.z;

				if (GTID == gtid)
				{
					//printf("[%2d == i, j = %2d] dVec = [%16.8le, %16.8le, %16.8le] mj = %16.8le d = %16.8le w = %16.8le a = [%16.8le, %16.8le, %16.8le]\n", GTID, (tile * blockDim.x) + j, dVec.x, dVec.y, dVec.z, mass[(tile * blockDim.x) + j], d, dVec.w, acc.x, acc.y, acc.z);
					//printf("[%2d == i, j = %2d] dVec = [%16.8le, %16.8le, %16.8le] mj = %16.8le d = %16.8le w = %16.8le a = [%16.8le, %16.8le, %16.8le]\n", GTID, (tile * blockDim.x) + j, dVec.x, dVec.y, dVec.z, sh_mass[j], d, dVec.w, acc.x, acc.y, acc.z);
					printf("[%2d == i, j = %2d] dVec = [%16.8le, %16.8le, %16.8le] mj = %16.8le d = %16.8le w = %16.8le a = [%16.8le, %16.8le, %16.8le]\n", GTID, (tile * blockDim.x) + j, dVec.x, dVec.y, dVec.z, sh_pos[j].w, d, dVec.w, acc.x, acc.y, acc.z);
				}
			}
		}
		__syncthreads();
	}
	if (N > gtid)
	{
		global_a[gtid] = acc;
	}
}
#undef GTID
#undef BLKIDX

/*
 * The same as above but the blockDim.x is used instead of tile_size (both variable have the same value)
 */
__global__
	void calc_gravity_accel_tile(const var4_t* global_x, const var_t* mass, var4_t* global_a)
{
	extern __shared__ var4_t sh_pos[];

	const int gtid = blockIdx.x * blockDim.x + threadIdx.x;

	var4_t acc = {0.0, 0.0, 0.0, 0.0};
	var4_t my_pos;

	// To avoid overruning the global_x buffer
	if (N > gtid)
	{
		my_pos = global_x[gtid];
	}
	
	for (int tile = 0; (tile * blockDim.x) < N; tile++)
	{
		const int idx = tile * blockDim.x + threadIdx.x;
		// To avoid overruning the global_x and mass buffer
		if (N > idx)
		{
			sh_pos[threadIdx.x]   = global_x[idx];
			sh_pos[threadIdx.x].w = mass[idx];
		}
		__syncthreads();

		for (int j = 0; j < blockDim.x; j++)
		{
			// To avoid overrun the input arrays
			if (N <= (tile * blockDim.x) + j)
			{
				break;
			}
			// To avoid self-interaction or mathematically division by zero
			if (gtid != (tile * blockDim.x) + j)
			{
				acc = body_body_interaction(my_pos, sh_pos[j], acc);
			}
		}
		__syncthreads();
	}
	if (N > gtid)
	{
		global_a[gtid] = acc;
	}
}

__global__
	void calc_gravity_accel_naive(const var4_t* global_x, const var_t* mass, var4_t* global_a)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (N > i)
	{
		global_a[i].x = global_a[i].y = global_a[i].z = global_a[i].w = 0.0;
		var4_t dVec = {0.0, 0.0, 0.0, 0.0};
		for (int j = 0; j < N; j++) 
		{
			/* Skip the body with the same index */
			if (i == j)
			{
				continue;
			}
			//global_a[i] = body_body_interaction(global_x[i], global_x[j], mass[j], global_a[i]);
			// 3 FLOP
			dVec.x = global_x[j].x - global_x[i].x;
			dVec.y = global_x[j].y - global_x[i].y;
			dVec.z = global_x[j].z - global_x[i].z;
			// 5 FLOP
			dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2

			// 20 FLOP
			var_t d = sqrt(dVec.w);								// = r
			// 2 FLOP
			dVec.w = mass[j] / (d*dVec.w);
			// 6 FLOP
			global_a[i].x += dVec.w * dVec.x;
			global_a[i].y += dVec.w * dVec.y;
			global_a[i].z += dVec.w * dVec.z;
		} // 36 FLOP
	}
}

// NOTE: Before calling this function, the global_a array must be cleared!
__global__
	void calc_gravity_accel_naive_sym(const var4_t* global_x, const var_t* mass, var4_t* global_a)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (N > i)
	{
		var4_t dVec = {0.0, 0.0, 0.0, 0.0};
		for (int j = i+1; j < N; j++) 
		{
			// 3 FLOP
			dVec.x = global_x[j].x - global_x[i].x;
			dVec.y = global_x[j].y - global_x[i].y;
			dVec.z = global_x[j].z - global_x[i].z;
			// 5 FLOP
			dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2

			// sqrt operation takes approximately 20 FLOP
			var_t d = sqrt(dVec.w);								// = r
			// 2 FLOP
			var_t r_3 = 1.0 / (d*dVec.w);
			// 1 FLOP
			dVec.w = mass[j] * r_3;
			// 6 FLOP
			global_a[i].x += dVec.w * dVec.x;
			global_a[i].y += dVec.w * dVec.y;
			global_a[i].z += dVec.w * dVec.z;

			// 2 FLOP
			dVec.w = mass[i] * r_3;
			// 6 FLOP
			global_a[j].x -= dVec.w * dVec.x;
			global_a[j].y -= dVec.w * dVec.y;
			global_a[j].z -= dVec.w * dVec.z;
		} // 36 + 8 = 44 FLOP
	}
}

static __global__
	void calc_bc
	(
		uint32_t n,
		var_t M0,
		const pp_disk_t::body_metadata_t* bmd, 
		const pp_disk_t::param_t* p, 
		const var4_t* r,
		const var4_t* v,
		var4_t* R,
		var4_t* V
	)
{
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ var4_t _V[];

	_V[i].x = _V[i].y = _V[i].z = _V[i].w = 0.0;
	__syncthreads();

if (0 == threadIdx.x)
	printf("[   ] %24.16le %24.16le %24.16le\n", V[i].x, V[i].y, V[i].z);

	if (n > i && 0 < bmd[i].id)
	{
		R->x += p[i].mass * r[i].x;	R->y += p[i].mass * r[i].y;	R->z += p[i].mass * r[i].z;
printf("[%3u] %24.16le %24.16le %24.16le\n", i, R->x, R->y, R->z);

		V->x += p[i].mass * v[i].x;	V->y += p[i].mass * v[i].y;	V->z += p[i].mass * v[i].z;
	}

	__syncthreads();
	if (0 == threadIdx.x)
	{
		R->x /= M0;	R->y /= M0;	R->z /= M0;
		V->x /= M0;	V->y /= M0;	V->z /= M0;
	}
}


} /* namespace kernel */

static void test_print_position(int n, const var4_t* r)
{
	for (int tid = 0; tid < n; tid++)
	{
		//printf("r[%4d]: %f\n", tid, r[tid]);
		printf("r[%4d].x: %f\n", tid, r[tid].x);
		printf("r[%4d].y: %f\n", tid, r[tid].y);
		printf("r[%4d].z: %f\n", tid, r[tid].z);
		printf("r[%4d].w: %f\n", tid, r[tid].w);
	}
}

void allocate_device_vector(void **d_ptr, size_t size)
{
	CUDA_SAFE_CALL(cudaMalloc(d_ptr, size));
}


#if 0
// Study and test how a struct is placed into the memory
// Study and test how an array of struct is placed into the memory
int main(int argc, const char** argv)
{
	sim_data_t sim_data;

	sim_data.y.resize(2);
	sim_data.y[0] = new var4_t[8];
	sim_data.y[1] = new var4_t[8];

	sim_data.d_y.resize(2);

	var_t xmax =  1.0;
	var_t xmin = -1.0;
	for (int i = 0; i < 8; i++)
	{
		sim_data.y[0][i].x = xmin + (var_t)rand() / RAND_MAX * (xmax - xmin);
		sim_data.y[0][i].y = xmin + (var_t)rand() / RAND_MAX * (xmax - xmin);
		sim_data.y[0][i].z = xmin + (var_t)rand() / RAND_MAX * (xmax - xmin);
		sim_data.y[0][i].w = xmin + (var_t)rand() / RAND_MAX * (xmax - xmin);
	}
	test_print_position(8, sim_data.y[0]);

	// Allocate device pointer.
	CUDA_SAFE_CALL(cudaMalloc((void**) &(sim_data.d_y[0]), 8*sizeof(var4_t)));
	// Copy pointer content (position and mass) from host to device.
	CUDA_SAFE_CALL(cudaMemcpy(sim_data.d_y[0], sim_data.y[0], 8*sizeof(var4_t), cudaMemcpyHostToDevice));
	
	kernel_print_position<<<1, 8>>>(8, sim_data.d_y[0]);
	CUDA_CHECK_ERROR();

	// Allocate pointer.
	var4_t*	v = 0;
	v = (var4_t*)malloc(8 * sizeof(var4_t));
	memset(v, 0, 8 * sizeof(var4_t));

	// Allocate device pointer.
	var4_t*	d_v = 0;
	CUDA_SAFE_CALL(cudaMalloc((void**) &(d_v), 8*sizeof(var4_t)));

	// Copy pointer content from host to device.
	CUDA_SAFE_CALL(cudaMemcpy(d_v, v, 8*sizeof(var4_t), cudaMemcpyHostToDevice));

	kernel_print_vector<<<1, 8>>>(8, d_v);
	CUDA_CHECK_ERROR();

	free(v);
	cudaFree(d_v);

	delete[] sim_data.y[0];
	delete[] sim_data.y[1];
	cudaFree(sim_data.d_y[0]);

	return EXIT_SUCCESS;
}
#endif

#if 0
// Study and test the vector<vector <var4_t*> > type
int main(int argc, const char** argv)
{
	vector<vector <var4_t*> >	d_f(8);		// size of the outer vector container

	for (int i = 0; i < 8; i++)
	{
		d_f[i].resize(2);					// size of the inner vector container
		for (int j = 0; j < 2; j++)
		{
			d_f[i][j] = new var4_t[4];		// allocate 4 var4_t type element for each i, j pair
		}
	}
}
#endif

#if 0
// Study how to wrap a var4_t* into thrust vector to find the maximal element
// howto find the index of the maximum element
int main(int argc, const char** argv)
{
	{
		int data[6] = {1, 0, 2, 2, 1, 3};
		int *result = thrust::max_element(data, data + 6);
		printf("result: %p\n", result);
		printf("*result: %d\n", *result);
		int i = 0;
	}

	//! Holds the leading local truncation error for each variable
	vector<var_t*> d_err(1);

	// Create a raw pointer to device memory
	allocate_device_vector((void**)&d_err[0], 8 * sizeof(var_t));

	set_element_of_array<<<1, 8>>>(8, 2, d_err[0], 3.1415926535897932384626433832795);
	set_element_of_array<<<1, 8>>>(8, 5, d_err[0], 5.987654321);
	kernel_print_array<<<1, 8>>>(8, d_err[0]);
	cudaDeviceSynchronize();

	// Wrap raw pointer with a device_ptr
	thrust::device_ptr<var_t> d_ptr(d_err[0]);

	printf("d_ptr: %p\n", d_ptr.get());

	thrust::device_ptr<var_t> d_ptr_max_element = thrust::max_element(d_ptr, d_ptr + 8);

	var_t max_element = 0.0;
	// Copy the max element from device memory to host memory
	cudaMemcpy((void*)&max_element, (void*)d_ptr_max_element.get(), sizeof(var_t), cudaMemcpyDeviceToHost);
	printf("Value of max_element: %lf\n", max_element);
	
	printf("d_ptr_max_element: %p\n", d_ptr_max_element.get());
	int idx_of_max_element = (d_ptr_max_element.get() - d_ptr.get());
	cout << "idx_of_max_element: " << idx_of_max_element << endl;

	max_element = *thrust::max_element(d_ptr, d_ptr + 8);
	cout << "max_element: " << max_element << endl;

	// Use device_ptr in thrust algorithms
	thrust::fill(d_ptr, d_ptr + 8, (var_t)0.0);

	kernel_print_array<<<1, 8>>>(8, d_err[0]);
	cudaDeviceSynchronize();

	cudaFree(d_err[0]);
}
#endif

#if 0
// Strudy for the c_rkf8 integrator
// Howto pass an array of address to the kernel
int main()
{
	vector<vector <var4_t*> >	d_f(2);
	var4_t** d_dydt;

	const int n_total = 2;
	const int r_max = 3;

	// ALLOCATION
	ALLOCATE_DEVICE_VECTOR((void**)&d_dydt, 2*r_max*sizeof(var4_t*));
	for (int i = 0; i < 2; i++)
	{
		d_f[i].resize(r_max);
		for (int r = 0; r < r_max; r++) 
		{
			ALLOCATE_DEVICE_VECTOR((void**) &(d_f[i][r]), n_total*sizeof(var4_t));
			copy_vector_to_device((void*)&d_dydt[i*r_max + r], &d_f[i][r], sizeof(var_t*));
		}
	}
	// ALLOCATION END

	// PRINT ALLOCATED ADDRESSES
	for (int i = 0; i < 2; i++)
	{
		for (int r = 0; r < r_max; r++) 
		{
			printf("d_f[%d][%2d]: %p\n", i, r, d_f[i][r]);
		}
	}

	kernel_print_memory_address<<<1, 1>>>(2*r_max, (var_t*)d_dydt);
	CUDA_CHECK_ERROR();
	// PRINT ALLOCATED ADDRESSES END

	// POPULATE ALLOCATED STORAGE
	for (int i = 0; i < 2; i++)
	{
		for (int r = 0; r < r_max; r++) 
		{
			kernel_populate<<<1, 1>>>(n_total, pow(-1.0, i) * r, (var_t*)d_f[i][r]);
		}
	}
	CUDA_CHECK_ERROR();
	// POPULATE ALLOCATED STORAGE END

	// PRINT DATA
	printf("d_f[][]:\n");
	for (int i = 0; i < 2; i++)
	{
		for (int r = 0; r < r_max; r++) 
		{
			kernel_print_array<<<1, 1>>>(n_total, (var_t*)d_f[i][r]);
			CUDA_CHECK_ERROR();
		}
	}
	// PRINT DATA END

	// PRINT DATA
	printf("d_dydt[]:\n");
	for (int i = 0; i < 2; i++)
	{
		for (int r = 0; r < r_max; r++) 
		{
			var_t** d_ptr = (var_t**)d_dydt;
			kernel_print_array<<<1, 1>>>(n_total, i*r_max + r, d_ptr);
			CUDA_CHECK_ERROR();
		}
	}	
	// PRINT DATA END

	cudaDeviceReset();
}

#endif

#if 0

void cpy_cnstnt_to_dvc(const void* dst, const void *src, size_t count)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dst, src, count));
}

int main(int argc, const char** argv)
{
	cudaError_t cuda_status = cudaSuccess;

	int id_active = -1;
	int n_device = 0;

	cuda_status = cudaGetDeviceCount(&n_device);
	if (cudaSuccess != cuda_status)
	{
		printf("Error: %s\n", cudaGetErrorString(cuda_status));
		exit(0);
	}
	printf("The number of CUDA device(s) : %2d\n", n_device);

	if (1 > n_device)
	{
		printf("No CUDA device was found. Exiting to system.\n");
		exit(0);
	}

    for (int dev = 0; dev < n_device; ++dev)
    {
		printf("Setting the device id: %2d\n", dev);
        cuda_status = cudaSetDevice(dev);
		if (cudaSuccess != cuda_status)
		{
			printf("Error: %s\n", cudaGetErrorString(cuda_status));
			exit(0);
		}
		cuda_status = cudaGetDevice(&id_active);
		if (cudaSuccess != cuda_status)
		{
			printf("Error: %s\n", cudaGetErrorString(cuda_status));
			exit(0);
		}
		printf("The id of the active device: %2d\n", id_active);

        cudaDeviceProp deviceProp;
        cuda_status = cudaGetDeviceProperties(&deviceProp, dev);
		if (cudaSuccess != cuda_status)
		{
			printf("Error: %s\n", cudaGetErrorString(cuda_status));
			exit(0);
		}

        cout << "The code runs on " << deviceProp.name << " device:" << endl;
		//[domain]:[bus]:[device].
		cout << "The PCI domain ID of the device : " << deviceProp.pciDomainID << endl;
		cout << "The PCI bus ID of the device    : " << deviceProp.pciBusID << endl;
		cout << "The PCI device ID of the device : " << deviceProp.pciDeviceID << endl;

		char pciBusId[255];
		cuda_status = cudaDeviceGetPCIBusId(pciBusId, 255, dev);
		if (cudaSuccess != cuda_status)
		{
			printf("Error: %s\n", cudaGetErrorString(cuda_status));
			exit(0);
		}
		cout << "Identifier string for the device in the following format [domain]:[bus]:[device].[function]: " << pciBusId << endl;

		device_query(cout, dev);
	}

	// Copy data to the global device memory
	var_t* tmp = new var_t[1024];
	var_t* d_tmp = 0x0;
	ALLOCATE_DEVICE_VECTOR((void **)&(d_tmp), 1024*sizeof(var_t));
	printf("ALLOCATE_DEVICE_VECTOR succeeded\n");
	copy_vector_to_device(d_tmp, tmp, 1024*sizeof(var_t));
	printf("copy_vector_to_device succeeded\n");


	//! the hit centrum distance: inside this limit the body is considered to have hitted the central body and removed from the simulation [AU]
	//! the ejection distance: beyond this limit the body is removed from the simulation [AU]
	//! two bodies collide when their mutual distance is smaller than the sum of their radii multiplied by this number. Real physical collision corresponds to the value of 1.0.
	//! Contains the threshold values: hit_centrum_dst, ejection_dst, collision_factor
	var_t thrshld[THRESHOLD_N];
	thrshld[THRESHOLD_HIT_CENTRUM_DISTANCE]         = 0.1;
	thrshld[THRESHOLD_EJECTION_DISTANCE]            = 100.0;
	thrshld[THRESHOLD_RADII_ENHANCE_FACTOR]         = 5.0;
	thrshld[THRESHOLD_HIT_CENTRUM_DISTANCE_SQUARED] = SQR(thrshld[THRESHOLD_HIT_CENTRUM_DISTANCE]);
	thrshld[THRESHOLD_EJECTION_DISTANCE_SQUARED]    = SQR(thrshld[THRESHOLD_EJECTION_DISTANCE]);

	try
	{
		cpy_cnstnt_to_dvc(dc_threshold, thrshld, THRESHOLD_N*sizeof(var_t));
		kernel_print_constant_memory<<<1, 1>>>();
		cudaDeviceSynchronize();
	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}

	cudaDeviceSynchronize();

	return 0;
}
#endif

#if 0
// Test the ALLOCATE_VECTOR and FREE_VECTOR macro functions
void free_h_vector(void **ptr, const char *file, int line)
{
	delete[] *ptr;
	*ptr = (void *)0x0;
}

int main(int argc, const char** argv)
{
	var_t *p = 0x0;

	p = new var_t[10];
	free_h_vector((void **)&p, __FILE__, __LINE__);

	sim_data_t sd;
	event_data_t ed;

	var4_t* ptr = 0x0;
	size_t size = 1024 * sizeof(var4_t);
	bool cpu = true;

	ALLOCATE_VECTOR((void **)&(ptr), size, cpu);

	FREE_VECTOR((void **)&(ptr), cpu);

	return 0;
}
#endif

#if 0
// Test how to copy a struct to the device's constant memroy

int main(int argc, char** argv)
{

	analytic_gas_disk_params_t params;

	params.gas_decrease          = GAS_DENSITY_CONSTANT;
	params.t0                    = 1.0;
	params.t1                    = 2.0;
	params.e_folding_time        = 3.0;

	params.c_vth                 = 4.0;
	params.alpha                 = 5.0;
	params.mean_molecular_weight = 6.0;
	params.particle_diameter     = 7.0;

	params.eta.x = 0.0, params.eta.y = 8.0;
	params.rho.x = 0.0, params.rho.y = -10.0;
	params.sch.x = 0.0, params.sch.y = -20.0;
	params.tau.x = 0.0, params.tau.y = -30.0;

	params.mfp.x  = 0.0,  params.mfp.y = -40.0;
	params.temp.x = 0.0,  params.temp.y = -50.0;

	try
	{
		copy_constant_to_device((void*)&dc_anal_gd_params, (void*)&(params), sizeof(analytic_gas_disk_params_t));
		kernel_print_analytic_gas_disk_params<<<1, 1>>>();
		cudaDeviceSynchronize();
	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}

	return 0;
}
#endif

#if 0
// Test how to read a binary file
// Test the linear index determination function

#define N_SEC 1024
#define N_RAD  512
int calc_linear_index(var4_t& rVec, var_t* used_rad)
{
	const var_t dalpha = TWOPI / N_SEC;

	var_t r = sqrt(SQR(rVec.x) + SQR(rVec.y));
	if (     used_rad[0] > r)
	{
		return 0;
	}
	else if (used_rad[N_RAD] < r)
	{
		return N_RAD * N_SEC - 1;
	}
	else
	{
		// TODO: implement a fast search for the cell
		// IDEA: populate the used_rad with the square of the distance, since it is much faster to calculate r^2
		// Determine which ring contains r
		int i_rad = 0;
		int i_sec = 0;
		for (int k = 0; k < N_RAD; k++)
		{
			if (used_rad[k] <= r && r < used_rad[k+1])
			{
				i_rad = k;
				break;
			}
		}

		var_t alpha = (rVec.y >= 0.0 ? atan2(rVec.y, rVec.x) : TWOPI + atan2(rVec.y, rVec.x));
		i_sec =  alpha / dalpha;
		int i_linear = i_rad * N_SEC + i_sec;

		return i_linear;
	}
}


int main(int argc, char** argv)
{
	string path = "C:\\Work\\Projects\\red.cuda\\TestRun\\InputTest\\Test_Fargo_Gas\\gasdens0.dat";
	size_t n = 512*1024;
	vector<var_t> data(n);
	var_t* raw_data = data.data();
	var_t* used_rad = new var_t[513];

	try
	{
		file::load_binary_file(path, n, raw_data);

		//cout.precision(16);
		//cout.setf(ios::right);
		//cout.setf(ios::scientific);
		//for (int i = 0; i < data.size(); i += 1024)
		//{
		//	cout << setw(8) << i << ": " << setw(25) << data[i] << endl;
		//}
		//cout << setw(8) << data.size() << endl;
	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}

	path  = "C:\\Work\\Projects\\red.cuda\\TestRun\\InputTest\\Test_Fargo_Gas\\used_rad.dat";
	try
	{
		string result;
		file::load_ascii_file(path, result);

		cout.precision(16);
		cout.setf(ios::right);
		cout.setf(ios::scientific);
		int m = 0;
		for (uint32_t i = 0; i < result.length(); i++)
		{
			string num;
			num.resize(30);
			int k = 0;
			while (i < result.length() && k < 30 && result[i] != '\n')
			{
				num[k] = result[i];
				k++;
				i++;
			}
			num.resize(k);
			if (!tools::is_number(num))
			{
				cout << "num: '" << num << "'";
				throw string("Invalid number (" + num + ") in file '" + path + "'!\n");
			}
			var_t r = atof(num.c_str());
			//cout << r << endl;

			used_rad[m] = r;
			m++;
		}
	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}

	for (int i_rad = 0; i_rad < N_RAD; i_rad++)
	{
		var_t r = used_rad[i_rad];
		for (int i_sec = 0; i_sec < N_SEC; i_sec += 256)
		{
			var_t theta = i_sec * (TWOPI / N_SEC);
			var4_t rVec = {r * cos(theta), r * sin(theta), 0.0, 0.0};
			int lin_index = calc_linear_index(rVec, used_rad);
			printf("r=%25.15le gasdensity[%6d]=%25.16le\n", r, lin_index, raw_data[lin_index]);
		}
	}

	for (int i_rad = 0; i_rad < 220; i_rad++)
	{
		var_t r = i_rad * 0.5;
		for (int i_sec = 0; i_sec < N_SEC; i_sec += 256)
		{
			var_t theta = i_sec * (TWOPI / N_SEC);
			var4_t rVec = {r * cos(theta), r * sin(theta), 0.0, 0.0};
			int lin_index = calc_linear_index(rVec, used_rad);
			printf("r=%25.15le gasdensity[%6d]=%25.16le\n", r, lin_index, raw_data[lin_index]);
		}
	}

	return 0;
}
#undef N_SEC
#undef N_RAD

#endif

#if 0 // Calculate the volume density from the surface density used by FARGO

int main(int argc, char** argv)
{
	var_t Sigma_0_f = 2.38971e-5; /* Surface density of the gas at r = 1 [AU] */
	var_t h = 0.05;               /* Thickness over Radius in the disc */
	var_t rho_0_f  = (1.0 / sqrt(TWOPI)) * Sigma_0_f / h;  /* Volume density of the gas at r = 1 [AU] */

	printf("rho_0_f = %25.16le [M/AU^3]\n", rho_0_f);
	printf("rho_0_f = %25.16le [g/cm^3]\n", rho_0_f * constants::SolarPerAu3ToGramPerCm3);
}

#endif

#if 0
/*
 * Implement and benchmark the tile-calculation method for the N-body gravity kernel.
 * Compare times needed by the tile and naive implementation.
 */

void cpu_calc_grav_accel_naive(int n, const var4_t* r, const var_t* mass, var4_t* a)
{
	memset(a, 0, n*sizeof(var4_t));

	for (int i = 0; i < n; i++)
	{
		//if (GTID == i)
		//{
		//	printf("[%2d == i] rVec = [%16.8le, %16.8le, %16.8le]\n", GTID, r[i].x, r[i].y, r[i].z);
		//}

		var4_t dVec = {0.0, 0.0, 0.0, 0.0};
		for (int j = 0; j < n; j++) 
		{
			if (i == j)
			{
				continue;
			}
			// r_ij 3 FLOP
			dVec.x = r[j].x - r[i].x;
			dVec.y = r[j].y - r[i].y;
			dVec.z = r[j].z - r[i].z;
			// 5 FLOP
			dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2

			// 20 FLOP
			var_t d = sqrt(dVec.w);								// = r
			// 2 FLOP
			var_t r_3 = 1.0 / (d*dVec.w);
			// 1 FLOP
			dVec.w = mass[j] * r_3;
			// 6 FLOP

			a[i].x += dVec.w * dVec.x;
			a[i].y += dVec.w * dVec.y;
			a[i].z += dVec.w * dVec.z;

			//if (GTID == i)
			//{
			//	printf("[%2d == i, j = %2d] dVec = [%16.8le, %16.8le, %16.8le] mj = %16.8le d = %16.8le w = %16.8le a = [%16.8le, %16.8le, %16.8le] \n", GTID, j, dVec.x, dVec.y, dVec.z, mass[j], d, dVec.w, a[i].x, a[i].y, a[i].z);
			//}
		} // 36 FLOP
	}
}

void cpu_calc_grav_accel_naive_sym(int n, const var4_t* r, const var_t* mass, var4_t* a)
{
	memset(a, 0, n*sizeof(var4_t));

	for (int i = 0; i < n; i++)
	{
		var4_t dVec = {0.0, 0.0, 0.0, 0.0};
		for (int j = i+1; j < n; j++) 
		{
			// r_ij 3 FLOP
			dVec.x = r[j].x - r[i].x;
			dVec.y = r[j].y - r[i].y;
			dVec.z = r[j].z - r[i].z;
			// 5 FLOP
			dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2

			// 20 FLOP
			var_t d = sqrt(dVec.w);								// = r
			// 2 FLOP
			var_t r_3 = 1.0 / (d*dVec.w);
			// 1 FLOP
			dVec.w = mass[j] * r_3;
			// 6 FLOP
			a[i].x += dVec.w * dVec.x;
			a[i].y += dVec.w * dVec.y;
			a[i].z += dVec.w * dVec.z;

			// 2 FLOP
			dVec.w = mass[i] * r_3;
			// 6 FLOP
			a[j].x -= dVec.w * dVec.x;
			a[j].y -= dVec.w * dVec.y;
			a[j].z -= dVec.w * dVec.z;
		} // 36 + 8 = 44 FLOP
	}
}

float gpu_calc_grav_accel_naive_benchmark(int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a)
{
	cudaEvent_t start, stop;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	dim3 grid((N + n_tpb - 1)/n_tpb);
	dim3 block(n_tpb);

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	kernel::calc_gravity_accel_naive<<<grid, block>>>(d_x, d_m, d_a);
	CUDA_CHECK_ERROR();

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float elapsed_time = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));

	return elapsed_time;
}

float gpu_calc_grav_accel_naive_sym_benchmark(int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a)
{
	cudaEvent_t start, stop;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	dim3 grid((N + n_tpb - 1)/n_tpb);
	dim3 block(n_tpb);

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	CUDA_SAFE_CALL(cudaMemset(d_a, 0, N*sizeof(var4_t)));
	kernel::calc_gravity_accel_naive_sym<<<grid, block>>>(d_x, d_m, d_a);
	CUDA_CHECK_ERROR();

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float elapsed_time = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));

	return elapsed_time;
}

float gpu_calc_grav_accel_tile_benchmark(int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a)
{
	cudaEvent_t start, stop;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	dim3 grid((N + n_tpb - 1)/n_tpb);
	dim3 block(n_tpb);

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	kernel::calc_gravity_accel_tile<<<grid, block, n_tpb * sizeof(var4_t)>>>(n_tpb, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float elapsed_time = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));

	return elapsed_time;
}

// The same as above but blockDim.x is used instead of tile_size
float gpu_calc_grav_accel_tile_benchmark_2(int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a)
{
	cudaEvent_t start, stop;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	dim3 grid((N + n_tpb - 1)/n_tpb);
	dim3 block(n_tpb);

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	kernel::calc_gravity_accel_tile<<<grid, block, n_tpb * sizeof(var4_t)>>>(d_x, d_m, d_a);
	CUDA_CHECK_ERROR();

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float elapsed_time = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));

	return elapsed_time;
}

void benchmark_CPU_and_kernels(const var4_t* d_x, const var_t* d_m, var4_t* d_a, const var4_t* h_x, const var_t* h_m, var4_t* h_a)
{
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));

	int half_warp_size = deviceProp.warpSize/2;

	{
		cout << endl << "CPU Gravity acceleration: Naive calculation:" << endl;

		clock_t t_start = clock();
		cpu_calc_grav_accel_naive(N, h_x, h_m, h_a);
		ttt_t elapsed_time = ((double)(clock() - t_start)/(double)CLOCKS_PER_SEC) * 1000.0; // [ms]

		printf("[%3d] dt: %10.6le [ms]\n", 1, elapsed_time);
		printf("\n");
	}
	{
		cout << endl << "CPU Gravity acceleration: Naive symmetric calculation:" << endl;

		clock_t t_start = clock();
		cpu_calc_grav_accel_naive_sym(N, h_x, h_m, h_a);
		ttt_t elapsed_time = ((double)(clock() - t_start)/(double)CLOCKS_PER_SEC) * 1000.0; // [ms]

		printf("[%3d] dt: %10.6le [ms]\n", 1, elapsed_time);
		printf("\n");
	}
	{
		cout << endl << "GPU Gravity acceleration: Naive calculation:" << endl;

		vector<float2> execution_time;
		uint32_t n_pass = 0;
		for (int n_tpb = half_warp_size; n_tpb <= deviceProp.maxThreadsPerBlock/2; n_tpb += half_warp_size)
		{
			clock_t t_start = clock();
			float cu_elt = gpu_calc_grav_accel_naive_benchmark(n_tpb, d_x, d_m, d_a);
			ttt_t elapsed_time = ((double)(clock() - t_start)/(double)CLOCKS_PER_SEC) * 1000.0; // [ms]

			float2 exec_t = {(float)elapsed_time, cu_elt};
			execution_time.push_back(exec_t);

			printf("[%3d] dt: %10.6le (%10.6le) [ms]\n", n_tpb, elapsed_time, cu_elt);
			n_pass++;
		}
		float min_y = 1.0e10;
		int min_idx = 0;
		for (uint32_t i = 0; i < n_pass; i++)
		{
			if (min_y > execution_time[i].y)
			{
				min_y = execution_time[i].y;
				min_idx = i;
			}
		}
		cout << "Minimum at n_tpb = " << ((min_idx + 1) * half_warp_size) << ", where execution time is: " << execution_time[min_idx].y << " [ms]" << endl;
		printf("\n");
	}
	{
		cout << endl << "GPU Gravity acceleration: Naive symmetric calculation:" << endl;

		vector<float2> execution_time;
		uint32_t n_pass = 0;
		for (int n_tpb = half_warp_size; n_tpb <= deviceProp.maxThreadsPerBlock/2; n_tpb += half_warp_size)
		{
			clock_t t_start = clock();
			float cu_elt = gpu_calc_grav_accel_naive_sym_benchmark(n_tpb, d_x, d_m, d_a);
			ttt_t elapsed_time = ((double)(clock() - t_start)/(double)CLOCKS_PER_SEC) * 1000.0; // [ms]

			float2 exec_t = {(float)elapsed_time, cu_elt};
			execution_time.push_back(exec_t);

			printf("[%3d] dt: %10.6le (%10.6le) [ms]\n", n_tpb, elapsed_time, cu_elt);
			n_pass++;
		}
		float min_y = 1.0e10;
		int min_idx = 0;
		for (uint32_t i = 0; i < n_pass; i++)
		{
			if (min_y > execution_time[i].y)
			{
				min_y = execution_time[i].y;
				min_idx = i;
			}
		}
		cout << "Minimum at n_tpb = " << ((min_idx + 1) * half_warp_size) << ", where execution time is: " << execution_time[min_idx].y << " [ms]" << endl;
		printf("\n");
	}
	{
		cout << "GPU Gravity acceleration: Tile calculation:" << endl;

		vector<float2> execution_time;
		uint32_t n_pass = 0;
		for (int n_tpb = half_warp_size; n_tpb <= deviceProp.maxThreadsPerBlock/2; n_tpb += half_warp_size)
		{
			clock_t t_start = clock();
			float cu_elt = gpu_calc_grav_accel_tile_benchmark(n_tpb, d_x, d_m, d_a);
			ttt_t elapsed_time = ((double)(clock() - t_start)/(double)CLOCKS_PER_SEC) * 1000.0; // [ms]

			float2 exec_t = {(float)elapsed_time, cu_elt};
			execution_time.push_back(exec_t);

			printf("[%3d] dt: %10.6le (%10.6le) [ms]\n", n_tpb, elapsed_time, cu_elt);
			n_pass++;
		}
		float min_y = 1.0e10;
		int min_idx = 0;
		for (uint32_t i = 0; i < n_pass; i++)
		{
			if (min_y > execution_time[i].y)
			{
				min_y = execution_time[i].y;
				min_idx = i;
			}
		}
		cout << "Minimum at n_tpb = " << ((min_idx + 1) * half_warp_size) << ", where execution time is: " << execution_time[min_idx].y << " [ms]" << endl;
		printf("\n");
	}
	{
		cout << "GPU Gravity acceleration: Tile calculation (without the explicit use of tile_size):" << endl;

		vector<float2> execution_time;
		uint32_t n_pass = 0;
		for (int n_tpb = half_warp_size; n_tpb <= deviceProp.maxThreadsPerBlock/2; n_tpb += half_warp_size)
		{
			clock_t t_start = clock();
			float cu_elt = gpu_calc_grav_accel_tile_benchmark_2(n_tpb, d_x, d_m, d_a);
			ttt_t elapsed_time = ((double)(clock() - t_start)/(double)CLOCKS_PER_SEC) * 1000.0; // [ms]

			float2 exec_t = {(float)elapsed_time, cu_elt};
			execution_time.push_back(exec_t);

			printf("[%3d] dt: %10.6le (%10.6le) [ms]\n", n_tpb, elapsed_time, cu_elt);
			n_pass++;
		}
		float min_y = 1.0e10;
		int min_idx = 0;
		for (uint32_t i = 0; i < n_pass; i++)
		{
			if (min_y > execution_time[i].y)
			{
				min_y = execution_time[i].y;
				min_idx = i;
			}
		}
		cout << "Minimum at n_tpb = " << ((min_idx + 1) * half_warp_size) << ", where execution time is: " << execution_time[min_idx].y << " [ms]" << endl;
		printf("\n");
	}
}

bool compare_vectors(int n, const var4_t* v1, const var4_t* v2, var_t tolerance, bool verbose)
{
	bool success = true;

	for (int i = 0; i < n; i++)
	{
		var_t diff = fabs(v1[i].x - v2[i].x);
		var_t rel  = diff / v1[i].x;
		if (tolerance <= diff)
		{
			if (success && verbose) printf("\n");
			if (verbose) printf("\tError: i = [%5d] v1.x = %16.8le v2.x = %16.8le D = %16.8le rel = %16.8le\n", i, v1[i].x, v2[i].x, diff, rel);
			success = false;
		}
		diff = fabs(v1[i].y - v2[i].y);
		rel  = diff / v1[i].y;
		if (tolerance <= diff)
		{
			if (success && verbose) printf("\n");
			if (verbose) printf("\tError: i = [%5d] v1.y = %16.8le v2.y = %16.8le D = %16.8le rel = %16.8le\n", i, v1[i].y, v2[i].y, diff, rel);
			success = false;
		}
		diff = fabs(v1[i].z - v2[i].z);
		rel  = diff / v1[i].z;
		if (tolerance <= diff)
		{
			if (success && verbose) printf("\n");
			if (verbose) printf("\tError: i = [%5d] v1.z = %16.8le v2.x = %16.8le D = %16.8le rel = %16.8le\n", i, v1[i].z, v2[i].z, diff, rel);
			success = false;
		}
	}
	if (success)
	{
		printf("The results are identical with a tolerance of %20.16le.\n", tolerance);
	}
	else
	{
		printf("The results are different with a tolerance of %20.16le.\n", tolerance);
	}
	return success;
}

void compare_results(int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a, const var4_t* h_x, const var_t* h_m, var4_t* h_a, var4_t* h_at, var_t tolerance)
{
	bool result = true;

	printf("\nComparing the computations of the gravitational accelerations performed\non the CPU with two different functions naive() and naive_sym(): ");
	cpu_calc_grav_accel_naive(    N, h_x, h_m, h_a );
	cpu_calc_grav_accel_naive_sym(N, h_x, h_m, h_at);

	result = compare_vectors(N, h_a, h_at, tolerance, false);

	memset(h_a,  0, N*sizeof(var4_t));
	memset(h_at, 0, N*sizeof(var4_t));

	dim3 grid((N + n_tpb - 1)/n_tpb);
	dim3 block(n_tpb);

	printf("\nComparing the computations of the gravitational accelerations performed on the CPU and GPU:\n");
	printf("1. CPU naive vs GPU naive    : ");
	cpu_calc_grav_accel_naive(N, h_x, h_m, h_a);
	kernel::calc_gravity_accel_naive<<<grid, block>>>(d_x, d_m, d_a);
	CUDA_CHECK_ERROR();
	copy_vector_to_host(h_at, d_a, N*sizeof(var4_t));
	result = compare_vectors(N, h_a, h_at, tolerance, false);
	memset(h_at, 0, N*sizeof(var4_t));

	printf("2. CPU naive vs GPU naive_sym: ");
	CUDA_SAFE_CALL(cudaMemset(d_a, 0, N*sizeof(var4_t)));
	kernel::calc_gravity_accel_naive_sym<<<grid, block>>>(d_x, d_m, d_a);
	CUDA_CHECK_ERROR();
	copy_vector_to_host(h_at, d_a, N*sizeof(var4_t));
	result = compare_vectors(N, h_a, h_at, tolerance, false);
	memset(h_at, 0, N*sizeof(var4_t));

	printf("3. CPU naive vs GPU tile     : ");
	kernel::calc_gravity_accel_tile<<<grid, block, n_tpb*sizeof(var4_t)>>>(n_tpb, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();
	copy_vector_to_host(h_at, d_a, N*sizeof(var4_t));
	result = compare_vectors(N, h_a, h_at, tolerance, false);
	memset(h_at, 0, N*sizeof(var4_t));

	printf("4. CPU naive vs GPU tile (without the explicit use of tile_size): ");
	kernel::calc_gravity_accel_tile<<<grid, block, n_tpb*sizeof(var4_t)>>>(d_x, d_m, d_a);
	CUDA_CHECK_ERROR();
	copy_vector_to_host(h_at, d_a, N*sizeof(var4_t));
	result = compare_vectors(N, h_a, h_at, tolerance, false);
	memset(h_at, 0, N*sizeof(var4_t));
}

int main(int argc, char** argv)
{
	var4_t* h_x = 0x0;
	var4_t* h_a = 0x0;
	var_t* h_m = 0x0;

	var4_t* d_x = 0x0;
	var4_t* d_a = 0x0;
	var_t* d_m = 0x0;

	var4_t* h_at = 0x0;

	try
	{
		device_query(cout, 0, false);

		ALLOCATE_HOST_VECTOR((void**)&h_x,  N*sizeof(var4_t)); 
		ALLOCATE_HOST_VECTOR((void**)&h_a,  N*sizeof(var4_t)); 
		ALLOCATE_HOST_VECTOR((void**)&h_at, N*sizeof(var4_t)); 
		ALLOCATE_HOST_VECTOR((void**)&h_m,  N*sizeof(var_t)); 

		ALLOCATE_DEVICE_VECTOR((void**)&d_x, N*sizeof(var4_t)); 
		ALLOCATE_DEVICE_VECTOR((void**)&d_a, N*sizeof(var4_t)); 
		ALLOCATE_DEVICE_VECTOR((void**)&d_m, N*sizeof(var_t)); 

		for (int i = 0; i < N; i++)
		{
			h_x[i].x = -50.0 + ((var_t)rand() / RAND_MAX) * 100.0;
			h_x[i].y = -50.0 + ((var_t)rand() / RAND_MAX) * 100.0;
			h_x[i].z = -50.0 + ((var_t)rand() / RAND_MAX) * 100.0;
			h_x[i].w = 0.0;

			h_m[i] = 1.0 * i;
		}

		copy_vector_to_device(d_x, h_x, N * sizeof(var4_t));
		copy_vector_to_device(d_m, h_m, N * sizeof(var_t));

#if 0
		{
			cpu_calc_grav_accel_naive(N, h_x, h_m, h_a);

			int n_tpb = 4;
			dim3 grid((N + n_tpb - 1)/n_tpb);
			dim3 block(n_tpb);

			kernel::calc_gravity_accel_tile<<<grid, block, n_tpb * sizeof(var4_t)>>>(d_x, d_m, d_a);
			CUDA_CHECK_ERROR();

			copy_vector_to_host(h_at, d_a, N*sizeof(var4_t));
			bool result = compare_vectors(N, h_a, h_at, 1.0e-15, true);
		}
#endif


// Compare the results computed on the CPU with 2 different methods
// and those computed on the GPU with 3 different methods
#if 1
		// Compare results computed on the CPU with those computed on the GPU
		int n_tpb = min(N, 4);
		compare_results(n_tpb, d_x, d_m, d_a, h_x, h_m, h_a, h_at, 1.0e-15);

		benchmark_CPU_and_kernels(d_x, d_m, d_a, h_x, h_m, h_a);
#endif

		FREE_HOST_VECTOR((void**)&h_x);
		FREE_HOST_VECTOR((void**)&h_a);
		FREE_HOST_VECTOR((void**)&h_m);

		FREE_DEVICE_VECTOR((void**)&d_x);
		FREE_DEVICE_VECTOR((void**)&d_a);
		FREE_DEVICE_VECTOR((void**)&d_m);

		FREE_HOST_VECTOR((void**)&h_at);
	}
	catch(const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}

	return (EXIT_SUCCESS);
}
#endif

#if 0
/*
 *  Howto call a template function which is in a library. i.e. in the redutilcu.lib
 */

int main()
{
	string result;

	int int_number = -123;
	result = redutilcu::number_to_string(int_number);
	cout << "int_number: " << result << endl;

	double dbl_number = 123.123;
	result = redutilcu::number_to_string(dbl_number);
	cout << "dbl_number: " << result << endl;

/*
 * A megoldas az volt, hogy peldanyositani kellett a különbözo tipusokkal a template fuggvenyt.
 */
}

#endif

#if 0
/*
 *  Howto extract data from string
 */

int main()
{
	string data = "       1                           star  0   0.0000000000000000e+000   1.0000000000000000e+000   4.6491301477493566e-003   2.3757256065676026e+006   0.0000000000000000e+000  0   0.0000000000000000e+000   3.6936661279371897e-008   7.2842030950759070e-007   0.0000000000000000e+000  -4.8439964006954156e-009   1.1641554643378494e-009   0.0000000000000000e+000";
	stringstream ss;
	ss << data;

	int id = -1;
	string name;
	int bodytype = 0;
	ttt_t t = 0.0;

	string path = "C:\\Work\\red.cuda.Results\\Dvorak\\2D\\NewRun_2\\Run_cf4.0\\D_gpu_ns_as_RKF8_result.txt";
	ifstream input(path);
	if (input) 
	{
		int ns, ngp, nrp, npp, nspl, npl, ntp;
		input >> ns >> ngp >> nrp >> npp >> nspl >> npl >> ntp;
	}
	else 
	{
		throw string("Cannot open " + path + ".");
	}

	ss >> id >> name >> bodytype >> t;

	cout << "id: " << id << endl;
	cout << "name: " << name << endl;
	cout << "bodytype: " << bodytype << endl;
	cout << "t: " << t << endl;
}

#endif

#if 0
/*
 *  Howto write/read data in binary representation to/from a file
 */
void print_state(const std::ios& stream)
{
	std::cout << " good()=" << stream.good();
	std::cout << " eof()=" << stream.eof();
	std::cout << " fail()=" << stream.fail();
	std::cout << " bad()=" << stream.bad();
}

void open_streams(ofstream** output)
{
	string path = "C:\\Work\\Projects\\red.cuda\\TestRun\\DDD\\hello.txt";

	output[OUTPUT_NAME_RESULT] = new ofstream(path.c_str(), ios::out);
	if (!(*output[OUTPUT_NAME_RESULT])) 
	{
		throw string("Cannot open " + path + ".");
	}
	print_state(*output[OUTPUT_NAME_RESULT]);

	*output[OUTPUT_NAME_RESULT] << "Hello World!";
}

int main()
{

	try
	{
		ofstream* output[OUTPUT_NAME_N];
		for (uint32_t i = 0; i < OUTPUT_NAME_N; i++)
		{
			output[i] = 0x0;
		}

		open_streams(output);
	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}


	cout << "           sizeof(bool): " << sizeof(bool) << endl;
	cout << "            sizeof(int): " << sizeof(int) << endl;
	cout << "        sizeof(int16_t): " << sizeof(int16_t) << endl;
	cout << "        sizeof(int32_t): " << sizeof(int32_t) << endl;
	cout << "        sizeof(int64_t): " << sizeof(int64_t) << endl;
	cout << "       sizeof(uint16_t): " << sizeof(uint16_t) << endl;
	cout << "       sizeof(uint32_t): " << sizeof(uint32_t) << endl;
	cout << "       sizeof(uint64_t): " << sizeof(uint64_t) << endl;
	cout << "         sizeof(double): " << sizeof(double) << endl;
	cout << "          sizeof(var_t): " << sizeof(var_t) << endl;
	cout << "          sizeof(ttt_t): " << sizeof(ttt_t) << endl;
	cout << "          sizeof(var4_t): " << sizeof(var4_t) << endl;
	cout << "        sizeof(pp_disk_t::param_t): " << sizeof(pp_disk_t::param_t) << endl;
	cout << "sizeof(body_metadata_t): " << sizeof(body_metadata_t) << endl;

	uint32_t n_bodies[] = {1, 2, 3, 4, 5, 6, 7};
	char name_buffer[30];
	memset(name_buffer, 0, sizeof(name_buffer));
	cout << "sizeof(name_buffer): " << sizeof(name_buffer) << endl;


	const int n_total = 2;
	sim_data_t *sim_data = new sim_data_t();
	// These will be only aliases to the actual storage space either in the HOST or DEVICE memory
	sim_data->y.resize(2);
	sim_data->yout.resize(2);
	allocate_host_storage(sim_data, n_total);

	var4_t* r = sim_data->h_y[0];
	var4_t* v = sim_data->h_y[1];
	pp_disk_t::param_t* p = sim_data->h_p;
	body_metadata_t* bmd = sim_data->h_body_md;
	ttt_t* epoch = sim_data->h_epoch;

	std::vector<string> body_names;
	// populate sim_data
	{
		int bdy_idx = 0;
		body_names.push_back("Star");
		r[bdy_idx].x = 1.0, r[bdy_idx].y = 2.0, r[bdy_idx].z = 3.0, r[bdy_idx].w = 4.0;
		v[bdy_idx].x = 1.0e-1, v[bdy_idx].y = 2.0e-1, v[bdy_idx].z = 3.0e-1, v[bdy_idx].w = 4.0e-1;
		p[bdy_idx].mass = 1.0, p[bdy_idx].radius = 2.0, p[bdy_idx].density = 3.0, p[bdy_idx].cd = 4.0;
		bmd[bdy_idx].id = 1, bmd[bdy_idx].body_type = BODY_TYPE_STAR, bmd[bdy_idx].mig_stop_at = MIGRATION_TYPE_NO, bmd[bdy_idx].mig_stop_at = 0.5;
		epoch[bdy_idx] = 1.2;

		bdy_idx++;
		body_names.push_back("Jupiter");
		r[bdy_idx].x = 1.0, r[bdy_idx].y = 2.0, r[bdy_idx].z = 3.0, r[bdy_idx].w = 4.0;
		v[bdy_idx].x = 1.0e-1, v[bdy_idx].y = 2.0e-1, v[bdy_idx].z = 3.0e-1, v[bdy_idx].w = 4.0e-1;
		p[bdy_idx].mass = 1.0, p[bdy_idx].radius = 2.0, p[bdy_idx].density = 3.0, p[bdy_idx].cd = 4.0;
		bmd[bdy_idx].id = 2, bmd[bdy_idx].body_type = BODY_TYPE_GIANTPLANET, bmd[bdy_idx].mig_stop_at = MIGRATION_TYPE_NO, bmd[bdy_idx].mig_stop_at = 0.5;
		epoch[bdy_idx] = 1.2;
	}

	// Write out binary data
	string path = "C:\\Work\\red.cuda.Results\\binary.dat";
	ofstream* sout;
	sout = new ofstream(path.c_str(), ios::out | ios::binary);
	if (sout)
	{
		print_state(*sout);
		cout << endl;

		for (uint32_t type = 0; type < BODY_TYPE_N; type++)
		{
			sout->write((char*)&n_bodies[type], sizeof(n_bodies[type]));
		}
		for (int i = 0; i < n_total; i++)
		{
			int orig_idx = bmd[i].id - 1;
			memset(name_buffer, 0, sizeof(name_buffer));
			strcpy(name_buffer, body_names[orig_idx].c_str());

			sout->write((char*)&epoch[i], sizeof(ttt_t));
			sout->write(name_buffer,      sizeof(name_buffer));
			sout->write((char*)&bmd[i],   sizeof(body_metadata_t));
			sout->write((char*)&p[i],     sizeof(pp_disk_t::param_t));
			sout->write((char*)&r[i],     sizeof(var4_t));
			sout->write((char*)&v[i],     sizeof(var4_t));
		}

		sout->close();
	}
	else
	{
		print_state(*sout);
		cout << endl;

		cerr << "Could not open " << path << "." << endl;
	}

	// Clear all data
	for (char type = 0; type < BODY_TYPE_N; type++)
	{
		n_bodies[type] = 0;
	}
	deallocate_host_storage(sim_data);
	sim_data = 0x0;
	body_names.clear();

	sim_data = new sim_data_t();
	// These will be only aliases to the actual storage space either in the HOST or DEVICE memory
	sim_data->y.resize(2);
	sim_data->yout.resize(2);
	allocate_host_storage(sim_data, n_total);

	r = sim_data->h_y[0];
	v = sim_data->h_y[1];
	p = sim_data->h_p;
	bmd = sim_data->h_body_md;
	epoch = sim_data->h_epoch;

	path = "C:\\Work\\Projects\\red.cuda\\TestRun\\DumpTest\\TwoBody\\Continue\\dump_0_1_1_0_0_0_0_0_.dat";
	ifstream sin;
	sin.open(path.c_str(), ios::in | ios::binary);
	if (sin)
	{
		print_state(sin);
		cout << endl;

		// Read back the data
		for (uint32_t type = 0; type < BODY_TYPE_N; type++)
		{
			sin.read((char*)&n_bodies[type], sizeof(n_bodies[type]));
		}
		// Print the data
		for (char type = 0; type < BODY_TYPE_N; type++)
		{
			cout << n_bodies[type] << endl;
		}

		for (uint32_t i = 0; i < n_total; i++)
		{
			memset(name_buffer, 0, sizeof(name_buffer));

			sin.read((char*)&epoch[i],  1*sizeof(ttt_t));
			sin.read(name_buffer,      30*sizeof(char));
			sin.read((char*)&bmd[i],    1*sizeof(body_metadata_t));
			sin.read((char*)&p[i],      1*sizeof(pp_disk_t::param_t));
			sin.read((char*)&r[i],      1*sizeof(var4_t));
			sin.read((char*)&v[i],      1*sizeof(var4_t));

			body_names.push_back(name_buffer);

			cout << epoch[i] << endl;
			cout << name_buffer << endl;
			tools::print_vector(&r[i]);
			tools::print_vector(&v[i]);
			tools::print_parameter(&p[i]);
			tools::print_body_metadata(&bmd[i]);
		}
		sin.close();
	}
	else
	{
		print_state(sin);
		cout << endl;

		cerr << "Could not open " << path.c_str() << "." << endl;
	}
}

#endif

#if 0
/*
 * Test the allocate/memset/copy/free cycle on the device.
 */
void call_kernel_print_sim_data(uint32_t n, sim_data_t* sim_data)
{
	printf("**********************************************************************\n");
	printf("                 DATA ON THE DEVICE                                   \n");
	printf("**********************************************************************\n\n");

	printf("position:\n");
	print_vector<<<1, 1>>>(n, sim_data->d_y[0]);
	CUDA_CHECK_ERROR();
}

void print_sim_data(uint32_t n, sim_data_t* sim_data)
{
	printf("**********************************************************************\n");
	printf("                 DATA ON THE HOST                                     \n");
	printf("**********************************************************************\n\n");

	printf("position:\n");
	const var4_t* v = sim_data->h_y[0];
	for (uint32_t i = 0; i < n; i++)
	{
		printf("%5d %20.16lf, %20.16lf, %20.16lf, %20.16lf\n", i, v[i].x, v[i].y, v[i].z, v[i].w);
	}
}

void copy_to_device(uint32_t n, sim_data_t* sim_data)
{
	for (int i = 0; i < 2; i++)
	{
		copy_vector_to_device((void *)sim_data->d_y[i],	    (void *)sim_data->h_y[i], n*sizeof(var4_t));
	}

	copy_vector_to_device((void *)sim_data->d_p,		(void *)sim_data->h_p,		  n*sizeof(pp_disk_t::param_t));
	copy_vector_to_device((void *)sim_data->d_body_md,	(void *)sim_data->h_body_md,  n*sizeof(body_metadata_t));
	copy_vector_to_device((void *)sim_data->d_epoch,	(void *)sim_data->h_epoch,	  n*sizeof(ttt_t));
}

int main()
{
	uint32_t n_body = 0;
	sim_data_t* sd = new sim_data_t;
	
	uint32_t n_bodies[] = {1, 99, 0, 0, 0, 0 ,0};
	for (int i = 0; i < 7; i++)
	{
		n_body += n_bodies[i];
	}

	allocate_host_storage(sd, n_body);

	tools::populate_data(n_bodies, sd);
	print_sim_data(n_body, sd);
	
	for (int i = 0; i < 10; i++)
	{
		allocate_device_storage(sd, n_body);

		call_kernel_print_sim_data(n_body, sd);

		copy_to_device(n_body, sd);

		call_kernel_print_sim_data(n_body, sd);

		deallocate_device_storage(sd);
	}

	return EXIT_SUCCESS;
}

#endif

#if 0
/*
 * Test the CUDA_SAFE_CALL() and CUDA_CHECK_ERROR() macro functions
 */
int main()
{
	try
	{
		cudaError_t cuda_status = cudaSuccess;
		CUDA_SAFE_CALL(cuda_status);
	}
	catch(const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}

	try
	{
		cudaError_t cuda_status = cudaErrorInitializationError;
		CUDA_SAFE_CALL(cuda_status);
	}
	catch(const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}

	try
	{
		CUDA_CHECK_ERROR();
	}
	catch(const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}

	return (EXIT_SUCCESS);
}
#endif

#if 0
/*
 * Test (*event_counter)++ expression's behaviour
 */
int main()
{
	int *event_counter = 0x0;

	event_counter = (int*)malloc(sizeof(int));

	*event_counter = 0;
	int k = (*event_counter)++;
	printf("k = %d\t*event_counter = %d\n", k, *event_counter);

	k = (*event_counter)++;
	printf("k = %d\t*event_counter = %d\n", k, *event_counter);

	k = (*event_counter)++;
	printf("k = %d\t*event_counter = %d\n", k, *event_counter);

	free(event_counter);
}
#endif

#if 0
/*
 * Test to parse and split the cpuinfo file on GNU/Linux systems
 */
void parse_cpu_info(vector<string>& data)
{
	char delimiter = ':';
	string line;

	size_t pos0 = 0;
	size_t pos1 = data[0].find_first_of('\n');;
	do
	{
		line = data[0].substr(pos0, pos1 - pos0);
		size_t p0 = line.find_first_of(delimiter);
		string key = line.substr(0, p0);
		string value = line.substr(p0+1, line.length());

printf("line = '%s'\nkey = '%s' value = '%s'\n", line.c_str(), key.c_str(), value.c_str());
		key = tools::trim(key);
		value = tools::trim(value);
printf("key = '%s' value = '%s'\n", key.c_str(), value.c_str());

		if ("model name" == key)
		{
printf("key 'model name' was found with value: '%s'\n", value.c_str());
			string cpu_info_model_name = value;
		}

		// Increaes by 1 in order to skip the newline at the end of the previous string
		pos0 = pos1 + 1;
		pos1 = data[0].find_first_of('\n', pos0+1);
	} while (pos1 != string::npos && pos1 <= data[0].length());
}


int main(int argc, const char** argv, const char** env)
{
	vector<string> result;
	string data;

	/* presume POSIX */
	//string path = "/proc/cpuinfo";
	string path = "C:\\Work\\Projects\\red.cuda\\TestRun\\CPU_info\\cpuinfo";

	ifstream input(path.c_str());
	if (!input)
	{
		throw string("Cannot open " + path + ".");
	}
	string line;
	while (getline(input, line))
	{
		if (line.empty())
		{
			result.push_back(data);
			data.clear();
		}
		else
		{
			data += line + '\n';
		}
	}
	result.push_back(data);
	data.clear();

	parse_cpu_info(result);
}

#endif

#if 0
/*
 * Print the sizeof int and other int types
 */
int main(int argc, const char** argv, const char** env)
{
	printf("sizeof(int)     : %2d\n", sizeof(int));
	printf("sizeof(int32_t) : %2d\n", sizeof(int32_t));
	printf("sizeof(uint32_t): %2d\n", sizeof(uint32_t));
	printf("sizeof(int64_t) : %2d\n", sizeof(int64_t));
	printf("sizeof(uint64_t): %2d\n", sizeof(uint64_t));

	
	// Test the uint32_t
	{
		uint32_t n_snk = 30000UL;
		uint32_t n_src = 80000UL;
		uint32_t n_interaction = n_snk * n_src;
		cout << "(n_sink * n_source = " << setw(6) << n_snk << " * " << setw(6) << n_src << " = " << setw(12) << n_interaction << ")---------------------------------------------------------------" << endl;
		printf("(n_sink * n_source = %6lu * %6lu = %12lu\n", n_snk, n_src, n_interaction);
		n_snk = 100000UL;
		n_src = 100000UL;
		n_interaction = n_snk * n_src;
		cout << "(n_sink * n_source = " << setw(6) << n_snk << " * " << setw(6) << n_src << " = " << setw(12) << n_interaction << ")---------------------------------------------------------------" << endl;
		printf("(n_sink * n_source = %6lu * %6lu = %12lu\n", n_snk, n_src, n_interaction);

		n_snk = -1;
		printf("(n_sink = %12lu\n", n_snk, n_src, n_interaction);
	}

	// Test the uint64_t
	{
		uint64_t n_snk = 30000ULL;
		uint64_t n_src = 80000ULL;
		uint64_t n_interaction = n_snk * n_src;
		cout << "(n_sink * n_source = " << setw(6) << n_snk << " * " << setw(6) << n_src << " = " << setw(12) << n_interaction << ")---------------------------------------------------------------" << endl;
		printf("(n_sink * n_source = %6llu * %6llu = %12llu\n", n_snk, n_src, n_interaction);
		n_snk = 100000ULL;
		n_src = 100000ULL;
		n_interaction = n_snk * n_src;
		cout << "(n_sink * n_source = " << setw(6) << n_snk << " * " << setw(6) << n_src << " = " << setw(12) << n_interaction << ")---------------------------------------------------------------" << endl;
		printf("(n_sink * n_source = %6llu * %6llu = %12llu\n", n_snk, n_src, n_interaction);

		n_snk = -1;
		printf("(n_sink = %12llu\n", n_snk, n_src, n_interaction);
	}
}
#endif

#if 0
/*
 * 
 */
int main(int argc, char* argv[], char* env[])
{
	red_test::run(argc, argv);
}
#endif

#if 0

dim3 grid;
dim3 block;
const uint32_t n_tpb = 16;

void set_kernel_launch_param(uint32_t n_data)
{
	uint32_t n_thread = min(n_tpb, n_data);
	uint32_t n_block = (n_data + n_thread - 1)/n_thread;

	grid.x	= n_block;
	block.x = n_thread;
}

void copy_to_device(uint32_t n, sim_data_t* sim_data)
{
	for (uint32_t i = 0; i < 2; i++)
	{
		copy_vector_to_device((void *)sim_data->d_y[i],	    (void *)sim_data->h_y[i], n*sizeof(var4_t));
	}

	copy_vector_to_device((void *)sim_data->d_p,		(void *)sim_data->h_p,		  n*sizeof(pp_disk_t::param_t));
	copy_vector_to_device((void *)sim_data->d_body_md,	(void *)sim_data->h_body_md,  n*sizeof(body_metadata_t));
	copy_vector_to_device((void *)sim_data->d_epoch,	(void *)sim_data->h_epoch,	  n*sizeof(ttt_t));
}

int main(int argc, char** argv)
{
	sim_data_t* sim_data = new sim_data_t;

	try
	{
		device_query(cout, 0, false);

		uint32_t n_bodies[] = {1, 99, 0, 0, 0, 0 ,0};
		uint32_t n_body = 0;
		for (int i = 0; i < 7; i++)
		{
			n_body += n_bodies[i];
		}
		allocate_host_storage(sim_data, n_body);
		allocate_device_storage(sim_data, n_body);
		tools::populate_data(n_bodies, sim_data);
		copy_to_device(n_body, sim_data);

		// Position and velocity of the system's barycenter
		var4_t h_R0 = {0.0, 0.0, 0.0, 0.0};
		var4_t h_V0 = {0.0, 0.0, 0.0, 0.0};
		var_t M = tools::get_total_mass(n_body, sim_data);
		tools::calc_bc(N, false, sim_data, M, &h_R0, &h_V0);
		if (1)
		{
			cout << "   Position and velocity of the barycenter on the host:" << endl;
			cout << "     R0: ";  tools::print_vector(&h_R0);
			cout << "     V0: ";  tools::print_vector(&h_V0);
		}

		// Position and velocity of the system's barycenter
		var4_t *d_R0 = 0x0;
		var4_t *d_V0 = 0x0;
		ALLOCATE_DEVICE_VECTOR((void**)&d_R0, 1*sizeof(var4_t));
		ALLOCATE_DEVICE_VECTOR((void**)&d_V0, 1*sizeof(var4_t));
		
		set_kernel_launch_param(n_body);
		kernel::calc_bc<<<grid, block>>>(n_body, M, sim_data->d_body_md, sim_data->d_p, sim_data->d_y[0], sim_data->d_y[1], d_R0, d_V0);
		CUDA_CHECK_ERROR();

		var4_t h_R   = {0.0, 0.0, 0.0, 0.0};
		var4_t h_V   = {0.0, 0.0, 0.0, 0.0};
		copy_vector_to_host(&h_R, d_R0, 1*sizeof(var4_t));
		copy_vector_to_host(&h_V, d_V0, 1*sizeof(var4_t));
		if (1)
		{
			cout << "   Position and velocity of the barycenter on the device:" << endl;
			cout << "     R0: ";  tools::print_vector(&h_R);
			cout << "     V0: ";  tools::print_vector(&h_V);
		}

		deallocate_host_storage(sim_data);
		deallocate_device_storage(sim_data);

		FREE_DEVICE_VECTOR((void**)&d_R0);
		FREE_DEVICE_VECTOR((void**)&d_V0);

		delete sim_data;
	}
	catch(const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}

	return (EXIT_SUCCESS);
}
#endif


#if 0
/*
 * Test the number_to_string() template function
 */
template <typename T>
std::string _number_to_string( T number, uint32_t width, bool fill)
{
	std::ostringstream ss;

	if (fill)
	{
		if (0 < width)
		{
			ss << setw(width) << setfill('0') << number;
		}
		else
		{
			ss << setfill('0') << number;
		}
	}
	else
	{
		if (0 < width)
		{
			ss << setw(width) << number;
		}
		else
		{
			ss << number;
		}
	}

	return ss.str();
}

template std::string _number_to_string<int>(int, uint32_t width, bool fill);
template std::string _number_to_string<uint32_t>(uint32_t, uint32_t width, bool fill);

int main(int argc, char** argv)
{
	int id = 10;

	cout << "id: " << _number_to_string(id, 0, false) << endl;
	cout << "id: " << _number_to_string(id, 10, false) << endl;
	cout << "id: " << _number_to_string(id, 0, true) << endl;
	cout << "id: " << _number_to_string(id, 7, true) << endl;
	
}

#endif

#if 1
/*
 *  Convert Dvorak input format to red-cuda format
 */

typedef struct dvorak_header
    {
	    ttt_t simulation_length;           //! length of the simulation [day]
	    ttt_t output_interval;             //! interval between two succesive output epoch [day]
        uint32_t data_format;               //! The format of the input data (0: orbital elements)
        uint32_t number_of_lie_terms;
        uint32_t n_massive;                //! Number of massive bodies
        uint32_t n_massless;               //! Number of massless bodies
        var_t accuray;                     //! 10-base logarithm of the accuracy
        ttt_t dt_min;                      //! Minimal stepsize [day]
        ttt_t sum_of_steps;                //! Sum of steps [day]
        uint32_t n_step;                   //! Number of steps
    } dvorak_header_t;



void read_dvorak_header(ifstream& input, dvorak_header_t& header)
{
    input >> header.simulation_length;
    input >> header.output_interval;
    input >> header.data_format;
    input >> header.number_of_lie_terms;
    input >> header.n_massive;
    input >> header.n_massless;
    input >> header.accuray;
    input >> header.dt_min;
    input >> header.sum_of_steps;
    input >> header.n_step;
}

int main()
{
	cout << "-----------------------------------------------" << endl;
	cout << "Did you replace the D in the input with E or e?" << endl;
	cout << "-----------------------------------------------" << endl;

	//string input_dir = "C:\\Work\\Dvorak\\Temp";
	string input_dir = "C:\\Work\\red.cuda.Results\\Dvorak\\3D\\Super\\Run_01";
    string filename = "super.in";

    pp_disk_t::sim_data_t sd;
    vector<string> names;

    string path = file::combine_path(input_dir, filename);
    try
    {
        uint32_t n_total;
        vector<orbelem_t> oe;
        vector<nbody_t::param_t> p;

        ifstream input(path);
	    if (input) 
	    {
            dvorak_header_t header;
            read_dvorak_header(input, header);

            n_total = header.n_massive + header.n_massless;

            for (uint32_t i = 0; i < n_total; i++)
            {
                orbelem_t _oe;
                nbody_t::param_t _p;
                input >> _oe.sma >> _oe.ecc >> _oe.inc >> _oe.peri >> _oe.node >> _oe.mean >> _p.mass >> _p.cd;

                _oe.inc  *= constants::DegreeToRadian;
                _oe.peri *= constants::DegreeToRadian;
                _oe.node *= constants::DegreeToRadian;
                _oe.mean *= constants::DegreeToRadian;

                oe.push_back(_oe);
                p.push_back(_p);
            }
	    }
	    else 
	    {
		    throw string("Cannot open " + path + ".");
	    }

        // Populate the sd container
        {
            sd.h_y.resize(2);
            sd.h_p       = new pp_disk_t::param_t[n_total];
            sd.h_y[0]    = new var4_t[n_total];
            sd.h_y[1]    = new var4_t[n_total];
            sd.h_body_md = new pp_disk_t::body_metadata_t[n_total];

            names.resize(n_total);

            int32_t id = 1;
            uint32_t idx_pp = 1;
            for (uint32_t i = 0; i < n_total; i++, id++)
            {
                // This is the star
                if (0 == i)
                {
                    names[i] = "Sun",

                    sd.h_body_md[i].id = id;
                    sd.h_body_md[i].body_type = BODY_TYPE_STAR;
                    sd.h_body_md[i].mig_type = MIGRATION_TYPE_NO;
                    sd.h_body_md[i].mig_stop_at = 0.0;
                    
                    sd.h_p[i].cd      = p[i].cd;
                    sd.h_p[i].density = p[i].density;
                    sd.h_p[i].mass    = p[i].mass;
                    sd.h_p[i].radius  = p[i].radius;
                    // Compute other physical properties
                    {
                        var_t density = 1.408 /* g/cm^3 */ * constants::GramPerCm3ToSolarPerAu3;
                        var_t R = tools::calc_radius(p[i].mass, density);

                        sd.h_p[i].radius = R;
                        sd.h_p[i].density = density;
                        sd.h_p[i].cd = 0.0;
                    }
                    
                    sd.h_y[0][i].x = sd.h_y[0][i].y = sd.h_y[0][i].z = 0.0;
                    sd.h_y[1][i].x = sd.h_y[1][i].y = sd.h_y[1][i].z = 0.0;
                }
                // This is the Jupiter
                else if (1 == i)
                {
                    names[i] = "Jupiter",

                    sd.h_body_md[i].id = id;
                    sd.h_body_md[i].body_type = BODY_TYPE_GIANTPLANET;
                    sd.h_body_md[i].mig_type = MIGRATION_TYPE_NO;
                    sd.h_body_md[i].mig_stop_at = 0.0;
                    
                    sd.h_p[i].cd      = p[i].cd;
                    sd.h_p[i].density = p[i].density;
                    sd.h_p[i].mass    = p[i].mass;
                    sd.h_p[i].radius  = p[i].radius;
                    // Compute other physical properties
                    {
                        var_t density = 1.326 /* g/cm^3 */ * constants::GramPerCm3ToSolarPerAu3;
                        var_t R = tools::calc_radius(p[i].mass, density);

                        sd.h_p[i].radius = R;
                        sd.h_p[i].density = density;
                        sd.h_p[i].cd = 0.0;
                    }
                    
                    var_t mu = K2*(p[0].mass + p[i].mass);
                    tools::calc_phase(mu, &(oe[i]), &(sd.h_y[0][i]), &(sd.h_y[1][i]));
                }
                // These are the other bodies
                else
                {
                    names[i] = "Proto" + number_to_string(idx_pp++);

                    sd.h_body_md[i].id = id;
                    sd.h_body_md[i].body_type = BODY_TYPE_PROTOPLANET;
                    sd.h_body_md[i].mig_type = MIGRATION_TYPE_NO;
                    sd.h_body_md[i].mig_stop_at = 0.0;
                    
                    sd.h_p[i].cd      = p[i].cd;       // This is the mass fraction
                    sd.h_p[i].density = p[i].density;
                    sd.h_p[i].mass    = p[i].mass;
                    sd.h_p[i].radius  = p[i].radius;
                    // Compute other physical properties
                    {
                        var_t density = 2.7 /* g/cm^3 */ * constants::GramPerCm3ToSolarPerAu3;
                        // Mass of solids
                        var_t m_s = p[i].mass * (1.0 - p[i].cd);
                        var_t R_s = tools::calc_radius(m_s, density);
                        var_t V_s = 4.0/3.0 * PI * CUBE(R_s);
                        // Mass of water
                        density = 1.0 /* g/cm^3 */ * constants::GramPerCm3ToSolarPerAu3;
                        var_t m_w = p[i].mass * (p[i].cd);
                        var_t R_w = tools::calc_radius(m_w, density);
                        var_t V_w = 4.0/3.0 * PI * CUBE(R_w);

                        var_t V_total = V_s + V_w;
                        var_t R_avg = pow((3.0 * V_total)/ (4.0 * PI), 0.333333333333333);

                        density = tools::calc_density(p[i].mass, R_avg);

                        sd.h_p[i].radius = R_avg;
                        sd.h_p[i].density = density;
                        sd.h_p[i].cd = 0.0;
                    }

                    var_t mu = K2*(p[0].mass + p[i].mass);
                    tools::calc_phase(mu, &(oe[i]), &(sd.h_y[0][i]), &(sd.h_y[1][i]));
                }
            }
        }

        ttt_t dt = tools::calc_orbital_period(K2, 1.0);

	    tools::transform_to_bc(n_total, &sd);
	    //tools::transform_time(n_total, &sd);
	    tools::transform_velocity(n_total, &sd);
	    ttt_t t0 = 0.0;
	    dt *= constants::Gauss;

		n_objects_t *n_bodies = new n_objects_t(1, 1, 0, 300, 0, 0, 0);
        filename = "input.info.txt";
        path = file::combine_path(input_dir, filename);
        ofstream output(path);
        if (output)
        {
    		file::print_data_info_record_ascii_RED(output, t0, dt, 0, n_bodies);
        }
        else
        {
		    throw string("Cannot open " + path + ".");
        }
        output.close();

        filename = "input.data.txt";
        path = file::combine_path(input_dir, filename);
        output.open(path, ofstream::out);
        if (output)
        {
            for (uint32_t i = 0; i < n_total; i++)
            {
                file::print_body_record_ascii_RED(output, names[i], &sd.h_p[i], &sd.h_body_md[i], &sd.h_y[0][i], &sd.h_y[1][i]);
            }
        }
        else
        {
		    throw string("Cannot open " + path + ".");
        }
        output.close();

    }
   	catch (const string& msg)
	{
        delete sd.h_body_md;
        delete sd.h_y[1];
        delete sd.h_y[0];
        delete sd.h_p;
		cerr << "Error: " << msg << endl;
	}

    return (EXIT_SUCCESS);
}

#endif