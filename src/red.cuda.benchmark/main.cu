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

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#include <ctime>
#endif

using namespace std;
using namespace redutilcu;

/* Remove if already defined */
typedef long long int64; 
typedef unsigned long long uint64;

typedef struct _cpu_info
		{
			string processor;
			string vendor_id;
			string cpu_family;
			string model;
			string model_name;
			string stepping;
			string microcode;
			string cpu_MHz;
			string cache_size;
			string physical_id;
			string siblings;
			string core_id;
			string cpu_cores;
			string apicid;
			string initial_apicid;
			string fpu;
			string fpu_exception;
			string cpuid_level;
			string wp;
			string flags;
			string bogomips;
			string clflush_size;
			string cache_alignment;
			string address_sizes;
			string power_management;
		} cpu_info_t;

typedef enum benchmark_output_name
		{
			BENCHMARK_OUTPUT_NAME_LOG,
			BENCHMARK_OUTPUT_NAME_RESULT,
			BENCHMARK_OUTPUT_NAME_SUMMARY,
			BENCHMARK_OUTPUT_NAME_N
		} benchmark_output_name_t;

typedef enum job_name
		{
			JOB_NAME_UNDEFINED,
			JOB_NAME_COMPARE,
			JOB_NAME_BENCMARK_CPU,
			JOB_NAME_BENCMARK_GPU,
			JOB_NAME_N
		} job_name_t;

typedef struct option
		{
			string             o_dir;
			string             base_fn;
			computing_device_t comp_dev;
			bool               compare;
			var_t              tol;
			int                id_dev;
			int                n0;
			int                n1;
			int                dn;
			int                n_iter;
			job_name_t         job_name;
		} option_t;

static string method_name[] = { "naive", "naive_sym", "tile", "tile_advanced" };
static string param_name[] =  { "n_body", "interaction_bound"                 };


//https://aufather.wordpress.com/2010/09/08/high-performance-time-measuremen-in-linux/

/* 
 *  -- Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both --
 * Returns the amount of microseconds elapsed since the UNIX epoch. Works on both
 * windows and linux.
 */
uint64 GetTimeMs64()
{
#ifdef _WIN32
	/* Windows */
	FILETIME ft;
	LARGE_INTEGER li;

	/* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
	* to a LARGE_INTEGER structure. */
	GetSystemTimeAsFileTime(&ft);
	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;

	uint64 ret = li.QuadPart;
	ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
	//ret /= 10000; /* From 100 nano seconds (10^-7) to 1 millisecond (10^-3) intervals */
	ret /= 10; /* From 100 nano seconds (10^-7) to 1 microsecond (10^-6) intervals */

	return ret;
#else
	/* Linux */
	struct timeval tv;

	gettimeofday(&tv, NULL);

	uint64 ret = tv.tv_usec;
	/* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
	//ret /= 1000;

	/* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
	//ret += (tv.tv_sec * 1000);
	/* Adds the seconds (10^0) after converting them to microseconds (10^-6) */
	ret += (tv.tv_sec * 1000000);

	return ret;
#endif
}

void print(computing_device_t comp_dev, string& method_name, string& param_name, interaction_bound int_bound, int n_body, int n_tpb, ttt_t Dt_CPU, ttt_t Dt_GPU, ofstream& sout, bool prn_to_scr)
{
	static const char* computing_device_name[] = 
	{
				"CPU",
				"GPU"
	};
	static char sep = ',';

	if (prn_to_scr)
	{
		cout << tools::get_time_stamp(false) << sep
     		 << setw(4) << computing_device_name[comp_dev] << sep
			 << setw(20) << method_name << sep
			 << setw(20) << param_name << sep
			 << setw(6) << int_bound.sink.y - int_bound.sink.x << sep
			 << setw(6) << int_bound.source.y - int_bound.source.x << sep
			 << setw(6) << n_body << sep
			 << setw(5) << n_tpb << sep
			 << scientific << setprecision(4) << setw(12) << Dt_CPU << sep
			 << scientific << setprecision(4) << setw(12) << Dt_GPU << endl;
	}

	sout << tools::get_time_stamp(false) << SEP
     	 << setw(4) << computing_device_name[comp_dev] << SEP
		 << setw(20) << method_name << SEP
		 << setw(20) << param_name << SEP
		 << setw(6) << int_bound.sink.y - int_bound.sink.x << SEP
		 << setw(6) << int_bound.source.y - int_bound.source.x << SEP
		 << setw(6) << n_body << SEP
		 << setw(5) << n_tpb << SEP
		 << scientific << setprecision(4) << setw(12) << Dt_CPU << SEP
		 << scientific << setprecision(4) << setw(12) << Dt_GPU << endl;
}

namespace kernel
{
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
 * riVec.xyz  = {x, y, z,  }
 * rjVec.xyzw = {x, y, z, m}
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
	void calc_gravity_accel_tile_verbose(int n_body, int tile_size, const var4_t* global_x, const var_t* mass, var4_t* global_a)
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
	if (n_body > gtid)
	{
		my_pos = global_x[gtid];
	}
	printf("gtid = %3d my_pos = [%10.6le, %10.6le, %10.6le]\n", gtid, my_pos.x, my_pos.y, my_pos.z);
	if (0 == blockIdx.x)
	{
		printf("[0 == blockIdx.x]: gtid = %3d my_pos = [%10.6le, %10.6le, %10.6le]\n", gtid, my_pos.x, my_pos.y, my_pos.z);
	}

	for (int tile = 0; (tile * tile_size) < n_body; tile++)
	{
		int idx = tile * blockDim.x + threadIdx.x;
		// To avoid overruning the global_x buffer
		if (n_body > idx)
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
				if (n_body <= (tile * tile_size) + j)
				{
					printf("Warning: n_body (%3d) <= tile * tile_size + j (%3d)\n", n_body, (tile * tile_size) + j);
				}
				// To avoid self-interaction or mathematically division by zero
				if (gtid == (tile * tile_size)+j)
				{
					printf("Warning: gtid (%3d) == (tile * tile_size)+j (%3d)\n", gtid, (tile * tile_size)+j);
				}
				printf("[0 == blockIdx.x]: tile = %3d j = %3d threadIdx.x = %3d idx = %3d my_pos = [%10.6le, %10.6le, %10.6le] sh_pos[j] = [%10.6le, %10.6le, %10.6le]\n", tile, j, threadIdx.x, idx, my_pos.x, my_pos.y, my_pos.z, sh_pos[j].x, sh_pos[j].y, sh_pos[j].z);
			}
			// To avoid overrun the mass buffer
			if (n_body <= (tile * tile_size) + j)
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
	if (n_body > gtid)
	{
		printf("gtid = %3d acc = [%14.6le, %14.6le, %14.6le]\n", gtid, acc.x, acc.y, acc.z);
		global_a[gtid] = acc;
	}
}

__global__
	void calc_gravity_accel_tile(int n_body, int tile_size, const var4_t* global_x, const var_t* mass, var4_t* global_a)
{
	extern __shared__ var4_t sh_pos[];

	var4_t my_pos = {0.0, 0.0, 0.0, 0.0};
	var4_t acc    = {0.0, 0.0, 0.0, 0.0};

	const int gtid = blockIdx.x * blockDim.x + threadIdx.x;

	// To avoid overruning the global_x buffer
	if (n_body > gtid)
	{
		my_pos = global_x[gtid];
	}
	for (int tile = 0; (tile * tile_size) < n_body; tile++)
	{
		int idx = tile * blockDim.x + threadIdx.x;
		// To avoid overruning the global_x buffer
		if (n_body > idx)
		{
			sh_pos[threadIdx.x] = global_x[idx];
		}
		__syncthreads();
		for (int j = 0; j < blockDim.x; j++)
		{
			// To avoid overrun the mass buffer
			if (n_body <= (tile * tile_size) + j)
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
	}
	// To avoid overruning the global_a buffer
	if (n_body > gtid)
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
	void calc_gravity_accel_tile_verbose(int n_body, const var4_t* global_x, const var_t* mass, var4_t* global_a)
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
	if (n_body > gtid)
	{
		my_pos = global_x[gtid];
		if (GTID == gtid)
		{
			printf("[%2d == i] rVec = [%16.8le, %16.8le, %16.8le]\n", GTID, my_pos.x, my_pos.y, my_pos.z);
			printf("[%2d == i], blockIdx.x = %2d\n", GTID, blockIdx.x);
		}
	}
	
	for (int tile = 0; (tile * blockDim.x) < n_body; tile++)
	{
		const int idx = tile * blockDim.x + threadIdx.x;
		if (BLKIDX == blockIdx.x)
		{
			printf("[%2d == blockIdx.x]: tile = %3d threadIdx.x = %3d idx = %3d\n", BLKIDX, tile, threadIdx.x, idx);
		}
		// To avoid overruning the global_x and mass buffer
		if (n_body > idx)
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
			if (n_body <= (tile * blockDim.x) + j)
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
	if (n_body > gtid)
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
	void calc_gravity_accel_tile(int n_body, const var4_t* global_x, const var_t* mass, var4_t* global_a)
{
	extern __shared__ var4_t sh_pos[];

	const int gtid = blockIdx.x * blockDim.x + threadIdx.x;

	var4_t acc = {0.0, 0.0, 0.0, 0.0};
	var4_t my_pos;

	// To avoid overruning the global_x buffer
	if (n_body > gtid)
	{
		my_pos = global_x[gtid];
	}
	
	for (int tile = 0; (tile * blockDim.x) < n_body; tile++)
	{
		const int idx = tile * blockDim.x + threadIdx.x;
		// To avoid overruning the global_x and mass buffer
		if (n_body > idx)
		{
			sh_pos[threadIdx.x]   = global_x[idx];
			sh_pos[threadIdx.x].w = mass[idx];
		}
		__syncthreads();

		for (int j = 0; j < blockDim.x; j++)
		{
			// To avoid overrun the input arrays
			if (n_body <= (tile * blockDim.x) + j)
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
	if (n_body > gtid)
	{
		global_a[gtid] = acc;
	}
}

__global__
	void calc_gravity_accel_naive(int n_body, const var4_t* global_x, const var_t* mass, var4_t* global_a)
{
	// i is the index of the SINK body
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (n_body > i)
	{
		global_a[i].x = global_a[i].y = global_a[i].z = global_a[i].w = 0.0;
		var4_t dVec = {0.0, 0.0, 0.0, 0.0};
		// j is the index of the SOURCE body
		for (int j = 0; j < n_body; j++) 
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
	void calc_gravity_accel_naive_sym(int n_body, const var4_t* global_x, const var_t* mass, var4_t* global_a)
{
	// i is the index of the SINK body
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (n_body > i)
	{
		var4_t dVec = {0.0, 0.0, 0.0, 0.0};
		// j is the index of the SOURCE body
		for (int j = i+1; j < n_body; j++) 
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
} /* namespace kernel */

namespace kernel2
{
static __global__
	void calc_grav_accel_int_mul_of_thread_per_block(interaction_bound int_bound, const pp_disk_t::param_t* p, const var4_t* r, var4_t* a)
{
	// i is the index of the SINK body
	const int i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	var4_t dVec;
	// This line (beyond my depth) speeds up the kernel
	a[i].x = a[i].y = a[i].z = a[i].w = 0.0;
	// j is the index of the SOURCE body
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
	void calc_grav_accel(ttt_t t, interaction_bound int_bound, const pp_disk_t::body_metadata_t* bmd, const pp_disk_t::param_t* p, const var4_t* r, const var4_t* v, var4_t* a, pp_disk_t::event_data_t* events, int *event_counter)
{
	// i is the index of the SINK body
	const int i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	if (i < int_bound.sink.y)
	{
		// This line (beyond my depth) speeds up the kernel
		a[i].x = a[i].y = a[i].z = a[i].w = 0.0;
		if (bmd[i].id > 0)
		{
			var4_t dVec = {0.0, 0.0, 0.0, 0.0};
			// j is the index of the SOURCE body
			for (int j = int_bound.source.x; j < int_bound.source.y; j++) 
			{
				/* Skip the body with the same index and those which are inactive ie. id < 0 */
				if (i == j || bmd[j].id < 0)
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
					//printf("t = %20.10le d = %20.10le %d. COLLISION detected: id: %5d id: %5d\n", t, d, k+1, bmd[survivIdx].id, bmd[mergerIdx].id);

					events[k].event_name = EVENT_NAME_COLLISION;
					events[k].d = d;
					events[k].t = t;
					events[k].id1 = bmd[survivIdx].id;
					events[k].id2 = bmd[mergerIdx].id;
					events[k].idx1 = survivIdx;
					events[k].idx2 = mergerIdx;
					events[k].r1 = r[survivIdx];
					events[k].v1 = v[survivIdx];
					events[k].r2 = r[mergerIdx];
					events[k].v2 = v[mergerIdx];
				}
			}
			// 36 FLOP
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
 * riVec.xyz  = {x, y, z,  }
 * rjVec.xyzw = {x, y, z, m}
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
	void calc_gravity_accel_tile(interaction_bound int_bound, int tile_size, const var4_t* r, const var_t* m, var4_t* a)
{
	extern __shared__ var4_t sh_pos[];

	var4_t my_pos = {0.0, 0.0, 0.0, 0.0};
	var4_t acc    = {0.0, 0.0, 0.0, 0.0};

	// i is the index of the SINK body
	const int i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	// To avoid overruning the r buffer
	if (int_bound.sink.y > i)
	{
		my_pos = r[i];
	}
	for (int tile = 0; (tile * tile_size) < int_bound.source.y; tile++)
	{
		// src_idx is the index of the SOURCE body in the tile
		int src_idx = int_bound.source.x + tile * tile_size + threadIdx.x;
		// To avoid overruning the r buffer
		if (int_bound.source.y > src_idx)
		{
			sh_pos[threadIdx.x] = r[src_idx];
		}
		__syncthreads();
		// j is the index of the SOURCE body in the current tile
		for (int j = 0; j < blockDim.x; j++)
		{
			// To avoid overrun the mass buffer
			if (int_bound.source.y <= int_bound.source.x + (tile * tile_size) + j)
			{
				break;
			}
			// To avoid self-interaction or mathematically division by zero
			if (i != int_bound.source.x + (tile * tile_size)+j)
			{
				acc = body_body_interaction(my_pos, sh_pos[j], m[src_idx], acc);
			}
		}
		__syncthreads();
	}

	// To avoid overruning the a buffer
	if (int_bound.sink.y > i)
	{
		a[i] = acc;
	}
}

/*
 * The same as above but the blockDim.x is used instead of tile_size (both variable have the same value)
 */
__global__
	void calc_gravity_accel_tile(interaction_bound int_bound, const var4_t* r, const var_t* m, var4_t* a)
{
	extern __shared__ var4_t sh_pos[];

	// i is the index of the SINK body
	const int i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	var4_t acc = {0.0, 0.0, 0.0, 0.0};
	var4_t my_pos;

	// To avoid overruning the r buffer
	if (int_bound.sink.y > i)
	{
		my_pos = r[i];
	}
	
	for (int tile = 0; (tile * blockDim.x) < int_bound.source.y; tile++)
	{
		// src_idx is the index of the SOURCE body in the tile
		const int src_idx = int_bound.source.x + tile * blockDim.x + threadIdx.x;
		// To avoid overruning the r and mass buffer
		if (int_bound.source.y > src_idx)
		{
			sh_pos[threadIdx.x]   = r[src_idx];
			sh_pos[threadIdx.x].w = m[src_idx];
		}
		__syncthreads();

		// j is the index of the SOURCE body in the current tile
		for (int j = 0; j < blockDim.x; j++)
		{
			// To avoid overrun the mass buffer
			if (int_bound.source.y <= int_bound.source.x + (tile * blockDim.x) + j)
			{
				break;
			}
			// To avoid self-interaction or mathematically division by zero
			if (i != int_bound.source.x + (tile * blockDim.x)+j)
			{
				acc = body_body_interaction(my_pos, sh_pos[j], acc);
			}
		}
		__syncthreads();
	}

	// To avoid overruning the a buffer
	if (int_bound.sink.y > i)
	{
		a[i] = acc;
	}
}

__global__
	void calc_gravity_accel_naive(interaction_bound int_bound, const var4_t* r, const var_t* m, var4_t* a)
{
	// i is the index of the SINK body
	const int i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	if (int_bound.sink.y > i)
	{
		a[i].x = a[i].y = a[i].z = a[i].w = 0.0;
		var4_t dVec = {0.0, 0.0, 0.0, 0.0};
		// j is the index of the SOURCE body
		for (int j = int_bound.source.x; j < int_bound.source.y; j++) 
		{
			/* Skip the body with the same index */
			if (i == j)
			{
				continue;
			}
			//global_a[i] = body_body_interaction(global_x[i], global_x[j], mass[j], global_a[i]);
			// 3 FLOP
			dVec.x = r[j].x - r[i].x;
			dVec.y = r[j].y - r[i].y;
			dVec.z = r[j].z - r[i].z;
			// 5 FLOP
			dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2

			// 20 FLOP
			//var_t d = sqrt(dVec.w);								// = r
			// 2 FLOP
			//dVec.w = m[j] / (d*dVec.w);

			dVec.w *= sqrt(dVec.w); // = r3
			dVec.w = m[j] / dVec.w; // = mj / r3

			// 6 FLOP
			a[i].x += dVec.w * dVec.x;
			a[i].y += dVec.w * dVec.y;
			a[i].z += dVec.w * dVec.z;
		} // 36 FLOP
	}
}

// NOTE: Before calling this function, the a array must be cleared!
// TODO: The result of this kernel differs from the real one IF the number of bodies larger than the warp size
// e.g. n_body = 50. What is the reason of this. (For n_body = 16, 32, 40, the result is correct).
__global__
	void calc_gravity_accel_naive_sym(interaction_bound int_bound, const var4_t* r, const var_t* m, var4_t* a)
{
	// i is the index of the SINK body
	const int i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	if (int_bound.sink.y > i)
	{
		var4_t dVec = {0.0, 0.0, 0.0, 0.0};
		// j is the index of the SOURCE body
		for (int j = i+1; j < int_bound.source.y; j++) 
		{
			// 3 FLOP
			dVec.x = r[j].x - r[i].x;
			dVec.y = r[j].y - r[i].y;
			dVec.z = r[j].z - r[i].z;
			// 5 FLOP
			dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2

			// sqrt operation takes approximately 20 FLOP
			var_t d = sqrt(dVec.w);								// = r
			// 2 FLOP
			var_t r_3 = 1.0 / (d*dVec.w);
			// 1 FLOP
			dVec.w = m[j] * r_3;
			// 6 FLOP
			a[i].x += dVec.w * dVec.x;
			a[i].y += dVec.w * dVec.y;
			a[i].z += dVec.w * dVec.z;

			// 2 FLOP
			dVec.w = m[i] * r_3;
			// 6 FLOP
			a[j].x -= dVec.w * dVec.x;
			a[j].y -= dVec.w * dVec.y;
			a[j].z -= dVec.w * dVec.z;
		} // 36 + 8 = 44 FLOP
	}
}
} /* namespace kernel_2 */


void cpu_calc_grav_accel_naive(int n_body, const var4_t* r, const var_t* mass, var4_t* a)
{
	for (int i = 0; i < n_body; i++)
	{
		var4_t dVec = {0.0, 0.0, 0.0, 0.0};
		for (int j = 0; j < n_body; j++) 
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
		} // 36 FLOP
	}
}

void cpu_calc_grav_accel_naive(interaction_bound int_bound, const var4_t* r, const var_t* mass, var4_t* a)
{
	for (unsigned int i = int_bound.sink.x; i < int_bound.sink.y; i++)
	{
		var4_t dVec = {0.0, 0.0, 0.0, 0.0};
		for (unsigned int j = int_bound.source.x; j < int_bound.source.y; j++) 
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
		} // 36 FLOP
	}
}

void cpu_calc_grav_accel_naive_sym(int n_body, const var4_t* r, const var_t* mass, var4_t* a)
{
	for (int i = 0; i < n_body; i++)
	{
		var4_t dVec = {0.0, 0.0, 0.0, 0.0};
		for (int j = i+1; j < n_body; j++) 
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

void cpu_calc_grav_accel_naive_sym(interaction_bound int_bound, const var4_t* r, const var_t* mass, var4_t* a)
{
	for (unsigned int i = int_bound.sink.x; i < int_bound.sink.y; i++)
	{
		var4_t dVec = {0.0, 0.0, 0.0, 0.0};
		for (unsigned int j = i + 1; j < int_bound.source.y; j++) 
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

void cpu_calc_grav_accel_naive_sym_advanced(interaction_bound int_bound, const var4_t* r, const var_t* mass, var4_t* a)
{
	const int n_sink   = int_bound.sink.y   - int_bound.sink.x;
	const int n_source = int_bound.source.y - int_bound.source.x;
	const int n_body = min(n_sink, n_source);

	for (unsigned int i = int_bound.sink.x; i < int_bound.sink.x + n_body; i++)
	{
		var4_t dVec = {0.0, 0.0, 0.0, 0.0};
		for (unsigned int j = i + 1; j < int_bound.sink.x + n_body; j++) 
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

	if (n_sink < n_source)
	{
		for (unsigned int i = int_bound.sink.x; i < int_bound.sink.y; i++)
		{
			var4_t dVec = {0.0, 0.0, 0.0, 0.0};
			for (unsigned int j = int_bound.source.x + n_body; j < int_bound.source.y; j++) 
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
			} // 36 FLOP
		}
	}
	else
	{
		for (unsigned int i = int_bound.sink.x + n_body; i < int_bound.sink.y; i++)
		{
			var4_t dVec = {0.0, 0.0, 0.0, 0.0};
			for (unsigned int j = int_bound.source.x; j < int_bound.source.y; j++) 
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
			} // 36 FLOP
		}
	}

}

//! Computes the time it takes for the GPU the calculate the gravitatinal acceleration using the naive method
/*!
	\param n_body specifies the numbetr of bodies (both the number of sink and source bodies)
	\param n_tpb specifies the number of threads per block to be used during the kernel launch
	\param d_x array that holds the position of the bodies
	\param d_m array that holds the mass of the bodies
	\param d_a array that will hold the computed acceleration of the bodies
	\return The elapsed time in ms took for the device to perform the computation
*/
float gpu_calc_grav_accel_naive_benchmark(int n_body, int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a)
{
	cudaEvent_t start, stop;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	dim3 grid((n_body + n_tpb - 1)/n_tpb);
	dim3 block(n_tpb);

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	kernel::calc_gravity_accel_naive<<<grid, block>>>(n_body, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float elapsed_time = 0.0f;
	// Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds).
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));

	return elapsed_time;
}

float gpu_calc_grav_accel_naive_benchmark(interaction_bound int_bound, int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a)
{
	cudaEvent_t start, stop;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	int n_body = int_bound.sink.y - int_bound.sink.x;

	dim3 grid((n_body + n_tpb - 1)/n_tpb);
	dim3 block(n_tpb);

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	kernel2::calc_gravity_accel_naive<<<grid, block>>>(int_bound, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float elapsed_time = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));

	return elapsed_time;
}

float gpu_calc_grav_accel_naive_sym_benchmark(int n_body, int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a)
{
	cudaEvent_t start, stop;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	dim3 grid((n_body + n_tpb - 1)/n_tpb);
	dim3 block(n_tpb);

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	CUDA_SAFE_CALL(cudaMemset(d_a, 0, n_body*sizeof(var4_t)));
	kernel::calc_gravity_accel_naive_sym<<<grid, block>>>(n_body, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float elapsed_time = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));

	return elapsed_time;
}

float gpu_calc_grav_accel_naive_sym_benchmark(interaction_bound int_bound, int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a)
{
	cudaEvent_t start, stop;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	int n_body = int_bound.sink.y - int_bound.sink.x;

	dim3 grid((n_body + n_tpb - 1)/n_tpb);
	dim3 block(n_tpb);

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	CUDA_SAFE_CALL(cudaMemset(d_a, 0, n_body*sizeof(var4_t)));
	kernel2::calc_gravity_accel_naive_sym<<<grid, block>>>(int_bound, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float elapsed_time = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));

	return elapsed_time;
}

// The blockDim.x is used instead of tile_size (reduce the number by 1 of kernel parameters)
float gpu_calc_grav_accel_tile_benchmark(int n_body, int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a)
{
	cudaEvent_t start, stop;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	dim3 grid((n_body + n_tpb - 1)/n_tpb);
	dim3 block(n_tpb);

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	kernel::calc_gravity_accel_tile<<<grid, block, n_tpb * sizeof(var4_t)>>>(n_body, n_tpb, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float elapsed_time = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));

	return elapsed_time;
}

// The blockDim.x is used instead of tile_size (reduce the number by 1 of kernel parameters)
float gpu_calc_grav_accel_tile_benchmark(interaction_bound int_bound, int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a)
{
	cudaEvent_t start, stop;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	int n_body = int_bound.sink.y - int_bound.sink.x;

	dim3 grid((n_body + n_tpb - 1)/n_tpb);
	dim3 block(n_tpb);

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	kernel2::calc_gravity_accel_tile<<<grid, block, n_tpb * sizeof(var4_t)>>>(int_bound, n_tpb, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float elapsed_time = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));

	return elapsed_time;
}

// The same as above
float gpu_calc_grav_accel_tile_benchmark_2(int n_body, int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a)
{
	cudaEvent_t start, stop;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	dim3 grid((n_body + n_tpb - 1)/n_tpb);
	dim3 block(n_tpb);

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	kernel::calc_gravity_accel_tile<<<grid, block, n_tpb * sizeof(var4_t)>>>(n_body, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float elapsed_time = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));

	return elapsed_time;
}

// The same as above
float gpu_calc_grav_accel_tile_benchmark_2(interaction_bound int_bound, int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a)
{
	cudaEvent_t start, stop;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	int n_body = int_bound.sink.y - int_bound.sink.x;

	dim3 grid((n_body + n_tpb - 1)/n_tpb);
	dim3 block(n_tpb);

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	kernel2::calc_gravity_accel_tile<<<grid, block, n_tpb * sizeof(var4_t)>>>(int_bound, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float elapsed_time = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));

	return elapsed_time;
}





void benchmark_CPU(int n_body, const var4_t* h_x, const var_t* h_m, var4_t* h_a, ofstream& o_result, ofstream& o_summary)
{
	ttt_t Dt_CPU = 0.0;
	ttt_t Dt_GPU = 0.0;
	int i = 0;

	//Naive method
	{
		uint64 t0 = GetTimeMs64();
		if (50 >= n_body)
		{
			for (i = 0; i < 1000; i++)
			{
				cpu_calc_grav_accel_naive(n_body, h_x, h_m, h_a);
			}
		}
		else if (50 < n_body && 200 >= n_body)
		{
			for (i = 0; i < 100; i++)
			{
				cpu_calc_grav_accel_naive(n_body, h_x, h_m, h_a);
			}
		}
		else if (200 < n_body && 2000 >= n_body)
		{
			for (i = 0; i < 10; i++)
			{
				cpu_calc_grav_accel_naive(n_body, h_x, h_m, h_a);
			}
		}
		else
		{
			cpu_calc_grav_accel_naive(n_body, h_x, h_m, h_a);
		}
		uint64 t1 = GetTimeMs64();
		Dt_CPU = ((ttt_t)(t1 - t0))/(ttt_t)(i == 0 ? 1 : i)/1000.0f;

		interaction_bound int_bound;
		print(COMPUTING_DEVICE_CPU, method_name[0], param_name[0], int_bound, n_body, 1, Dt_CPU, Dt_GPU, o_result, false);
		print(COMPUTING_DEVICE_CPU, method_name[0], param_name[0], int_bound, n_body, 1, Dt_CPU, Dt_GPU, o_summary, true);
	}

	//Naive symmetric method
	{
		uint64 t0 = GetTimeMs64();
		if (50 >= n_body)
		{
			for (i = 0; i < 1000; i++)
			{
				cpu_calc_grav_accel_naive_sym(n_body, h_x, h_m, h_a);
			}
		}
		else if (50 < n_body && 200 >= n_body)
		{
			for (i = 0; i < 100; i++)
			{
				cpu_calc_grav_accel_naive_sym(n_body, h_x, h_m, h_a);
			}
		}
		else if (200 < n_body && 2000 >= n_body)
		{
			for (i = 0; i < 10; i++)
			{
				cpu_calc_grav_accel_naive_sym(n_body, h_x, h_m, h_a);
			}
		}
		else
		{
			cpu_calc_grav_accel_naive_sym(n_body, h_x, h_m, h_a);
		}
		Dt_CPU = ((ttt_t)(GetTimeMs64() - t0))/(ttt_t)(i == 0 ? 1 : i)/1000.0f;

		interaction_bound int_bound;
		print(COMPUTING_DEVICE_CPU, method_name[1], param_name[0], int_bound, n_body, 1, Dt_CPU, Dt_GPU, o_result, false);
		print(COMPUTING_DEVICE_CPU, method_name[1], param_name[0], int_bound, n_body, 1, Dt_CPU, Dt_GPU, o_summary, true);
	}
}

void benchmark_CPU(interaction_bound int_bound, const var4_t* h_x, const var_t* h_m, var4_t* h_a, ofstream& o_result, ofstream& o_summary)
{
	const uint64_t n_sink   = int_bound.sink.y   - int_bound.sink.x;
	const uint64_t n_source = int_bound.source.y - int_bound.source.x;
	const uint64_t n_interaction = n_sink * n_source;

	ttt_t Dt_CPU = 0.0;
	ttt_t Dt_GPU = 0.0;

	int i = 1;
	//Naive method
	{
		uint64 t0 = GetTimeMs64();
		if (SQR(100ULL) >= n_interaction)
		{
			for (i = 1; i <= 5000; i++)
			{
				cpu_calc_grav_accel_naive(int_bound, h_x, h_m, h_a);
			}
		}
		else if ((SQR(100ULL) < n_interaction && SQR(200ULL) >= n_interaction))
		{
			for (i = 1; i <= 1000; i++)
			{
				cpu_calc_grav_accel_naive(int_bound, h_x, h_m, h_a);
			}
		}
		else if ((SQR(200ULL) < n_interaction && SQR(600ULL) >= n_interaction))
		{
			for (i = 1; i <= 100; i++)
			{
				cpu_calc_grav_accel_naive(int_bound, h_x, h_m, h_a);
			}
		}
		else
		{
			cpu_calc_grav_accel_naive(int_bound, h_x, h_m, h_a);
		}
		Dt_CPU = ((ttt_t)(GetTimeMs64() - t0) / (ttt_t)i )/1000.0;

		print(COMPUTING_DEVICE_CPU, method_name[0], param_name[1], int_bound, 0, 1, Dt_CPU, Dt_GPU, o_result, false);
		print(COMPUTING_DEVICE_CPU, method_name[0], param_name[1], int_bound, 0, 1, Dt_CPU, Dt_GPU, o_summary, true);
	}

	//Naive symmetric method
	if (n_sink == n_source)
	{
		uint64 t0 = GetTimeMs64();
		if (100 >= n_sink)
		{
			for (i = 1; i <= 5000; i++)
			{
				cpu_calc_grav_accel_naive_sym(int_bound, h_x, h_m, h_a);
			}
		}
		else if ((100 < n_sink && 200 >= n_sink))
		{
			for (i = 1; i <= 1000; i++)
			{
				cpu_calc_grav_accel_naive_sym(int_bound, h_x, h_m, h_a);
			}
		}
		else if ((200 < n_sink && 600 >= n_sink))
		{
			for (i = 1; i <= 100; i++)
			{
				cpu_calc_grav_accel_naive_sym(int_bound, h_x, h_m, h_a);
			}
		}
		else
		{
			cpu_calc_grav_accel_naive_sym(int_bound, h_x, h_m, h_a);
		}
		Dt_CPU = ((ttt_t)(GetTimeMs64() - t0) / (ttt_t)i )/1000.0;

		print(COMPUTING_DEVICE_CPU, method_name[1], param_name[1], int_bound, 0, 1, Dt_CPU, Dt_GPU, o_result, false);
		print(COMPUTING_DEVICE_CPU, method_name[1], param_name[1], int_bound, 0, 1, Dt_CPU, Dt_GPU, o_summary, true);
	}

	//Naive symmetric advanced method
	{
		uint64 t0 = GetTimeMs64();
		if (SQR(100ULL) >= n_interaction)
		{
			for (i = 1; i <= 5000; i++)
			{
				cpu_calc_grav_accel_naive_sym_advanced(int_bound, h_x, h_m, h_a);
			}
		}
		else if ((SQR(100ULL) < n_interaction && SQR(200ULL) >= n_interaction))
		{
			for (i = 1; i <= 1000; i++)
			{
				cpu_calc_grav_accel_naive_sym_advanced(int_bound, h_x, h_m, h_a);
			}
		}
		else if ((SQR(200ULL) < n_interaction && SQR(600ULL) >= n_interaction))
		{
			for (i = 1; i <= 100; i++)
			{
				cpu_calc_grav_accel_naive_sym_advanced(int_bound, h_x, h_m, h_a);
			}
		}
		else
		{
			cpu_calc_grav_accel_naive_sym_advanced(int_bound, h_x, h_m, h_a);
		}
		Dt_CPU = ((ttt_t)(GetTimeMs64() - t0) / (ttt_t)i )/1000.0;

		string m_name = "naive_sym_advanced";
		print(COMPUTING_DEVICE_CPU, m_name, param_name[1], int_bound, 0, 1, Dt_CPU, Dt_GPU, o_result, false);
		print(COMPUTING_DEVICE_CPU, m_name, param_name[1], int_bound, 0, 1, Dt_CPU, Dt_GPU, o_summary, true);
	}
}

void benchmark_GPU(int n_body, int dev_id, const var4_t* d_x, const var_t* d_m, var4_t* d_a, ofstream& o_result, ofstream& o_summary)
{
	interaction_bound int_bound;

	o_result.setf(ios::right);
	o_result.setf(ios::scientific);

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev_id));
	int half_warp_size = deviceProp.warpSize/2;

	vector<float> execution_time;
	ttt_t Dt_CPU = 0.0;
	ttt_t Dt_GPU = 0.0;

	//Naive method
	{
		uint32_t n_pass = 0;
		for (int n_tpb = half_warp_size; n_tpb <= deviceProp.maxThreadsPerBlock/2; n_tpb += half_warp_size)
		{
			Dt_GPU = gpu_calc_grav_accel_naive_benchmark(n_body, n_tpb, d_x, d_m, d_a);
			execution_time.push_back(Dt_GPU);
			n_pass++;

			print(COMPUTING_DEVICE_GPU, method_name[0], param_name[0], int_bound, n_body, n_tpb, Dt_CPU, Dt_GPU, o_result, false);
		}
		int min_idx = min_element(execution_time.begin(), execution_time.end()) - execution_time.begin();
		print(COMPUTING_DEVICE_GPU, method_name[0], param_name[0], int_bound, n_body, (min_idx + 1) * half_warp_size, Dt_CPU, execution_time[min_idx], o_summary, true);
	}
	execution_time.clear();

	//Naive symmetric method
	{
		uint32_t n_pass = 0;
		for (int n_tpb = half_warp_size; n_tpb <= deviceProp.maxThreadsPerBlock/2; n_tpb += half_warp_size)
		{
			float Dt_GPU = gpu_calc_grav_accel_naive_sym_benchmark(n_body, n_tpb, d_x, d_m, d_a);
			execution_time.push_back(Dt_GPU);
			n_pass++;

			print(COMPUTING_DEVICE_GPU, method_name[1], param_name[0], int_bound, n_body, n_tpb, Dt_CPU, Dt_GPU, o_result, false);
		}
		int min_idx = min_element(execution_time.begin(), execution_time.end()) - execution_time.begin();
		print(COMPUTING_DEVICE_GPU, method_name[1], param_name[0], int_bound, n_body, (min_idx + 1) * half_warp_size, Dt_CPU, execution_time[min_idx], o_summary, true);
	}
	execution_time.clear();

	//Tile method
	{
		uint32_t n_pass = 0;
		for (int n_tpb = half_warp_size; n_tpb <= deviceProp.maxThreadsPerBlock/2; n_tpb += half_warp_size)
		{
			float Dt_GPU = gpu_calc_grav_accel_tile_benchmark(n_body, n_tpb, d_x, d_m, d_a);
			execution_time.push_back(Dt_GPU);
			n_pass++;

			print(COMPUTING_DEVICE_GPU, method_name[2], param_name[0], int_bound, n_body, n_tpb, Dt_CPU, Dt_GPU, o_result, false);
		}
		int min_idx = min_element(execution_time.begin(), execution_time.end()) - execution_time.begin();
		print(COMPUTING_DEVICE_GPU, method_name[2], param_name[0], int_bound, n_body, (min_idx + 1) * half_warp_size, Dt_CPU, execution_time[min_idx], o_summary, true);
	}
	execution_time.clear();

	//Tile method (advanced)
	{
		uint32_t n_pass = 0;
		for (int n_tpb = half_warp_size; n_tpb <= deviceProp.maxThreadsPerBlock/2; n_tpb += half_warp_size)
		{
			float Dt_GPU = gpu_calc_grav_accel_tile_benchmark_2(n_body, n_tpb, d_x, d_m, d_a);
			execution_time.push_back(Dt_GPU);
			n_pass++;

			print(COMPUTING_DEVICE_GPU, method_name[3], param_name[0], int_bound, n_body, n_tpb, Dt_CPU, Dt_GPU, o_result, false);
		}
		int min_idx = min_element(execution_time.begin(), execution_time.end()) - execution_time.begin();
		print(COMPUTING_DEVICE_GPU, method_name[3], param_name[0], int_bound, n_body, (min_idx + 1) * half_warp_size, Dt_CPU, execution_time[min_idx], o_summary, true);
	}
	execution_time.clear();
}

void benchmark_GPU(interaction_bound int_bound, int dev_id, const var4_t* d_x, const var_t* d_m, var4_t* d_a, ofstream& o_result, ofstream& o_summary)
{
	const int n_sink   = int_bound.sink.y   - int_bound.sink.x;
	const int n_source = int_bound.source.y - int_bound.source.x;

	o_result.setf(ios::right);
	o_result.setf(ios::scientific);

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev_id));
	int half_warp_size = deviceProp.warpSize/2;

	vector<float> execution_time;
	ttt_t Dt_CPU = 0.0;
	ttt_t Dt_GPU = 0.0;

	//Naive method
	{
		uint32_t n_pass = 0;
		for (int n_tpb = half_warp_size; n_tpb <= deviceProp.maxThreadsPerBlock/2; n_tpb += half_warp_size)
		{
			Dt_GPU = gpu_calc_grav_accel_naive_benchmark(int_bound, n_tpb, d_x, d_m, d_a);
			execution_time.push_back(Dt_GPU);
			n_pass++;

			print(COMPUTING_DEVICE_GPU, method_name[0], param_name[1], int_bound, 0, n_tpb, Dt_CPU, Dt_GPU, o_result, false);
		}
		int min_idx = min_element(execution_time.begin(), execution_time.end()) - execution_time.begin();
		print(COMPUTING_DEVICE_GPU, method_name[0], param_name[1], int_bound, 0, (min_idx + 1) * half_warp_size, Dt_CPU, execution_time[min_idx], o_summary, true);
	}
	execution_time.clear();

	//Naive symmetric method
	if (n_sink == n_source)
	{
		uint32_t n_pass = 0;
		for (int n_tpb = half_warp_size; n_tpb <= deviceProp.maxThreadsPerBlock/2; n_tpb += half_warp_size)
		{
			float Dt_GPU = gpu_calc_grav_accel_naive_sym_benchmark(int_bound, n_tpb, d_x, d_m, d_a);
			execution_time.push_back(Dt_GPU);
			n_pass++;

			print(COMPUTING_DEVICE_GPU, method_name[1], param_name[1], int_bound, 0, n_tpb, Dt_CPU, Dt_GPU, o_result, false);
		}
		int min_idx = min_element(execution_time.begin(), execution_time.end()) - execution_time.begin();
		print(COMPUTING_DEVICE_GPU, method_name[1], param_name[1], int_bound, 0, (min_idx + 1) * half_warp_size, Dt_CPU, execution_time[min_idx], o_summary, true);
	}
	else
	{
		print(COMPUTING_DEVICE_GPU, method_name[1], param_name[1], int_bound, 0, 0, 0, 0, o_summary, true);
	}
	execution_time.clear();

	//Tile method
	{
		uint32_t n_pass = 0;
		for (int n_tpb = half_warp_size; n_tpb <= deviceProp.maxThreadsPerBlock/2; n_tpb += half_warp_size)
		{
			float Dt_GPU = gpu_calc_grav_accel_tile_benchmark(int_bound, n_tpb, d_x, d_m, d_a);
			execution_time.push_back(Dt_GPU);
			n_pass++;

			print(COMPUTING_DEVICE_GPU, method_name[2], param_name[1], int_bound, 0, n_tpb, Dt_CPU, Dt_GPU, o_result, false);
		}
		int min_idx = min_element(execution_time.begin(), execution_time.end()) - execution_time.begin();
		print(COMPUTING_DEVICE_GPU, method_name[2], param_name[1], int_bound, 0, (min_idx + 1) * half_warp_size, Dt_CPU, execution_time[min_idx], o_summary, true);
	}
	execution_time.clear();

	//Tile method (advanced)
	{
		uint32_t n_pass = 0;
		for (int n_tpb = half_warp_size; n_tpb <= deviceProp.maxThreadsPerBlock/2; n_tpb += half_warp_size)
		{
			float Dt_GPU = gpu_calc_grav_accel_tile_benchmark_2(int_bound, n_tpb, d_x, d_m, d_a);
			execution_time.push_back(Dt_GPU);
			n_pass++;

			print(COMPUTING_DEVICE_GPU, method_name[3], param_name[1], int_bound, 0, n_tpb, Dt_CPU, Dt_GPU, o_result, false);
		}
		int min_idx = min_element(execution_time.begin(), execution_time.end()) - execution_time.begin();
		print(COMPUTING_DEVICE_GPU, method_name[3], param_name[1], int_bound, 0, (min_idx + 1) * half_warp_size, Dt_CPU, execution_time[min_idx], o_summary, true);
	}
	execution_time.clear();
}

void populate(int n, var4_t* h_x, var_t* h_m)
{
	for (int i = 0; i < n; i++)
	{
		h_x[i].x = -50.0 + ((var_t)rand() / RAND_MAX) * 100.0;
		h_x[i].y = -50.0 + ((var_t)rand() / RAND_MAX) * 100.0;
		h_x[i].z = -50.0 + ((var_t)rand() / RAND_MAX) * 100.0;
		h_x[i].w = 0.0;

		h_m[i] = 1.0;
	}
}

bool compare_vectors(int n, const var4_t* v1, const var4_t* v2, var_t tolerance, bool verbose)
{
	vector<var_t> list_ad;
	vector<var_t> list_rd;

	bool success = true;

	for (int i = 0; i < n; i++)
	{
		var_t diff = fabs(v1[i].x - v2[i].x);
		var_t rel  = diff / fabs(v1[i].x);
		if (tolerance <= diff)
		{
			if (success && verbose) printf("\n");
			if (verbose) printf("\tError: i = [%5d] v1.x = %16.8le v2.x = %16.8le D = %16.8le rel = %16.8le\n", i, v1[i].x, v2[i].x, diff, rel);
			list_ad.push_back(diff);
			list_rd.push_back(rel);
			success = false;
		}
		diff = fabs(v1[i].y - v2[i].y);
		rel  = diff / fabs(v1[i].y);
		if (tolerance <= diff)
		{
			if (success && verbose) printf("\n");
			if (verbose) printf("\tError: i = [%5d] v1.y = %16.8le v2.y = %16.8le D = %16.8le rel = %16.8le\n", i, v1[i].y, v2[i].y, diff, rel);
			list_ad.push_back(diff);
			list_rd.push_back(rel);
			success = false;
		}
		diff = fabs(v1[i].z - v2[i].z);
		rel  = diff / fabs(v1[i].z);
		if (tolerance <= diff)
		{
			if (success && verbose) printf("\n");
			if (verbose) printf("\tError: i = [%5d] v1.z = %16.8le v2.x = %16.8le D = %16.8le rel = %16.8le\n", i, v1[i].z, v2[i].z, diff, rel);
			list_ad.push_back(diff);
			list_rd.push_back(rel);
			success = false;
		}
	}
	if (success)
	{
		printf("Identical\n");
	}
	else
	{
	    var_t max_ad = *max_element(list_ad.begin(), list_ad.end());
		var_t max_rd = *max_element(list_rd.begin(), list_rd.end());

		printf("Different (max. absolut difference: %16.8le, max. relative difference: %16.8le\n", max_ad, max_rd);
	}
	return success;
}

void compare_results(int n_body, int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a, const var4_t* h_x, const var_t* h_m, var4_t* h_a, var4_t* h_at, var_t tolerance)
{
	printf("\nComparing the computations of the gravitational accelerations performed\non the CPU with two different functions naive() and naive_sym(): ");
	cpu_calc_grav_accel_naive(    n_body, h_x, h_m, h_a );
	cpu_calc_grav_accel_naive_sym(n_body, h_x, h_m, h_at);

	bool result = compare_vectors(n_body, h_a, h_at, tolerance, false);

	memset(h_a,  0, n_body*sizeof(var4_t));
	memset(h_at, 0, n_body*sizeof(var4_t));

	dim3 grid((n_body + n_tpb - 1)/n_tpb);
	dim3 block(n_tpb);

	printf("\nComparing the computations of the gravitational accelerations performed on the CPU and GPU:\n");
	printf("1. CPU naive vs GPU naive    : ");
	cpu_calc_grav_accel_naive(n_body, h_x, h_m, h_a);
	kernel::calc_gravity_accel_naive<<<grid, block>>>(n_body, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();
	copy_vector_to_host(h_at, d_a, n_body*sizeof(var4_t));
	result = compare_vectors(n_body, h_a, h_at, tolerance, false);
	memset(h_at, 0, n_body*sizeof(var4_t));

	printf("2. CPU naive vs GPU naive_sym: ");
	CUDA_SAFE_CALL(cudaMemset(d_a, 0, n_body*sizeof(var4_t)));
	kernel::calc_gravity_accel_naive_sym<<<grid, block>>>(n_body, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();
	copy_vector_to_host(h_at, d_a, n_body*sizeof(var4_t));
	result = compare_vectors(n_body, h_a, h_at, tolerance, false);
	memset(h_at, 0, n_body*sizeof(var4_t));

	printf("3. CPU naive vs GPU tile     : ");
	kernel::calc_gravity_accel_tile<<<grid, block, n_tpb*sizeof(var4_t)>>>(n_tpb, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();
	copy_vector_to_host(h_at, d_a, n_body*sizeof(var4_t));
	result = compare_vectors(n_body, h_a, h_at, tolerance, false);
	memset(h_at, 0, n_body*sizeof(var4_t));

	printf("4. CPU naive vs GPU tile (without the explicit use of tile_size): ");
	kernel::calc_gravity_accel_tile<<<grid, block, n_tpb*sizeof(var4_t)>>>(n_body, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();
	copy_vector_to_host(h_at, d_a, n_body*sizeof(var4_t));
	result = compare_vectors(n_body, h_a, h_at, tolerance, false);
	memset(h_at, 0, n_body*sizeof(var4_t));
}

void compare_CPU_results(interaction_bound int_bound, int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a, const var4_t* h_x, const var_t* h_m, var4_t* h_a, var4_t* h_at, var_t tolerance, bool verbose)
{
	const int n_sink   = int_bound.sink.y   - int_bound.sink.x;
	const int n_source = int_bound.source.y - int_bound.source.x;

	printf("Comparing the computations of the gravitational accelerations performed on the CPU:\n");

	memset(h_a,  0, n_sink * sizeof(var4_t));
	memset(h_at, 0, n_sink * sizeof(var4_t));

	// Computing base result
	cpu_calc_grav_accel_naive(    int_bound, h_x, h_m, h_a );

	if (n_sink == n_source)
	{
		printf("\tNaive vs. naive symmetric          : ");
		cpu_calc_grav_accel_naive_sym(int_bound, h_x, h_m, h_at);
	}
	else
	{
		printf("\tNaive vs. naive symmetric advanced : ");
		cpu_calc_grav_accel_naive_sym_advanced(int_bound, h_x, h_m, h_at);
	}

	bool result = compare_vectors(n_sink, h_a, h_at, tolerance, verbose);
}

void compare_CPU_GPU_results(interaction_bound int_bound, int n_tpb, const var4_t* d_x, const var_t* d_m, var4_t* d_a, const var4_t* h_x, const var_t* h_m, var4_t* h_a, var4_t* h_at, var_t tolerance, bool verbose)
{
	const int n_sink = int_bound.sink.y - int_bound.sink.x;
	const int n_source = int_bound.source.y - int_bound.source.x;

	printf("Comparing the computations of the gravitational accelerations performed on the CPU and GPU:\n");

	memset(h_a,  0, n_sink * sizeof(var4_t));
	memset(h_at, 0, n_sink * sizeof(var4_t));

	dim3 grid((n_sink + n_tpb - 1)/n_tpb);
	dim3 block(n_tpb);

	// DEBUG CODE
	//var_t* _m = (var_t *)h_m;
	//_m[0] *= 10.0;

	// Compute base result and store it in h_a. This will be compared with the GPU result.
	cpu_calc_grav_accel_naive(int_bound, h_x, h_m, h_a);

	printf("\tCPU naive vs. GPU naive            : ");
	kernel2::calc_gravity_accel_naive<<<grid, block>>>(int_bound, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();
	copy_vector_to_host(h_at, d_a, n_sink * sizeof(var4_t));
	bool result = compare_vectors(n_sink, h_a, h_at, tolerance, verbose);

	memset(h_at, 0, n_sink * sizeof(var4_t));
	CUDA_SAFE_CALL(cudaMemset(d_a, 0, n_sink * sizeof(var4_t)));

	printf("\tCPU naive vs. GPU naive symmetric  : ");
	if (n_sink == n_source)
	{
		kernel2::calc_gravity_accel_naive_sym<<<grid, block>>>(int_bound, d_x, d_m, d_a);
		CUDA_CHECK_ERROR();
		copy_vector_to_host(h_at, d_a, n_sink * sizeof(var4_t));
		result = compare_vectors(n_sink, h_a, h_at, tolerance, verbose);

		memset(h_at, 0, n_sink * sizeof(var4_t));
		CUDA_SAFE_CALL(cudaMemset(d_a, 0, n_sink * sizeof(var4_t)));
	}
	else
	{
		printf("The number of sink body does not equal to the number of source body.\n");
	}

	printf("\tCPU naive vs. GPU tile             : ");
	kernel2::calc_gravity_accel_tile<<<grid, block, n_tpb * sizeof(var4_t)>>>(int_bound, n_tpb, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();
	copy_vector_to_host(h_at, d_a, n_sink * sizeof(var4_t));
	result = compare_vectors(n_sink, h_a, h_at, tolerance, verbose);

	memset(h_at, 0, n_sink * sizeof(var4_t));
	CUDA_SAFE_CALL(cudaMemset(d_a, 0, n_sink * sizeof(var4_t)));

	printf("\tCPU naive vs. GPU advanced tile    : ");
	kernel2::calc_gravity_accel_tile<<<grid, block, n_tpb * sizeof(var4_t)>>>(int_bound, d_x, d_m, d_a);
	CUDA_CHECK_ERROR();
	copy_vector_to_host(h_at, d_a, n_sink * sizeof(var4_t));
	result = compare_vectors(n_sink, h_a, h_at, tolerance, verbose);

	memset(h_at, 0, n_sink * sizeof(var4_t));
	CUDA_SAFE_CALL(cudaMemset(d_a, 0, n_sink * sizeof(var4_t)));
}

void compare_results(option_t& opt, bool verbose)
{
	var4_t* h_x = 0x0;
	var4_t* h_a = 0x0;
	var_t* h_m = 0x0;

	var4_t* d_x = 0x0;
	var4_t* d_a = 0x0;
	var_t* d_m = 0x0;

	var4_t* h_at = 0x0;

	if (COMPUTING_DEVICE_GPU == opt.comp_dev)
	{
		set_device(opt.id_dev, cout);
	}

	// The user wants a benchmark only for the specified number of bodies
	if (0 == opt.n1)
	{
		opt.n1 = opt.n0;
	}
	// The user defined dn in order to carry out a benchmark for different number of bodies
	// so the benchmark will be carry on for different numbers from n0 to n1
	else
	{
		;
	}

	// Total number of bodies is the larger of n_snk and n_src
	int n_total = max(opt.n0, opt.n1);
	ALLOCATE_HOST_VECTOR((void**)&h_x,  n_total*sizeof(var4_t)); 
	ALLOCATE_HOST_VECTOR((void**)&h_a,  n_total*sizeof(var4_t)); 
	ALLOCATE_HOST_VECTOR((void**)&h_at, n_total*sizeof(var4_t)); 
	ALLOCATE_HOST_VECTOR((void**)&h_m,  n_total*sizeof(var_t)); 

	if (COMPUTING_DEVICE_GPU == opt.comp_dev)
	{
		ALLOCATE_DEVICE_VECTOR((void**)&d_x, n_total*sizeof(var4_t)); 
		ALLOCATE_DEVICE_VECTOR((void**)&d_a, n_total*sizeof(var4_t)); 
		ALLOCATE_DEVICE_VECTOR((void**)&d_m, n_total*sizeof(var_t)); 
	}

	populate(n_total, h_x, h_m);

	if (COMPUTING_DEVICE_GPU == opt.comp_dev)
	{
		copy_vector_to_device(d_x, h_x, n_total*sizeof(var4_t));
		copy_vector_to_device(d_m, h_m, n_total*sizeof(var_t));
	}

	uint2_t sink   = {0, opt.n0};
	uint2_t source = {0, opt.n1};
	interaction_bound int_bound(sink, source);

	cout << "tolerance level   = " << scientific << opt.tol << endl;
	cout << "n_sink * n_source = " << setw(6) << opt.n0 << " * " << setw(6) << opt.n1 << " = " << setw(12) << opt.n0*opt.n1 << endl << endl;
	compare_CPU_results(int_bound, 1, d_x, d_m, d_a, h_x, h_m, h_a, h_at, opt.tol, verbose);
	printf("\n");

	if (COMPUTING_DEVICE_GPU == opt.comp_dev)
	{
		compare_CPU_GPU_results(int_bound, 256, d_x, d_m, d_a, h_x, h_m, h_a, h_at, opt.tol, verbose);
	}

	FREE_HOST_VECTOR((void**)&h_x);
	FREE_HOST_VECTOR((void**)&h_a);
	FREE_HOST_VECTOR((void**)&h_m);
	FREE_HOST_VECTOR((void**)&h_at);

	if (COMPUTING_DEVICE_GPU == opt.comp_dev)
	{
		FREE_DEVICE_VECTOR((void**)&d_x);
		FREE_DEVICE_VECTOR((void**)&d_a);
		FREE_DEVICE_VECTOR((void**)&d_m);
	}
}

//! Benchmarking the different CPU and GPU gravitational interaction functions. 
/*!
  The number of interaction is the number of sink bodies multiplied by the number of source bodies, i.e. the layout
  can be a rectangle of sink and source bodies.
  \param id_dev an integer identifing the device performing the computationa
  \param n0 the initial number of bodies (both the sink and source)
  \param n1 the maximum number of bodies (both the sink and source)
  \param dn in increase in the number of n0 and n1
  \param n_iter after n_iter iteration the increase dn will be multiplied by 10
  \param o_result output filestream to store the detailed results of the computation
  \param o_summary output filestream to store the summary of the computation
*/
void benchmark(int id_dev, int n0, int n1, int dn, int n_iter, ofstream& o_result, ofstream& o_summary)
{
	var4_t* h_x = 0x0;
	var4_t* h_a = 0x0;
	var_t* h_m = 0x0;

	var4_t* d_x = 0x0;
	var4_t* d_a = 0x0;
	var_t* d_m = 0x0;

	var4_t* h_at = 0x0;

	set_device(id_dev, cout);

	// The user wants a benchmark only for the specified number of bodies
	if (0 == n1)
	{
		n1 = n0;
	}
	// The user defined dn in order to carry out a benchmark for different number of bodies
	// so the benchmark will be carry on for different numbers from n0 to n1
	else
	{
		;
	}

	int dn_snk = dn;
	int k_snk = 1;
	for (int n_snk = n0; n_snk <= n1; n_snk += dn_snk, k_snk++)
	{
		int dn_src = dn;
		int k_src = 1;
		for (int n_src = n0; n_src <= n1; n_src += dn_src, k_src++)
		{
			// Total number of bodies is the larger of n_snk and n_src
			int n_total = max(n_snk, n_src);
			ALLOCATE_HOST_VECTOR((void**)&h_x,  n_total*sizeof(var4_t)); 
			ALLOCATE_HOST_VECTOR((void**)&h_a,  n_total*sizeof(var4_t)); 
			ALLOCATE_HOST_VECTOR((void**)&h_at, n_total*sizeof(var4_t)); 
			ALLOCATE_HOST_VECTOR((void**)&h_m,  n_total*sizeof(var_t)); 

			ALLOCATE_DEVICE_VECTOR((void**)&d_x, n_total*sizeof(var4_t)); 
			ALLOCATE_DEVICE_VECTOR((void**)&d_a, n_total*sizeof(var4_t)); 
			ALLOCATE_DEVICE_VECTOR((void**)&d_m, n_total*sizeof(var_t)); 

			populate(n_total, h_x, h_m);

			copy_vector_to_device(d_x, h_x, n_total*sizeof(var4_t));
			copy_vector_to_device(d_m, h_m, n_total*sizeof(var_t));

			uint2_t sink = {0, n_snk};
			uint2_t source = {0, n_src};
			interaction_bound int_bound(sink, source);

			cout << "(n_sink * n_source = " << setw(6) << n_snk << " * " << setw(6) << n_src << " = " << setw(12) << (uint32_t)n_snk*(uint32_t)n_src << ")---------------------------------------------------------------" << endl;
			cout << "CPU Gravity acceleration:" << endl;
			benchmark_CPU(int_bound, h_x, h_m, h_a, o_result, o_summary);
			printf("\n");

			cout << "-------------------------------------------------------------------------------------------------------------------" << endl;
			cout << "CPU Gravity acceleration:" << endl;
			benchmark_GPU(int_bound, id_dev, d_x, d_m, d_a, o_result, o_summary);
			printf("\n");

			if (0 < n_iter && 0 == k_src % n_iter)
			{
				k_src = 1;
				dn_src *= 10;
			}

			FREE_HOST_VECTOR((void**)&h_x);
			FREE_HOST_VECTOR((void**)&h_a);
			FREE_HOST_VECTOR((void**)&h_m);
			FREE_HOST_VECTOR((void**)&h_at);

			FREE_DEVICE_VECTOR((void**)&d_x);
			FREE_DEVICE_VECTOR((void**)&d_a);
			FREE_DEVICE_VECTOR((void**)&d_m);
		}
		if (0 < n_iter && 0 == k_snk % n_iter)
		{
			k_snk = 1;
			dn_snk *= 10;
		}
	}
}

void benchmark_GPU(option& opt, ofstream& o_result, ofstream& o_summary)
{
	var4_t* h_x = 0x0;
	var4_t* h_a = 0x0;
	var_t* h_m = 0x0;

	var4_t* d_x = 0x0;
	var4_t* d_a = 0x0;
	var_t* d_m = 0x0;

	set_device(opt.id_dev, cout);

	// The user wants a benchmark only for the specified number of bodies
	if (0 == opt.n1)
	{
		opt.n1 = opt.n0;
	}
	// The user defined dn in order to carry out a benchmark for different number of bodies
	// so the benchmark will be carry on for different numbers from n0 to n1
	else
	{
		;
	}

	cout << "GPU Gravity acceleration:" << endl;
	int dn_snk = opt.dn;
	int k_snk = 1;
	for (int n_snk = opt.n0; n_snk <= opt.n1; n_snk += dn_snk, k_snk++)
	{
		int dn_src = opt.dn;
		int k_src = 1;
		for (int n_src = opt.n0; n_src <= opt.n1; n_src += dn_src, k_src++)
		{
			// Total number of bodies is the larger of n_snk and n_src
			int n_total = max(n_snk, n_src);
			ALLOCATE_HOST_VECTOR((void**)&h_x,  n_total*sizeof(var4_t)); 
			ALLOCATE_HOST_VECTOR((void**)&h_a,  n_total*sizeof(var4_t)); 
			ALLOCATE_HOST_VECTOR((void**)&h_m,  n_total*sizeof(var_t)); 

			ALLOCATE_DEVICE_VECTOR((void**)&d_x, n_total*sizeof(var4_t)); 
			ALLOCATE_DEVICE_VECTOR((void**)&d_a, n_total*sizeof(var4_t)); 
			ALLOCATE_DEVICE_VECTOR((void**)&d_m, n_total*sizeof(var_t)); 

			populate(n_total, h_x, h_m);

			copy_vector_to_device(d_x, h_x, n_total*sizeof(var4_t));
			copy_vector_to_device(d_m, h_m, n_total*sizeof(var_t));

			uint2_t sink = {0, n_snk};
			uint2_t source = {0, n_src};
			interaction_bound int_bound(sink, source);

			cout << "(n_sink * n_source = " << setw(6) << n_snk << " * " << setw(6) << n_src << " = " << setw(12) << (uint32_t)n_snk*(uint32_t)n_src << ")-------------------------------------------------------------------" << endl;
			benchmark_GPU(int_bound, opt.id_dev, d_x, d_m, d_a, o_result, o_summary);

			if (0 < opt.n_iter && 0 == k_src % opt.n_iter)
			{
				k_src = 1;
				dn_src *= 10;
			}

			FREE_HOST_VECTOR((void**)&h_x);
			FREE_HOST_VECTOR((void**)&h_a);
			FREE_HOST_VECTOR((void**)&h_m);

			FREE_DEVICE_VECTOR((void**)&d_x);
			FREE_DEVICE_VECTOR((void**)&d_a);
			FREE_DEVICE_VECTOR((void**)&d_m);
		}
		if (0 < opt.n_iter && 0 == k_snk % opt.n_iter)
		{
			k_snk = 1;
			dn_snk *= 10;
		}
	}
}

void benchmark_CPU(option& opt, ofstream& o_result, ofstream& o_summary)
{
	var4_t* h_x = 0x0;
	var4_t* h_a = 0x0;
	var_t* h_m = 0x0;

	// The user wants a benchmark only for the specified number of bodies
	if (0 == opt.n1)
	{
		opt.n1 = opt.n0;
	}
	// The user defined dn in order to carry out a benchmark for different number of bodies
	// so the benchmark will be carry on for different numbers from n0 to n1
	else
	{
		;
	}

	cout << "CPU Gravity acceleration:" << endl;
	int dn_snk = opt.dn;
	int k_snk = 1;
	for (int n_snk = opt.n0; n_snk <= opt.n1; n_snk += dn_snk, k_snk++)
	{
		int dn_src = opt.dn;
		int k_src = 1;
		for (int n_src = opt.n0; n_src <= opt.n1; n_src += dn_src, k_src++)
		{
			// Total number of bodies is the larger of n_snk and n_src
			int n_total = max(n_snk, n_src);
			ALLOCATE_HOST_VECTOR((void**)&h_x,  n_total*sizeof(var4_t)); 
			ALLOCATE_HOST_VECTOR((void**)&h_a,  n_total*sizeof(var4_t)); 
			ALLOCATE_HOST_VECTOR((void**)&h_m,  n_total*sizeof(var_t)); 

			populate(n_total, h_x, h_m);

			uint2_t sink = {0, n_snk};
			uint2_t source = {0, n_src};
			interaction_bound int_bound(sink, source);

			cout << "(n_sink * n_source = " << setw(6) << n_snk << " * " << setw(6) << n_src << " = " << setw(12) << (uint32_t)n_snk*(uint32_t)n_src << ")-------------------------------------------------------------------" << endl;
			benchmark_CPU(int_bound, h_x, h_m, h_a, o_result, o_summary);
			printf("\n");

			if (0 < opt.n_iter && 0 == k_src % opt.n_iter)
			{
				k_src = 1;
				dn_src *= 10;
			}

			FREE_HOST_VECTOR((void**)&h_x);
			FREE_HOST_VECTOR((void**)&h_a);
			FREE_HOST_VECTOR((void**)&h_m);
		}
		if (0 < opt.n_iter && 0 == k_snk % opt.n_iter)
		{
			k_snk = 1;
			dn_snk *= 10;
		}
	}
}

string create_prefix()
{
	string prefix;

	string config;
#ifdef _DEBUG
	config = "D";
#else
	config = "R";
#endif
	prefix += config;

	return prefix;
}

void write_log(int argc, const char** argv, const char** env, option_t& opt, ofstream& sout)
{
	string data;

	sout << tools::get_time_stamp(false) << " starting:" << endl;
	for (int i = 0; i < argc; i++)
	{
		sout << argv[i] << SEP;
	}
	sout << endl << endl;

	char **p = (char **)env;
	for ( ; *p != 0x0; *p++)
	{
		string s = *p;
#if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__WINDOWS__) || defined(__TOS_WIN__)
		if(      s.find("COMPUTERNAME=") < s.length())
		{
			sout << s << endl;
		}
		else if (s.find("USERNAME=") < s.length())
		{
			sout << s << endl;
		}
		else if (s.find("OS=") < s.length())
		{
			sout << s << endl;
		}
#else  /* presume POSIX */
		if(      s.find("HOSTNAME=") < s.length())
		{
			sout << s << endl;
		}
		else if (s.find("USER=") < s.length())
		{
			sout << s << endl;
		}
		else if (s.find("OSTYPE=") < s.length())
		{
			sout << s << endl;
		}
#endif
	}
	sout << endl;

#if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__WINDOWS__) || defined(__TOS_WIN__)

	p = (char **)env;
	for ( ; *p != 0x0; *p++)
	{
		string s = *p;
		if(      s.find("PROCESSOR_IDENTIFIER=") < s.length())
		{
			sout << s << endl;
		}
		else if (s.find("NUMBER_OF_PROCESSORS=") < s.length())
		{
			sout << s << endl;
		}
		else if (s.find("PROCESSOR_ARCHITECTURE=") < s.length())
		{
			sout << s << endl;
		}
		else if (s.find("PROCESSOR_LEVEL=") < s.length())
		{
			sout << s << endl;
		}
	}
#else  /* presume POSIX */
	string path = "/proc/cpuinfo";

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
			break;
		}
		sout << line << endl;
	}
#endif
	sout << endl;

	if (COMPUTING_DEVICE_GPU == opt.comp_dev)
	{
		device_query(sout, opt.id_dev, false);
	}

	sout.flush();
}

void open_streams(string& o_dir, string& result_filename, string& summray_filename, string& log_filename, ofstream** output)
{
	string path = file::combine_path(o_dir, result_filename);
	output[BENCHMARK_OUTPUT_NAME_RESULT] = new ofstream(path.c_str(), ios::out);
	if (!*output[BENCHMARK_OUTPUT_NAME_RESULT]) 
	{
		throw string("Cannot open " + path + ".");
	}

	path = file::combine_path(o_dir, summray_filename);
	output[BENCHMARK_OUTPUT_NAME_SUMMARY] = new ofstream(path.c_str(), ios::out);
	if (!*output[BENCHMARK_OUTPUT_NAME_SUMMARY]) 
	{
		throw string("Cannot open " + path + ".");
	}

	path = file::combine_path(o_dir, log_filename);
	output[BENCHMARK_OUTPUT_NAME_LOG] = new ofstream(path.c_str(), ios::out);
	if (!*output[BENCHMARK_OUTPUT_NAME_LOG]) 
	{
		throw string("Cannot open " + path + ".");
	}
}

void create_filename(option_t& opt, cpu_info_t& cpu_info, string& result_filename, string& summary_filename, string& log_filename)
{
	const char sep = '_';

	string cuda_dev_name;
	string cpu_name;

	if (COMPUTING_DEVICE_GPU == opt.comp_dev)
	{
		cuda_dev_name = redutilcu::get_device_name(opt.id_dev);
		std::replace(cuda_dev_name.begin(), cuda_dev_name.end(), ' ', '_');
	}
	else
	{
		cpu_name = cpu_info.model_name;
		std::replace(cpu_name.begin(), cpu_name.end(), ',', '_');
		std::replace(cpu_name.begin(), cpu_name.end(), '(', '_');
		std::replace(cpu_name.begin(), cpu_name.end(), ')', '_');
		std::replace(cpu_name.begin(), cpu_name.end(), ' ', '_');
	}

#if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__WINDOWS__) || defined(__TOS_WIN__)
	result_filename = create_prefix() + sep;
#endif
	result_filename += opt.base_fn;

	if (COMPUTING_DEVICE_GPU == opt.comp_dev)
	{
		result_filename += sep + cuda_dev_name;
	}
	else
	{
		result_filename += sep + cpu_name;
	}

	summary_filename = result_filename + ".summary.txt";
	log_filename     = result_filename + ".info.txt";
	result_filename += ".txt";
}

void parse_cpu_info(vector<string>& data, cpu_info_t& cpu_info)
{
#if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__WINDOWS__) || defined(__TOS_WIN__)
	char delimiter = '=';
#else  /* presume POSIX */
	char delimiter = ':';
#endif

	string line;
	size_t pos0 = 0;
	size_t pos1 = data[0].find_first_of('\n');;
	do
	{
		line = data[0].substr(pos0, pos1 - pos0);
		size_t p0 = line.find_first_of(delimiter);
		string key = line.substr(0, p0);
		string value = line.substr(p0+1, line.length());

#if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__WINDOWS__) || defined(__TOS_WIN__)
		if ("PROCESSOR_IDENTIFIER" == key)
		{
			cpu_info.model_name = value;
		}
#else  /* presume POSIX */
		key = tools::trim(key);
		value = tools::trim(value);
		if ("model name" == key)
		{
			cpu_info.model_name = value;
		}
#endif
		// Increase by 1 in order to skip the newline at the end of the previous string
		pos0 = pos1 + 1;
		pos1 = data[0].find_first_of('\n', pos0+1);
	} while (pos1 != std::string::npos && pos1 <= data[0].length());
}

void read_cpu_description(const char** env, vector<string>& result)
{
	string data;

#if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__WINDOWS__) || defined(__TOS_WIN__)

	char **p = (char **)env;
	for ( ; *p != 0x0; *p++)
	{
		string element = *p;
		data += element + '\n';
	}
#else  /* presume POSIX */

	string path = "/proc/cpuinfo";

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
#endif
	result.push_back(data);
	data.clear();
}

int parse_options(int argc, const char **argv, option_t& opt, bool& verbose)
{
	int i = 1;

	while (i < argc)
	{
		string p = argv[i];
		string v;

		if (     p == "-oDir")
		{
			i++;
			opt.o_dir = argv[i];
		}
		else if (p == "-bFile")
		{
			i++;
			opt.base_fn = argv[i];
		}
		else if (p == "-CPU")
		{
			opt.comp_dev = COMPUTING_DEVICE_CPU;
		}
		else if (p == "-GPU")
		{
			opt.comp_dev = COMPUTING_DEVICE_GPU;
		}
		else if (p == "-devId")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			opt.id_dev = atoi(argv[i]);
		}
		else if (p == "-tol")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			opt.tol = atof(argv[i]);
			opt.compare = true;
		}
		else if (p == "-n0")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			opt.n0 = atoi(argv[i]);
		}
		else if (p == "-n1")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			opt.n1 = atoi(argv[i]);
		}
		else if (p == "-dn")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			opt.dn = atoi(argv[i]);
		}
		else if (p == "-n_iter")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			opt.n_iter = atoi(argv[i]);
		}
		else if (p == "-v" || p == "--verbose")
		{
			i++;
			v = argv[i];
			if      (v == "true")
			{
				verbose = true;
			}
			else if (v == "false")
			{
				verbose = false;
			}
			else
			{
				throw string("Invalid value at: " + p);
			}
		}
		else if (p == "-h")
		{
			printf("Usage:\n");
			printf("\n\t-CPU               : the benchmark will be carry on the CPU\n");
			printf("\n\t-GPU               : the benchmark will be carry on the GPU\n");
			printf("\n\t-devId <number>    : the id of the GPU to benchmark\n");
			printf("\n\t-n0 <number>       : the starting number of SI bodies\n");
			printf("\n\t-n1 <number>       : the end number of SI bodies\n");
			printf("\n\t-dn <number>       : at each iteration the number of bodies will be increased by dn\n");
			printf("\n\t-n_iter <number>   : after n_iter the value of dn will be multiplyed by a factor of 10\n");
			printf("\n\t-tol <number>      : the comparison will be done with the defined tolarance level (default value is 1.0e-16)\n");
			printf("\n\t-v <true|false>    : the detailed result of the comparison will be printed to the screen (default value is false)\n");
			printf("\n\t-oDir <filename>   : the output file will be stored in this directory\n");
			printf("\n\t-bFile <filename>  : the base filename of the output (without extension), it will be extended by CPU and GPU name\n");
			printf("\n\t-h                 : print this help\n");
			exit(EXIT_SUCCESS);
		}
		else
		{
			throw string("Invalid switch on command-line: " + p + ".");
		}
		i++;
	}

	if (opt.compare)
	{
		opt.job_name = JOB_NAME_COMPARE;
	}
	else
	{	
		if (COMPUTING_DEVICE_CPU == opt.comp_dev)
		{
			opt.job_name = JOB_NAME_BENCMARK_CPU;
		}
		else if (COMPUTING_DEVICE_GPU == opt.comp_dev)
		{
			opt.job_name = JOB_NAME_BENCMARK_GPU;
		}
	}

	return i;
}

void create_default_option(option_t& opt)
{
	opt.base_fn  = "";
	opt.compare  = false;
	opt.comp_dev = COMPUTING_DEVICE_CPU;
	opt.dn       = 1;
	opt.id_dev   = 0;
	opt.job_name = JOB_NAME_UNDEFINED;
	opt.n0       = 0;
	opt.n1       = 0;
	opt.n_iter   = 10;
	opt.o_dir    = "";
	opt.tol      = 1.0e-16;
}

// -oDir C:\Work\red.cuda.Results\Benchmark -bFile benchmark_rect -devId 0 -n0 10 -n1 1000 -dn 10 -n_iter 10
int main(int argc, const char** argv, const char** env)
{
	static const string header_str  = "date       time     dev  method_name             param_name        n_snk  n_src  n_bdy  n_tpb Dt_CPU[ms]    Dt_GPU[ms]";

	option_t opt;
	create_default_option(opt);
	
	string result_filename;
	string summary_filename;
	string log_filename;

	bool verbose = false;

	ofstream* output[BENCHMARK_OUTPUT_NAME_N];
	memset(output, 0x0, sizeof(output));

	time_t start = time(NULL);
	try
	{
		parse_options(argc, argv, opt, verbose);

		switch (opt.job_name)
		{
		case JOB_NAME_COMPARE:
			{
			compare_results(opt, verbose);
			break;
			}
		case JOB_NAME_BENCMARK_CPU:
			{
			vector<string> cpu_data;
			cpu_info_t cpu_info;

			read_cpu_description(env, cpu_data);
			parse_cpu_info(cpu_data, cpu_info);

			create_filename(opt, cpu_info, result_filename, summary_filename, log_filename);
			open_streams(opt.o_dir, result_filename, summary_filename, log_filename, output);

			write_log(argc, argv, env, opt, *output[BENCHMARK_OUTPUT_NAME_LOG]);

			*output[BENCHMARK_OUTPUT_NAME_RESULT ] << header_str << endl;
			*output[BENCHMARK_OUTPUT_NAME_SUMMARY] << header_str << endl;

			benchmark_CPU(opt, *output[BENCHMARK_OUTPUT_NAME_RESULT], *output[BENCHMARK_OUTPUT_NAME_SUMMARY]);
			break;
			}
		case JOB_NAME_BENCMARK_GPU:
			{
			vector<string> cpu_data;
			cpu_info_t cpu_info;

			cudaDeviceProp deviceProp;
			CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, opt.id_dev));

			create_filename(opt, cpu_info, result_filename, summary_filename, log_filename);
			open_streams(opt.o_dir, result_filename, summary_filename, log_filename, output);

			write_log(argc, argv, env, opt, *output[BENCHMARK_OUTPUT_NAME_LOG]);

			*output[BENCHMARK_OUTPUT_NAME_RESULT ] << header_str << endl;
			*output[BENCHMARK_OUTPUT_NAME_SUMMARY] << header_str << endl;

			benchmark_GPU(opt, *output[BENCHMARK_OUTPUT_NAME_RESULT], *output[BENCHMARK_OUTPUT_NAME_SUMMARY]);
			}
			break;
		default:
				throw string("Invalid job type.");
		}
	}
	catch(const string& msg)
	{
		cerr << "Error: " << msg << endl;
		exit(EXIT_FAILURE);
	}
	cout << "Total time: " << time(NULL) - start << " s" << endl;

	return (EXIT_SUCCESS);
}
