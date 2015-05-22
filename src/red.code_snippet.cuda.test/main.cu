// includes system
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
#include "red_constants.h"
#include "red_type.h"
#include "red_macro.h"
#include "redutilcu.h"
#include "red_test.h"

__constant__ var_t dc_threshold[THRESHOLD_N];
__constant__ analytic_gas_disk_params_t dc_anal_gd_params;


using namespace std;
using namespace redutilcu;

static __global__
	void kernel_print_constant_memory()
{
	printf("dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE        ] : %lf\n", dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE]);
	printf("dc_threshold[THRESHOLD_EJECTION_DISTANCE           ] : %lf\n", dc_threshold[THRESHOLD_EJECTION_DISTANCE]);
	printf("dc_threshold[THRESHOLD_RADII_ENHANCE_FACTOR        ] : %lf\n", dc_threshold[THRESHOLD_RADII_ENHANCE_FACTOR]);
	printf("dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE_SQUARED] : %lf\n", dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE_SQUARED]);
	printf("dc_threshold[THRESHOLD_EJECTION_DISTANCE_SQUARED   ] : %lf\n", dc_threshold[THRESHOLD_EJECTION_DISTANCE_SQUARED]);
}

static __global__
	void kernel_print_analytic_gas_disk_params()
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
	void	kernel_calc_grav_accel_int_mul_of_thread_per_block
	(
		interaction_bound int_bound, 
		const param_t* p, 
		const vec_t* r, 
		vec_t* a
	)
{
	const int i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	vec_t dVec;
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
	void	kernel_calc_grav_accel
	(
		ttt_t t, 
		interaction_bound int_bound, 
		const body_metadata_t* body_md, 
		const param_t* p, 
		const vec_t* r, 
		const vec_t* v, 
		vec_t* a,
		event_data_t* events,
		int *event_counter
	)
{
	const int i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	if (i < int_bound.sink.y)
	{
		// This line (beyond my depth) speeds up the kernel
		a[i].x = a[i].y = a[i].z = a[i].w = 0.0;
		if (body_md[i].id > 0)
		{
			vec_t dVec = {0.0, 0.0, 0.0, 0.0};
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
					unsigned int k = atomicAdd(event_counter, 1);

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

__global__
	void kernel_print_array(int n, const var_t* v)
{
	printf("v: %p\n", v);

	for (int i = 0; i < n; i++)
	{
		printf("v[%4d] : %20.16lf\n", i, v[i]);
	}
}

__global__
	void kernel_print_array(int n, int k, var_t** v)
{
	printf("k: %2d\n", k);
	printf("v: %p\n", v);
	printf("v[%2d]: %p\n", k, v[k]);

	for (int i = 0; i < n; i++)
	{
		printf("v[%2d][%2d] : %20.16lf\n", k, i, v[k][i]);
	}
}

__global__
	void set_element_of_array(int n, int idx, var_t* v, var_t value)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n && idx == i)
	{
		v[idx] = value;
	}
}

__global__
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

__global__
	void kernel_print_position(int n, const vec_t* r)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
	{
		//printf("r[%4d]: %f\n", tid, r[tid]);
		printf("r[%4d].x: %f\n", tid, r[tid].x);
		printf("r[%4d].y: %f\n", tid, r[tid].y);
		printf("r[%4d].z: %f\n", tid, r[tid].z);
		printf("r[%4d].w: %f\n", tid, r[tid].w);
	}
}

static void test_print_position(int n, const vec_t* r)
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
	cudaMalloc(d_ptr, size);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("cudaMalloc failed");
	}
}


#if 0
// Study and test how a struct is placed into the memory
// Study and test how an array of struct is placed into the memory
int main(int argc, const char** argv)
{
	sim_data_t sim_data;

	sim_data.y.resize(2);
	sim_data.y[0] = new vec_t[8];
	sim_data.y[1] = new vec_t[8];

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
	cudaMalloc((void**) &(sim_data.d_y[0]), 8*sizeof(vec_t));
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		cerr << "cudaMalloc failed" << endl;
		return EXIT_FAILURE;
	}
	// Copy pointer content (position and mass) from host to device.
	cudaMemcpy(sim_data.d_y[0], sim_data.y[0], 8*sizeof(vec_t), cudaMemcpyHostToDevice);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		cerr << "cudaMemcpy failed" << endl;
		return EXIT_FAILURE;
	}
	
	kernel_print_position<<<1, 8>>>(8, sim_data.d_y[0]);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		cerr << "kernel_print_position failed" << endl;
		return EXIT_FAILURE;
	}
	cudaDeviceSynchronize();

	// Allocate pointer.
	vec_t*	v = 0;
	v = (vec_t*)malloc(8 * sizeof(vec_t));
	memset(v, 0, 8 * sizeof(vec_t));

	// Allocate device pointer.
	vec_t*	d_v = 0;
	cudaMalloc((void**) &(d_v), 8*sizeof(vec_t));
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		cerr << "cudaMalloc failed" << endl;
		return EXIT_FAILURE;
	}

	// Copy pointer content from host to device.
	cudaMemcpy(d_v, v, 8*sizeof(vec_t), cudaMemcpyHostToDevice);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		cerr << "cudaMemcpy failed" << endl;
		return EXIT_FAILURE;
	}

	kernel_print_vector<<<1, 8>>>(8, d_v);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		cerr << "kernel_print_vector failed" << endl;
		return EXIT_FAILURE;
	}
	cudaDeviceSynchronize();

	free(v);
	cudaFree(d_v);

	delete[] sim_data.y[0];
	delete[] sim_data.y[1];
	cudaFree(sim_data.d_y[0]);

	return EXIT_SUCCESS;
}
#endif

#if 0
// Study and test the vector<vector <vec_t*> > type
int main(int argc, const char** argv)
{
	vector<vector <vec_t*> >	d_f(8);		// size of the outer vector container

	for (int i = 0; i < 8; i++)
	{
		d_f[i].resize(2);					// size of the inner vector container
		for (int j = 0; j < 2; j++)
		{
			d_f[i][j] = new vec_t[4];		// allocate 4 vec_t type element for each i, j pair
		}
	}
}
#endif

#if 0
// Study how to wrap a vec_t* into thrust vector to find the maximal element
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

static __global__
	void kernel_print_memory_address(int n, var_t* adr)
{
	for (int i = 0; i < n; i++)
	{
		printf("adr[%2d]: %p\n", i, adr[i]);
	}
}

static __global__
	void kernel_populate(int n, var_t value, var_t* dst)
{
	for (int i = 0; i < n; i++)
	{
		dst[i] = value;
	}
}

int main()
{
	vector<vector <vec_t*> >	d_f(2);
	vec_t** d_dydt;

	const int n_total = 2;
	const int r_max = 3;

	// ALLOCATION
	ALLOCATE_DEVICE_VECTOR((void**)&d_dydt, 2*r_max*sizeof(vec_t*));
	for (int i = 0; i < 2; i++)
	{
		d_f[i].resize(r_max);
		for (int r = 0; r < r_max; r++) 
		{
			ALLOCATE_DEVICE_VECTOR((void**) &(d_f[i][r]), n_total*sizeof(vec_t));
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
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("kernel_print_memory_address failed");
	}
	cudaDeviceSynchronize();
	// PRINT ALLOCATED ADDRESSES END

	// POPULATE ALLOCATED STORAGE
	for (int i = 0; i < 2; i++)
	{
		for (int r = 0; r < r_max; r++) 
		{
			kernel_populate<<<1, 1>>>(n_total, pow(-1.0, i) * r, (var_t*)d_f[i][r]);
		}
	}	
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("kernel_populate failed");
	}
	cudaDeviceSynchronize();
	// POPULATE ALLOCATED STORAGE END

	// PRINT DATA
	printf("d_f[][]:\n");
	for (int i = 0; i < 2; i++)
	{
		for (int r = 0; r < r_max; r++) 
		{
			kernel_print_array<<<1, 1>>>(n_total, (var_t*)d_f[i][r]);
			cudaStatus = HANDLE_ERROR(cudaGetLastError());
			if (cudaSuccess != cudaStatus)
			{
				throw string("kernel_print_array failed");
			}
			cudaDeviceSynchronize();
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
			cudaStatus = HANDLE_ERROR(cudaGetLastError());
			if (cudaSuccess != cudaStatus)
			{
				throw string("kernel_print_array failed");
			}
			cudaDeviceSynchronize();
		}
	}	
	// PRINT DATA END

	cudaDeviceReset();
}

#endif

#if 0

void cpy_cnstnt_to_dvc(const void* dst, const void *src, size_t count)
{
	cudaMemcpyToSymbol(dst, src, count);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("cudaMemcpyToSymbol failed (copy_constant_to_device)");
	}
}

int main(int argc, const char** argv)
{
	cudaError_t cudaStatus = cudaSuccess;

	int id_active = -1;
	int n_device = 0;

	cudaStatus = cudaGetDeviceCount(&n_device);
	if (cudaSuccess != cudaStatus)
	{
		printf("Error: %s\n", cudaGetErrorString(cudaStatus));
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
        cudaStatus = cudaSetDevice(dev);
		if (cudaSuccess != cudaStatus)
		{
			printf("Error: %s\n", cudaGetErrorString(cudaStatus));
			exit(0);
		}
		cudaStatus = cudaGetDevice(&id_active);
		if (cudaSuccess != cudaStatus)
		{
			printf("Error: %s\n", cudaGetErrorString(cudaStatus));
			exit(0);
		}
		printf("The id of the active device: %2d\n", id_active);

        cudaDeviceProp deviceProp;
        cudaStatus = cudaGetDeviceProperties(&deviceProp, dev);
		if (cudaSuccess != cudaStatus)
		{
			printf("Error: %s\n", cudaGetErrorString(cudaStatus));
			exit(0);
		}

        cout << "The code runs on " << deviceProp.name << " device:" << endl;
		//[domain]:[bus]:[device].
		cout << "The PCI domain ID of the device : " << deviceProp.pciDomainID << endl;
		cout << "The PCI bus ID of the device    : " << deviceProp.pciBusID << endl;
		cout << "The PCI device ID of the device : " << deviceProp.pciDeviceID << endl;

		char pciBusId[255];
		cudaStatus = cudaDeviceGetPCIBusId(pciBusId, 255, dev);
		if (cudaSuccess != cudaStatus)
		{
			printf("Error: %s\n", cudaGetErrorString(cudaStatus));
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

	vec_t* ptr = 0x0;
	size_t size = 1024 * sizeof(vec_t);
	bool cpu = true;

	ALLOCATE_VECTOR((void **)&(ptr), size, cpu);

	FREE_VECTOR((void **)&(ptr), cpu);

	return 0;
}
#endif

#if 0
int main(int argc, char** argv)
{
	red_test::run(argc, argv);

	//string result = file::combine_path("aa", "bb");
	//cout << result << endl;

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
int calc_linear_index(vec_t& rVec, var_t* used_rad)
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
		for (unsigned int i = 0; i < result.length(); i++)
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
			vec_t rVec = {r * cos(theta), r * sin(theta), 0.0, 0.0};
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
			vec_t rVec = {r * cos(theta), r * sin(theta), 0.0, 0.0};
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

#if 0 // Implement the tile-calculation method for the N-body gravity kernel

__host__ __device__
vec_t body_body_interaction(vec_t riVec, vec_t rjVec, var_t mj, vec_t aiVec)
{
	vec_t dVec = {0.0, 0.0, 0.0, 0.0};

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


__device__
vec_t tile_calculation(vec_t my_pos, var_t* mass, vec_t accel)
{
	int i;
	extern __shared__ vec_t[] sh_pos;

	for (i = 0; i < blockDim.x; i++)
	{
		accel = body_body_interaction(my_pos, sh_pos[i], mass[i], accel);
	}
}

__global__
void kernel_calculate_acceleration(void* dev_x, var_t* mass, void* dev_a)
{
	extern __shared__ vec_t[] sh_pos;

	vec_t* global_x = (vec_t*)dev_x;
	vec_t* global_a = (vec_t*)dev_a;

	vec_t my_pos = {0.0, 0.0, 0.0, 0.0};
	vec_t acc    = {0.0, 0.0, 0.0, 0.0};

	int i    = 0;
	int tile = 0;

	int gtid = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = 0, tile = 0; i < N; i += p, tile++)
	{
		int idx = tile * blockDim.x + threadIdx.x;
		sh_pos[threadIdx.x] = global_x[idx];
		__syncthreads();
		acc = tile_calculation(my_pos, mass, acc);
		__syncthreads();
	}

	global_a[gtid] = acc;
}

int main(int argc, char** argv)
{
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
 * A megoldas az volt, hogy peldanyositani kellett a k�l�nb�zo tipusokkal a template fuggvenyt.
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


#if 1
/*
 *  Howto write/read data in binary representation to/from a file
 */

int main()
{
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
	cout << "          sizeof(vec_t): " << sizeof(vec_t) << endl;
	cout << "        sizeof(param_t): " << sizeof(param_t) << endl;
	cout << "sizeof(body_metadata_t): " << sizeof(body_metadata_t) << endl;

	unsigned int n_bodies[] = {1, 2, 3, 4, 5, 6, 7};
	char name_buffer[30];
	memset(name_buffer, 0, sizeof(name_buffer));
	cout << "sizeof(name_buffer): " << sizeof(name_buffer) << endl;


	const int n_total = 2;
	sim_data_t *sim_data = new sim_data_t();
	// These will be only aliases to the actual storage space either in the HOST or DEVICE memory
	sim_data->y.resize(2);
	sim_data->yout.resize(2);
	allocate_host_storage(sim_data, n_total);

	vec_t* r = sim_data->h_y[0];
	vec_t* v = sim_data->h_y[1];
	param_t* p = sim_data->h_p;
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
	ofstream sout (path.c_str(), ios::out | ios::binary);
	if (sout)
	{
		for (unsigned int type = 0; type < BODY_TYPE_N; type++)
		{
			if (BODY_TYPE_PADDINGPARTICLE == type)
			{
				continue;
			}
			sout.write((char*)&n_bodies[type], sizeof(n_bodies[type]));
		}
		for (int i = 0; i < n_total; i++)
		{
			int orig_idx = bmd[i].id - 1;
			memset(name_buffer, 0, sizeof(name_buffer));
			strcpy(name_buffer, body_names[orig_idx].c_str());

			sout.write((char*)&epoch[i], sizeof(ttt_t));
			sout.write(name_buffer,      sizeof(name_buffer));
			sout.write((char*)&bmd[i],   sizeof(body_metadata_t));
			sout.write((char*)&p[i],     sizeof(param_t));
			sout.write((char*)&r[i],     sizeof(vec_t));
			sout.write((char*)&v[i],     sizeof(vec_t));
		}

		sout.close();
	}
	else
	{
		cerr << "Could not open " << path << "." << endl;
	}

	// Clear all data
	for (char type = 0; type < BODY_TYPE_N; type++)
	{
		if (BODY_TYPE_PADDINGPARTICLE == type)
		{
			continue;
		}
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
		// Read back the data
		for (unsigned int type = 0; type < BODY_TYPE_N; type++)
		{
			if (BODY_TYPE_PADDINGPARTICLE == type)
			{
				continue;
			}
			sin.read((char*)&n_bodies[type], sizeof(n_bodies[type]));
		}
		// Print the data
		for (char type = 0; type < BODY_TYPE_N; type++)
		{
			if (BODY_TYPE_PADDINGPARTICLE == type)
			{
				continue;
			}
			cout << n_bodies[type] << endl;
		}

		for (unsigned int i = 0; i < n_total; i++)
		{
			memset(name_buffer, 0, sizeof(name_buffer));

			sin.read((char*)&epoch[i],  1*sizeof(ttt_t));
			sin.read(name_buffer,      30*sizeof(char));
			sin.read((char*)&bmd[i],    1*sizeof(body_metadata_t));
			sin.read((char*)&p[i],      1*sizeof(param_t));
			sin.read((char*)&r[i],      1*sizeof(vec_t));
			sin.read((char*)&v[i],      1*sizeof(vec_t));

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
		cerr << "Could not open " << path.c_str() << "." << endl;
	}
}

#endif
