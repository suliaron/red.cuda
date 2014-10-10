// includes system
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdlib.h>
#include <string>

// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes Thrust
#include "thrust\device_ptr.h"
#include "thrust\fill.h"
#include "thrust\extrema.h"

// includes project
#include "number_of_bodies.h"
#include "red_type.h"
#include "red_macro.h"
#include "util.h"


using namespace std;

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
		a[i].x = 0.0;
		a[i].y = 0.0;
		a[i].z = 0.0;
		a[i].w = 0.0;
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
				if (i > 0 && i > j && d < /* dc_threshold[THRESHOLD_COLLISION_FACTOR] */ 5.0 * (p[i].radius + p[j].radius))
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
					events[k].id.x = body_md[survivIdx].id;
					events[k].id.y = body_md[mergerIdx].id;
					events[k].idx.x = survivIdx;
					events[k].idx.y = mergerIdx;
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
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		printf("v[%4d] : %20.16lf\n", i, v[i]);
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

void test_print_position(int n, const vec_t* r)
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

// Measure the execution time of the kernel computing the gravitational acceleleration

ttt_t t;
//interaction_bound int_bound;
body_metadata_t* body_md;
param_t* p;
vec_t* r;
vec_t* v;
vec_t* a;
event_data_t* events;
event_data_t* d_events;
int *event_counter;
int *d_event_counter;

void allocate_storage(const number_of_bodies *n_bodies, sim_data_t *sim_data)
{
	const int nBody = n_bodies->total;

	sim_data = new sim_data_t;

	sim_data->y.resize(2);
	for (int i = 0; i < 2; i++)
	{
		sim_data->y[i]	= new vec_t[nBody];
	}
	sim_data->p	= new param_t[nBody];
	sim_data->body_md	= new body_metadata_t[nBody];
	sim_data->epoch		= new ttt_t[nBody];

	events = new event_data_t[nBody];

	sim_data->d_y.resize(2);
	sim_data->d_yout.resize(2);
	// Allocate device pointer.
	for (int i = 0; i < 2; i++)
	{
		ALLOCATE_DEVICE_VECTOR((void **)&(sim_data->d_y[i]),	nBody*sizeof(vec_t));
		ALLOCATE_DEVICE_VECTOR((void **)&(sim_data->d_yout[i]),	nBody*sizeof(vec_t));
	}
	ALLOCATE_DEVICE_VECTOR((void **)&(sim_data->d_p),			nBody*sizeof(param_t));
	ALLOCATE_DEVICE_VECTOR((void **)&(sim_data->d_body_md),		nBody*sizeof(body_metadata_t));
	ALLOCATE_DEVICE_VECTOR((void **)&(sim_data->d_epoch),		nBody*sizeof(ttt_t));

	ALLOCATE_DEVICE_VECTOR((void **)&d_events,					nBody*sizeof(event_data_t));
	ALLOCATE_DEVICE_VECTOR((void **)&d_event_counter,				1*sizeof(int));
}

void copy_to_device(const number_of_bodies *n_bodies, const sim_data_t *sim_data)
{
	const int n = n_bodies->total;

	for (int i = 0; i < 2; i++)
	{
		copy_vector_to_device((void *)sim_data->d_y[i],	(void *)sim_data->y[i],		n*sizeof(vec_t));
	}
	copy_vector_to_device((void *)sim_data->d_p,		(void *)sim_data->p,		n*sizeof(param_t));
	copy_vector_to_device((void *)sim_data->d_body_md,	(void *)sim_data->body_md,	n*sizeof(body_metadata_t));
	copy_vector_to_device((void *)sim_data->d_epoch,	(void *)sim_data->epoch,	n*sizeof(ttt_t));
	copy_vector_to_device((void *)d_event_counter,		(void *)&event_counter,		1*sizeof(int));
}

void populate_data(const number_of_bodies *n_bodies, sim_data_t *sim_data)
{
}

void deallocate_storage(sim_data_t *sim_data)
{
	for (int i = 0; i < 2; i++)
	{
		delete[] sim_data->y[i];
	}
	delete[] sim_data->p;
	delete[] sim_data->body_md;
	delete[] sim_data->epoch;
	delete[] events;

	for (int i = 0; i < 2; i++)
	{
		cudaFree(sim_data->d_y[i]);
		cudaFree(sim_data->d_yout[i]);
	}
	cudaFree(sim_data->d_p);
	cudaFree(sim_data->d_body_md);
	cudaFree(sim_data->d_epoch);
	cudaFree(d_events);
	cudaFree(d_event_counter);

	delete sim_data;
}

int main(int argc, const char** argv)
{
	number_of_bodies n_bodies = number_of_bodies(1, 0, 0, 5000, 0, 0, 5000);

	sim_data_t *sim_data = 0x0;
	t = 0.0;
	*event_counter = 0;

	allocate_storage(&n_bodies, sim_data);
	populate_data(&n_bodies, sim_data);
	copy_to_device(&n_bodies, sim_data);



	deallocate_storage(sim_data);
}
