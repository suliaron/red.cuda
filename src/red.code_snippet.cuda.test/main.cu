// includes system
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <string>

// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes Thrust

// includes project
#include "red_type.h"
#include "red_macro.h"


using namespace std;

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
		printf("r[%4d]: %f\n", tid, r[tid]);
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
		printf("r[%4d]: %f\n", tid, r[tid]);
		printf("r[%4d].x: %f\n", tid, r[tid].x);
		printf("r[%4d].y: %f\n", tid, r[tid].y);
		printf("r[%4d].z: %f\n", tid, r[tid].z);
		printf("r[%4d].w: %f\n", tid, r[tid].w);
	}
}

#if 0
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