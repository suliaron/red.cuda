// includes system
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <iostream>
#include <fstream>
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

// Measure the execution time of the kernel computing the gravitational acceleleration

class number_of_bodies
{
public:
	number_of_bodies(int n_s, int n_gp, int n_rp, int n_pp, int n_spl, int n_pl, int n_tp, int n_tpb, bool ups);

	void update_numbers(body_metadata_t *body_md);

	int	get_n_total();
	//! Calculates the number of bodies with mass, i.e. sum of the number of stars, giant planets, 
	/*  rocky planets, protoplanets, super-planetesimals and planetesimals.
	*/
	int	get_n_massive();
	//! Calculates the number of bodies which are self-interacting (i.e. it returns n_s + n_gp + n_rp + n_pp)
	int	get_n_SI();
	//! Calculates the number of non-self-interacting bodies (i.e. it returns n_spl + n_pl)
	int	get_n_NSI();
	//! Calculates the number of non-interacting bodies (i.e. returns n_tp)
	int	get_n_NI();
	//! Calculates the number of bodies which feels the drag force, i.e. sum of the number of super-planetesimals and planetesimals.
	int	get_n_GD();
	//! Calculates the number of bodies which are experiencing type I migration, i.e. sum of the number of rocky- and proto-planets.
	int	get_n_MT1();
	//! Calculates the number of bodies which are experiencing type II migration, i.e. the number of giant planets.
	int	get_n_MT2();

	int get_n_prime_total();
	int	get_n_prime_massive();

	int get_n_prime_SI();
	int get_n_prime_NSI();
	int get_n_prime_NI();

	//int	get_n_prime_GD();
	//int	get_n_prime_MT1();
	//int	get_n_prime_MT2();

	interaction_bound get_bound_SI();
	interaction_bound get_bound_NSI();
	interaction_bound get_bound_NI();
	interaction_bound get_bound_GD();
	interaction_bound get_bound_MT1();
	interaction_bound get_bound_MT2();

	int	n_s;
	int	n_gp;
	int	n_rp;
	int	n_pp;
	int	n_spl;
	int	n_pl;
	int	n_tp;

private:
	int n_tpb;
	bool ups;

	int2_t sink;
	int2_t source;
};

number_of_bodies::number_of_bodies(int n_s, int n_gp, int n_rp, int n_pp, int n_spl, int n_pl, int n_tp, int n_tpb, bool ups) :
		n_s(n_s),
		n_gp(n_gp),
		n_rp(n_rp),
		n_pp(n_pp),
		n_spl(n_spl),
		n_pl(n_pl),
		n_tp(n_tp),
		n_tpb(n_tpb),
		ups(ups)
{
    sink.x   = sink.y   = 0;
    source.x = source.y = 0;
}

void number_of_bodies::update_numbers(body_metadata_t *body_md)
{
	int n_active_body		= 0;
	int n_inactive_body		= 0;

	int	star				= 0;
	int	giant_planet		= 0;
	int	rocky_planet		= 0;
	int	proto_planet		= 0;
	int	super_planetesimal	= 0;
	int	planetesimal		= 0;
	int	test_particle		= 0;

	int n_total = ups ? get_n_prime_total() : get_n_total();
	for (int i = 0; i < n_total; i++)
	{
		if (0 < body_md[i].id && BODY_TYPE_PADDINGPARTICLE > body_md[i].body_type)
		{
			n_active_body++;
		}
		// Count the inactive bodies by type
		else
		{
			n_inactive_body++;
			switch (body_md[i].body_type)
			{
			case BODY_TYPE_STAR:
				star++;
				break;
			case BODY_TYPE_GIANTPLANET:
				giant_planet++;
				break;
			case BODY_TYPE_ROCKYPLANET:
				rocky_planet++;
				break;
			case BODY_TYPE_PROTOPLANET:
				proto_planet++;
				break;
			case BODY_TYPE_SUPERPLANETESIMAL:
				super_planetesimal++;
				break;
			case BODY_TYPE_PLANETESIMAL:
				planetesimal++;
				break;
			case BODY_TYPE_TESTPARTICLE:
				test_particle++;
				break;
			default:
				throw string("Undefined body type!");
			}
		}
	}
	cout << "There are " << star << " inactive star" << endl;
	cout << "There are " << giant_planet << " inactive giant planet" << endl;
	cout << "There are " << rocky_planet << " inactive rocky planet" << endl;
	cout << "There are " << proto_planet << " inactive protoplanet" << endl;
	cout << "There are " << super_planetesimal << " inactive super planetesimal" << endl;
	cout << "There are " << planetesimal << " inactive planetesimal" << endl;
	cout << "There are " << test_particle << " inactive test particle" << endl;

	n_s		-= star;
	n_gp	-= giant_planet;
	n_rp	-= rocky_planet;
	n_pp	-= proto_planet;
	n_spl	-= super_planetesimal;
	n_pl	-= planetesimal;
	n_tp	-= test_particle;
}

int	number_of_bodies::get_n_SI()
{
	return (n_s + n_gp + n_rp + n_pp);
}

int number_of_bodies::get_n_NSI()
{
	return (n_spl + n_pl);
}

int	number_of_bodies::get_n_NI()
{
	return n_tp;
}

int	number_of_bodies::get_n_total()
{
	return (n_s + n_gp + n_rp + n_pp + n_spl + n_pl + n_tp);
}

int	number_of_bodies::get_n_GD()
{
	return (n_spl + n_pl);
}

int	number_of_bodies::get_n_MT1()
{
	return (n_rp + n_pp);
}

int	number_of_bodies::get_n_MT2()
{
	return n_gp;
}

int	number_of_bodies::get_n_massive()
{
	return (get_n_SI() + get_n_NSI());
}

int number_of_bodies::get_n_prime_SI()
{
	// The number of self-interacting (SI) bodies aligned to n_tbp
	return ((get_n_SI() + n_tpb - 1) / n_tpb) * n_tpb;
}

int number_of_bodies::get_n_prime_NSI()
{
	// The number of non-self-interacting (NSI) bodies aligned to n_tbp
	return ((get_n_NSI() + n_tpb - 1) / n_tpb) * n_tpb;
}

int number_of_bodies::get_n_prime_NI()
{
	// The number of non-interacting (NI) bodies aligned to n_tbp
	return ((get_n_NI() + n_tpb - 1) / n_tpb) * n_tpb;
}

int number_of_bodies::get_n_prime_total()
{
	return (get_n_prime_SI() + get_n_prime_NSI() + get_n_prime_NI());
}

int	number_of_bodies::get_n_prime_massive()
{
	return (get_n_prime_SI() + get_n_prime_NSI());
}

interaction_bound number_of_bodies::get_bound_SI()
{
	if (ups)
	{
		sink.x	 = 0, sink.y   = sink.x + get_n_prime_SI();
		source.x = 0, source.y = source.x + get_n_prime_massive();
	}
	else
	{
		sink.x   = 0, sink.y   = sink.x + get_n_SI();
		source.x = 0, source.y = source.x + get_n_massive();
	}

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_NSI()
{
	if (ups)
	{
		sink.x   = get_n_prime_SI(), sink.y   = sink.x + get_n_prime_NSI();
		source.x = 0,				 source.y = source.x + get_n_prime_SI();;
	}
	else
	{
		sink.x   = get_n_SI(), sink.y   = sink.x + get_n_NSI();
		source.x = 0,		   source.y = source.x + get_n_SI();
	}

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_NI()
{
	if (ups)
	{
		sink.x   = get_n_prime_massive(), sink.y   = sink.x + get_n_prime_NI();
		source.x = 0,				      source.y = source.x + get_n_prime_massive();
	}
	else
	{
		sink.x   = get_n_massive(), sink.y   = sink.x + get_n_NI();
		source.x = 0,   	        source.y = source.x + get_n_massive();
	}

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_GD()
{
	if (ups)
	{
		sink.x   = get_n_prime_SI(), sink.y   = sink.x + n_spl + n_pl;
		source.x = 0,		         source.y = 0;
	}
	else
	{
		sink.x   = get_n_SI(), sink.y   = sink.x + n_spl + n_pl;
		source.x = 0,		   source.y = 0;
	}

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_MT1()
{
	sink.x   = n_s + n_gp, sink.y   = sink.x + n_pp + n_rp;
	source.x = 0,	       source.y = source.x + 0;

	return interaction_bound(sink, source);
}

interaction_bound number_of_bodies::get_bound_MT2()
{
	sink.x   = n_s,   sink.y = sink.x + n_gp;
	source.x = 0, 	source.y = source.x + 0;

	return interaction_bound(sink, source);
}

ostream& operator<<(ostream& stream, const number_of_bodies* n_bodies)
{
	const char* body_type_name[] = 
	{
		"STAR",
		"GIANTPLANET",
		"ROCKYPLANET",
		"PROTOPLANET",
		"SUPERPLANETESIMAL",
		"PLANETESIMAL",
		"TESTPARTICLE",
	};

	stream << "Number of bodies:" << endl;
	stream << setw(20) << body_type_name[0] << ": " << n_bodies->n_s << endl;
	stream << setw(20) << body_type_name[1] << ": " << n_bodies->n_gp << endl;
	stream << setw(20) << body_type_name[2] << ": " << n_bodies->n_rp << endl;
	stream << setw(20) << body_type_name[3] << ": " << n_bodies->n_pp << endl;
	stream << setw(20) << body_type_name[4] << ": " << n_bodies->n_spl << endl;
	stream << setw(20) << body_type_name[5] << ": " << n_bodies->n_pl << endl;
	stream << setw(20) << body_type_name[6] << ": " << n_bodies->n_tp << endl;
		
	return stream;
}

// Draw a number from a given distribution
var_t generate_random(var_t xmin, var_t xmax, var_t p(var_t))
{
	var_t x;
	var_t y;

	do
	{
		x = xmin + (var_t)rand() / RAND_MAX * (xmax - xmin);
		y = (var_t)rand() / RAND_MAX;
	}
	while (y > p(x));

	return x;
}

var_t pdf_const(var_t x)
{
	return 1;
}


dim3	grid;
dim3	block;

ttt_t t;
body_metadata_t* body_md;
param_t* p;
vec_t* r;
vec_t* v;
vec_t* a;
event_data_t* events;
event_data_t* d_events;
int event_counter;
int *d_event_counter;

void allocate_storage(number_of_bodies *n_bodies, sim_data_t *sim_data)
{
	int nBody = n_bodies->get_n_total();

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
		allocate_device_vector((void **)&(sim_data->d_y[i]),	nBody*sizeof(vec_t));
		allocate_device_vector((void **)&(sim_data->d_yout[i]), nBody*sizeof(vec_t));
	}
	allocate_device_vector((void **)&(sim_data->d_p),			nBody*sizeof(param_t));
	allocate_device_vector((void **)&(sim_data->d_body_md),		nBody*sizeof(body_metadata_t));
	allocate_device_vector((void **)&(sim_data->d_epoch),		nBody*sizeof(ttt_t));

	allocate_device_vector((void **)&d_events,					nBody*sizeof(event_data_t));
	allocate_device_vector((void **)&d_event_counter,				1*sizeof(int));

	/* Total of 9 cudaMalloc */
}

void allocate_pinned_storage(number_of_bodies *n_bodies, sim_data_t *sim_data)
{
	int nBody = n_bodies->get_n_total();

	sim_data->y.resize(2);
	for (int i = 0; i < 2; i++)
	{
		cudaMallocHost((void **)&sim_data->y[i],	nBody*sizeof(vec_t));
	}
	cudaMallocHost((void **)&sim_data->p,			nBody*sizeof(param_t));
	cudaMallocHost((void **)&sim_data->body_md,		nBody*sizeof(body_metadata_t));
	cudaMallocHost((void **)&sim_data->epoch,		nBody*sizeof(ttt_t));

	cudaMallocHost((void **)&events,				nBody*sizeof(event_data_t));

	sim_data->d_y.resize(2);
	sim_data->d_yout.resize(2);
	// Allocate device pointer.
	for (int i = 0; i < 2; i++)
	{
		allocate_device_vector((void **)&(sim_data->d_y[i]),	nBody*sizeof(vec_t));
		allocate_device_vector((void **)&(sim_data->d_yout[i]), nBody*sizeof(vec_t));
	}
	allocate_device_vector((void **)&(sim_data->d_p),			nBody*sizeof(param_t));
	allocate_device_vector((void **)&(sim_data->d_body_md),		nBody*sizeof(body_metadata_t));
	allocate_device_vector((void **)&(sim_data->d_epoch),		nBody*sizeof(ttt_t));

	allocate_device_vector((void **)&d_events,					nBody*sizeof(event_data_t));
	allocate_device_vector((void **)&d_event_counter,				1*sizeof(int));

	/* Total of 9 cudaMalloc */
}

void copy_to_device(number_of_bodies *n_bodies, const sim_data_t *sim_data)
{
	int n = n_bodies->get_n_total();

	for (int i = 0; i < 2; i++)
	{
		copy_vector_to_device((void *)sim_data->d_y[i],	(void *)sim_data->y[i],		n*sizeof(vec_t));
	}
	copy_vector_to_device((void *)sim_data->d_p,		(void *)sim_data->p,		n*sizeof(param_t));
	copy_vector_to_device((void *)sim_data->d_body_md,	(void *)sim_data->body_md,	n*sizeof(body_metadata_t));
	copy_vector_to_device((void *)sim_data->d_epoch,	(void *)sim_data->epoch,	n*sizeof(ttt_t));
	copy_vector_to_device((void *)d_event_counter,		(void *)&event_counter,		1*sizeof(int));

	/* Total of 6 cudaMemcpy calls */
}

void populate_data(const number_of_bodies *n_bodies, sim_data_t *sim_data)
{
	int idx = 0;

	sim_data->body_md[idx].body_type = BODY_TYPE_STAR;
	sim_data->body_md[idx].id = idx + 1;

	sim_data->p[idx].mass = 1.0;
	sim_data->p[idx].radius = 1.0 * constants::SolarRadiusToAu;
	for (int j = 0; j < 2; j++)
	{
		sim_data->y[j][idx].x = 0.0;
		sim_data->y[j][idx].y = 0.0;
		sim_data->y[j][idx].z = 0.0;
	}

	for (int i = 0; i < n_bodies->n_pp; i++, idx++)
	{
		sim_data->body_md[idx].body_type = BODY_TYPE_ROCKYPLANET;
		sim_data->body_md[idx].id = idx+1;

		sim_data->p[idx].mass = generate_random(1.0, 10.0, pdf_const) * constants::EarthToSolar;
		sim_data->p[idx].radius = generate_random(4000.0, 8000.0, pdf_const) * constants::KilometerToAu ;
		for (int j = 0; j < 2; j++)
		{
			sim_data->y[j][idx].x = generate_random(-10.0, 10.0, pdf_const);
			sim_data->y[j][idx].y = generate_random(-10.0, 10.0, pdf_const);
			sim_data->y[j][idx].z = generate_random(-10.0, 10.0, pdf_const);
		}
	}

	for (int i = 0; i < n_bodies->n_tp; i++, idx++)
	{
		sim_data->body_md[idx].body_type = BODY_TYPE_TESTPARTICLE;
		sim_data->body_md[idx].id = idx+1;

		sim_data->p[idx].mass = 0.0;
		sim_data->p[idx].radius = 0.0;
		for (int j = 0; j < 2; j++)
		{
			sim_data->y[j][idx].x = generate_random(-10.0, 10.0, pdf_const);
			sim_data->y[j][idx].y = generate_random(-10.0, 10.0, pdf_const);
			sim_data->y[j][idx].z = generate_random(-10.0, 10.0, pdf_const);
		}
	}
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

	/* Total of 9 cudaFree */
}

void deallocate_pinned_storage(sim_data_t *sim_data)
{
	for (int i = 0; i < 2; i++)
	{
		cudaFreeHost(sim_data->y[i]);
	}
	cudaFreeHost(sim_data->p);
	cudaFreeHost(sim_data->body_md);
	cudaFreeHost(sim_data->epoch);
	cudaFreeHost(events);

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

	/* Total of 9 cudaFree */
}

void set_kernel_launch_param(int n_tpb, int n_data)
{
	int		n_thread = min(n_tpb, n_data);
	int		n_block = (n_data + n_thread - 1)/n_thread;

	grid.x	= n_block;
	block.x = n_thread;
}

void call_kernel_calc_grav_accel(ttt_t curr_t, number_of_bodies *n_bodies, sim_data_t *sim_data, const vec_t* r, const vec_t* v, vec_t* dy)
{
	cudaError_t cudaStatus = cudaSuccess;
	
	int n_sink = n_bodies->get_n_SI();
	if (0 < n_sink) {
		interaction_bound int_bound = n_bodies->get_bound_SI();

		for (int n_tpb = 16; n_tpb <= 512; n_tpb += 16)
		{
			set_kernel_launch_param(n_tpb, n_sink);
			kernel_calc_grav_accel<<<grid, block>>>
				(curr_t, int_bound, sim_data->d_body_md, sim_data->d_p, r, v, dy, d_events, d_event_counter);

			cudaDeviceSynchronize();

			cudaStatus = HANDLE_ERROR(cudaGetLastError());
			if (cudaSuccess != cudaStatus) {
				throw string("kernel_calc_grav_accel failed");
			}
		}
	}

	n_sink = n_bodies->n_tp;
	if (0 < n_sink) {
		interaction_bound int_bound = n_bodies->get_bound_NI();

		for (int n_tpb = 16; n_tpb <= 512; n_tpb += 16)
			{
			set_kernel_launch_param(n_tpb, n_sink);
			kernel_calc_grav_accel<<<grid, block>>>
				(curr_t, int_bound, sim_data->d_body_md, sim_data->d_p, r, v, dy, d_events, d_event_counter);

			cudaDeviceSynchronize();

			cudaStatus = HANDLE_ERROR(cudaGetLastError());
			if (cudaSuccess != cudaStatus) {
				throw string("kernel_calc_grav_accel failed");
			}
		}
	}
}

void call_kernel_calc_grav_accel_int_mul_of_thread_per_block(ttt_t curr_t, number_of_bodies *n_bodies, sim_data_t *sim_data, const vec_t* r, const vec_t* v, vec_t* dy)
{
	cudaError_t cudaStatus = cudaSuccess;
	
	int n_sink = n_bodies->get_n_SI();
	if (0 < n_sink) {
		interaction_bound int_bound = n_bodies->get_bound_SI();

		for (int n_tpb = 16; n_tpb <= 512; n_tpb += 16)
		{
			set_kernel_launch_param(n_tpb, n_sink);
			kernel_calc_grav_accel_int_mul_of_thread_per_block <<<grid, block>>> (int_bound, sim_data->d_p, r, dy);

			cudaDeviceSynchronize();

			cudaStatus = HANDLE_ERROR(cudaGetLastError());
			if (cudaSuccess != cudaStatus) {
				throw string("kernel_calc_grav_accel failed");
			}
		}
	}

	n_sink = n_bodies->n_tp;
	if (0 < n_sink) {
		interaction_bound int_bound = n_bodies->get_bound_NI();

		for (int n_tpb = 16; n_tpb <= 512; n_tpb += 16)
		{
			set_kernel_launch_param(n_tpb, n_sink);
			kernel_calc_grav_accel_int_mul_of_thread_per_block <<<grid, block>>> (int_bound, sim_data->d_p, r, dy);

			cudaDeviceSynchronize();

			cudaStatus = HANDLE_ERROR(cudaGetLastError());
			if (cudaSuccess != cudaStatus) {
				throw string("kernel_calc_grav_accel failed");
			}
		}
	}
}

// http://devblogs.nvidia.com/parallelforall/
int main(int argc, const char** argv)
{
	number_of_bodies n_bodies = number_of_bodies(1, 0, 0, 20 * THREADS_PER_BLOCK - 1, 0, 0, 20 * THREADS_PER_BLOCK, 64, false);
	sim_data_t *sim_data = new sim_data_t;

	t = 0.0;
	event_counter = 0;

	allocate_storage(&n_bodies, sim_data);
	populate_data(&n_bodies, sim_data);
	copy_to_device(&n_bodies, sim_data);

	call_kernel_calc_grav_accel(t, &n_bodies, sim_data, sim_data->d_y[0], sim_data->d_y[1], sim_data->d_yout[1]);

	call_kernel_calc_grav_accel_int_mul_of_thread_per_block(t, &n_bodies, sim_data, sim_data->d_y[0], sim_data->d_y[1], sim_data->d_yout[1]);

	deallocate_storage(sim_data);

	// Needed by nvprof.exe
	cudaDeviceReset();
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

#if 1
// Test how to read a binary file
#include<iterator>

int main(int argc, char** argv)
{
	string path = "C:\\Work\\Projects\\red.cuda\\TestRun\\InputTest\\Test_Fargo_Gas\\gasvtheta0.dat";
	size_t n = 512*1024;
	vector<var_t> data(n);
	var_t* raw_data = data.data();

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
		for (int i = 0; i < result.length(); i++)
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
			cout << r << endl;
		}
	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
	}

	return 0;
}
#endif
