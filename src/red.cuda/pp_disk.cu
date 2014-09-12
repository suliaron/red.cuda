// includes system
#include <iostream>
#include <string>
#include <fstream>

// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes project
#include "gas_disk.h"
#include "nbody_exception.h"
#include "pp_disk.h"
#include "redutilcu.h"
#include "red_constants.h"
#include "red_macro.h"
#include "red_type.h"

using namespace std;
using namespace redutilcu;

/****************** DEVICE functions begins here ******************/

static __host__ __device__
	vec_t	vector_subtract(const vec_t* a, const vec_t* b)
{
	vec_t result;

	result.x = a->x - b->x;
    result.y = a->y - b->y;
    result.z = a->z - b->z;

	return result;
}

/****************** DEVICE functions ends   here ******************/


/****************** KERNEL functions begins here ******************/

static __global__
	void	kernel_calc_grav_accel(
										ttt_t t, 
										interaction_bound int_bound, 
										const body_metadata_t* body_md, 
										const param_t* p, 
										const vec_t* r, 
										const vec_t* v, 
										vec_t* a
										)
{
	const int bodyIdx = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	if (body_md[bodyIdx].id >= 0 && bodyIdx < int_bound.sink.y) {

		a[bodyIdx].x = 0.0;
		a[bodyIdx].y = 0.0;
		a[bodyIdx].z = 0.0;
		a[bodyIdx].w = 0.0;

		vec_t dVec = {0.0, 0.0, 0.0, 0.0};
		/* Skip the body with the same index and those which are inactive ie. id < 0 */
		for (int j = int_bound.source.x; j < int_bound.source.y; j++) 
		{
			if (j == bodyIdx || body_md[j].id < 0)
			{
				continue;
			}
			// 3 FLOP
			dVec.x = r[j].x - r[bodyIdx].x;
			dVec.y = r[j].y - r[bodyIdx].y;
			dVec.z = r[j].z - r[bodyIdx].z;

			// 5 FLOP
			dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2
			// TODO: use rsqrt()
			// 20 FLOP
			var_t r = sqrt(dVec.w);								// = r

			// 2 FLOP
			dVec.w = p[j].mass / (r*dVec.w);

			// 6 FLOP
			a[bodyIdx].x += dVec.w * dVec.x;
			a[bodyIdx].y += dVec.w * dVec.y;
			a[bodyIdx].z += dVec.w * dVec.z;
		}
		// 36 FLOP
		a[bodyIdx].x *= K2;
		a[bodyIdx].y *= K2;
		a[bodyIdx].z *= K2;
	}
}

static __global__
	void	kernel_transform_to(
								int n,
								int refBodyId,
								const param_t *p,
								vec_t* r, 
								vec_t* v
								)
{
	int	bodyIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (n > bodyIdx && refBodyId != bodyIdx)
	{
		vec_t rVec = vector_subtract((vec_t*)(&r[bodyIdx]), (vec_t*)(&r[refBodyId]));
		vec_t vVec = vector_subtract((vec_t*)(&v[bodyIdx]), (vec_t*)(&v[refBodyId]));
		r[bodyIdx].x = rVec.x;
		r[bodyIdx].y = rVec.y;
		r[bodyIdx].z = rVec.z;
		v[bodyIdx].x = vVec.x;
		v[bodyIdx].y = vVec.y;
		v[bodyIdx].z = vVec.z;
	}
}

static __global__
	void kernel_print_position(int n, const vec_t* r)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
	{
		printf("r[%d]: (%20.16lf, %20.16lf, %20.16lf, %20.16lf)\n", tid, r[tid].x, r[tid].y, r[tid].z, r[tid].w);
	}
}

static __global__
	void kernel_print_velocity(int n, const vec_t* v)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
	{
		printf("v[%d]: (%20.16lf, %20.16lf, %20.16lf, %20.16lf)\n", tid, v[tid].x, v[tid].y, v[tid].z, v[tid].w);
	}
}

static __global__
	void kernel_print_parameters(int n, const param_t* par)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		printf("par[%4d].mass    : %20.16lf\n", i, par[i].mass);
		printf("par[%4d].density : %20.16lf\n", i, par[i].density);
		printf("par[%4d].radius	 : %20.16lf\n", i, par[i].radius);
		printf("par[%4d].cd      : %20.16lf\n", i, par[i].cd);
	}
}

static __global__
	void kernel_print_body_metadata(int n, const body_metadata_t* body_md)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		printf("body_md[%4d].id          : %20.4d\n", i,   body_md[i].id);
		printf("body_md[%4d].body_type   : %20.4d\n", i,   body_md[i].body_type);
		printf("body_md[%4d].mig_type    : %20.4d\n", i,   body_md[i].mig_type);
		printf("body_md[%4d].mig_stop_at : %20.16lf\n", i, body_md[i].mig_stop_at);
	}
}

static __global__
	void kernel_print_epochs(int n, const ttt_t* epoch)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		printf("epoch[%4d] : %20.16lf\n", i, epoch[i]);
	}
}

static __global__
	void kernel_print_vector(int n, const vec_t* v)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
	{
		printf("[%d]: (%20.16lf, %20.16lf, %20.16lf, %20.16lf)\n", tid, v[tid].x, v[tid].y, v[tid].z, v[tid].w);
	}
}

/****************** KERNEL functions ends  here  ******************/

/****************** TEST functions begins here ********************/

void test_print_position(int n, const vec_t* r)
{
	for (int i = 0; i < n; i++)
	{
		printf("r[%4d].x : %20.16lf\n", i, r[i].x);
		printf("r[%4d].y : %20.16lf\n", i, r[i].y);
		printf("r[%4d].z : %20.16lf\n", i, r[i].z);
		printf("r[%4d].w : %20.16lf\n", i, r[i].w);
	}
}


// Test function: print out all the simulation data contained on the device
void pp_disk::test_call_kernel_print_sim_data()
{
	const int n = n_bodies->get_n_total();

	set_kernel_launch_param(n);

	kernel_print_position<<<grid, block>>>(n, sim_data->d_y[0]);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_print_position failed", cudaStatus);
	}
	cudaDeviceSynchronize();

	kernel_print_velocity<<<grid, block>>>(n, sim_data->d_y[1]);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_print_velocity failed", cudaStatus);
	}
	cudaDeviceSynchronize();

	kernel_print_parameters<<<grid, block>>>(n, sim_data->d_p);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_print_parameters failed", cudaStatus);
	}
	cudaDeviceSynchronize();

	kernel_print_body_metadata<<<grid, block>>>(n, sim_data->d_body_md);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_print_body_metadata failed", cudaStatus);
	}
	cudaDeviceSynchronize();

	kernel_print_epochs<<<grid, block>>>(n, sim_data->d_epoch);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_print_epochs failed", cudaStatus);
	}
	cudaDeviceSynchronize();
}

/****************** TEST   functions ends  here  ******************/

void pp_disk::set_kernel_launch_param(int n_data)
{
	int		n_thread = min(THREADS_PER_BLOCK, n_data);
	int		n_block = (n_data + n_thread - 1)/n_thread;

	grid.x	= n_block;
	block.x = n_thread;
}

void pp_disk::call_kernel_calc_grav_accel(ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy)
{
	cudaError_t cudaStatus = cudaSuccess;
	
	int nBodyTocalc = n_bodies->get_n_self_interacting();
	if (0 < nBodyTocalc) {
		interaction_bound int_bound = n_bodies->get_self_interacting();
		set_kernel_launch_param(nBodyTocalc);

		kernel_calc_grav_accel<<<grid, block>>>
			(curr_t, int_bound, sim_data->d_body_md, sim_data->d_p, r, v, dy);
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("kernel_calc_grav_accel failed", cudaStatus);
		}
	}
}

void pp_disk::calc_dy(int i, int rr, ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy)
{
	cudaError_t cudaStatus = cudaSuccess;
	const int n = n_bodies->get_n_total();

	switch (i)
	{
	case 0:
		// Copy velocities from previous step
		cudaMemcpy(dy, v, n * sizeof(vec_t), cudaMemcpyDeviceToDevice);
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("cudaMemcpy failed", cudaStatus);
		}
#if 0
		set_kernel_launch_param(n);
		printf("velocity:\n");
		kernel_print_vector<<<grid, block>>>(n, dy);
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("kernel_print_vector failed", cudaStatus);
		}
		cudaDeviceSynchronize();
#endif
		break;
	case 1:
		if (rr == 0)
		{
			// Set the d field of the event_data_t struct to the threshold distance when collision must be looked for
			// This is set to the radius of the star enhanced by 1 %.
		}
		// Calculate accelerations originated from the gravitational force
		call_kernel_calc_grav_accel(curr_t, r, v, dy);
#if 0
		set_kernel_launch_param(n);
		printf("acceleration:\n");
		kernel_print_vector<<<grid, block>>>(n, dy);
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("kernel_print_vector failed", cudaStatus);
		}
		cudaDeviceSynchronize();
#endif
		break;
	}
}

void pp_disk::call_kernel_transform_to(int refBodyId)
{
	cudaError_t cudaStatus = cudaSuccess;
	int	nBodyToCalc = n_bodies->get_n_total();

	set_kernel_launch_param(nBodyToCalc);
	
	kernel_transform_to<<<grid, block>>>(nBodyToCalc, refBodyId, sim_data->d_p, sim_data->d_y[0], sim_data->d_y[1]);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_transform_to failed", cudaStatus);
	}
}

pp_disk::pp_disk(string& path, gas_disk *gd) :
	g_disk(gd),
	d_g_disk(0),
	t(0.0),
	sim_data(0),
	n_bodies(0)
{
	n_bodies = get_number_of_bodies(path);
	allocate_storage();
	load(path);
}

pp_disk::~pp_disk()
{
	for (int i = 0; i < 2; i++)
	{
		delete[] sim_data->y[i];
	}
	delete[] sim_data->p;
	delete[] sim_data->body_md;
	delete[] sim_data->epoch;

	for (int i = 0; i < 2; i++)
	{
		cudaFree(sim_data->d_y[i]);
		cudaFree(sim_data->d_yout[i]);
	}
	cudaFree(sim_data->d_p);
	cudaFree(sim_data->d_body_md);
	cudaFree(sim_data->d_epoch);

	delete sim_data;
}

number_of_bodies* pp_disk::get_number_of_bodies(string& path)
{
	int ns, ngp, nrp, npp, nspl, npl, ntp;
	ns = ngp = nrp = npp = nspl = npl = ntp = 0;

	ifstream input(path.c_str());
	if (input) 
	{
		input >> ns >> ngp >> nrp >> npp >> nspl >> npl >> ntp;
		input.close();
	}
	else 
	{
		throw string("Cannot open " + path + ".");
	}
	return new number_of_bodies(ns, ngp, nrp, npp, nspl, npl, ntp);
}

void pp_disk::allocate_storage()
{
	const int n = n_bodies->total;

	sim_data = new sim_data_t;

	sim_data->y.resize(2);
	for (int i = 0; i < 2; i++)
	{
		sim_data->y[i]	= new vec_t[n];
	}
	sim_data->p	= new param_t[n];
	sim_data->body_md	= new body_metadata_t[n];
	sim_data->epoch		= new ttt_t[n];

	sim_data->d_y.resize(2);
	sim_data->d_yout.resize(2);
	// Allocate device pointer.
	for (int i = 0; i < 2; i++)
	{
		allocate_device_vector((void **)&(sim_data->d_y[i]),	n*sizeof(vec_t));
		allocate_device_vector((void **)&(sim_data->d_yout[i]),	n*sizeof(vec_t));
	}
	allocate_device_vector((void **)&(sim_data->d_p),			n*sizeof(param_t));
	allocate_device_vector((void **)&(sim_data->d_body_md),		n*sizeof(body_metadata_t));
	allocate_device_vector((void **)&(sim_data->d_epoch),		n*sizeof(ttt_t));
}

void pp_disk::copy_to_device()
{
	const int n = n_bodies->total;

	for (int i = 0; i < 2; i++)
	{
		copy_vector_to_device((void *)sim_data->d_y[i],	(void *)sim_data->y[i],		n*sizeof(vec_t));
	}
	copy_vector_to_device((void *)sim_data->d_p,		(void *)sim_data->p,		n*sizeof(param_t));
	copy_vector_to_device((void *)sim_data->d_body_md,	(void *)sim_data->body_md,	n*sizeof(body_metadata_t));
	copy_vector_to_device((void *)sim_data->d_epoch,	(void *)sim_data->epoch,	n*sizeof(ttt_t));
}

void pp_disk::copy_to_host()
{
	const int n = n_bodies->total;

	for (int i = 0; i < 2; i++)
	{
		copy_vector_to_host((void *)sim_data->y[i],		(void *)sim_data->d_y[i],	n*sizeof(vec_t));
	}
	copy_vector_to_host((void *)sim_data->p,			(void *)sim_data->d_p,		n*sizeof(param_t));
	copy_vector_to_host((void *)sim_data->body_md,		(void *)sim_data->d_body_md,n*sizeof(body_metadata_t));
	copy_vector_to_host((void *)sim_data->epoch,		(void *)sim_data->d_epoch,	n*sizeof(ttt_t));
}

void pp_disk::copy_variables_to_host()
{
	const int n = n_bodies->total;

	for (int i = 0; i < 2; i++)
	{
		copy_vector_to_host((void *)sim_data->y[i],		(void *)sim_data->d_y[i],	n*sizeof(vec_t));
	}
}

void pp_disk::allocate_device_vector(void **d_ptr, size_t size)
{
	cudaMalloc(d_ptr, size);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw nbody_exception("cudaMalloc failed", cudaStatus);
	}
}

void pp_disk::copy_vector_to_device(void* dst, const void *src, size_t count)
{
	cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw nbody_exception("cudaMemcpy failed (copy_vector_to_device)", cudaStatus);
	}
}

void pp_disk::copy_vector_to_host(void* dst, const void *src, size_t count)
{
	cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw nbody_exception("cudaMemcpy failed (copy_vector_to_host)", cudaStatus);
	}
}

var_t pp_disk::get_mass_of_star()
{
	body_metadata_t* body_md = sim_data->body_md;
	for (int j = 0; j < n_bodies->get_n_massive(); j++ ) {
		if (body_md[j].body_type == BODY_TYPE_STAR)
		{
			return sim_data->p[j].mass;
		}
	}
	throw string("No star is included!");
}

var_t pp_disk::get_total_mass()
{
	var_t totalMass = 0.0;

	param_t* p = sim_data->p;
	for (int j = 0; j < n_bodies->get_n_massive(); j++ ) {
		totalMass += p[j].mass;
	}

	return totalMass;
}

void pp_disk::compute_bc(vec_t* R0, vec_t* V0)
{
	const param_t* p = sim_data->p;
	const vec_t* r = sim_data->y[0];
	const vec_t* v = sim_data->y[1];

	for (int j = 0; j < n_bodies->get_n_massive(); j++ ) {
		var_t m = p[j].mass;
		R0->x += m * r[j].x;
		R0->y += m * r[j].y;
		R0->z += m * r[j].z;

		V0->x += m * v[j].x;
		V0->y += m * v[j].y;
		V0->z += m * v[j].z;
	}
	var_t M0 = get_total_mass();

	R0->x /= M0;	R0->y /= M0;	R0->z /= M0;
	V0->x /= M0;	V0->y /= M0;	V0->z /= M0;
}

void pp_disk::transform_to_bc()
{
	cout << "Transforming to barycentric system ... ";

	// Position and velocity of the system's barycenter
	vec_t R0 = {0.0, 0.0, 0.0, 0.0};
	vec_t V0 = {0.0, 0.0, 0.0, 0.0};

	compute_bc(&R0, &V0);

	vec_t* r = sim_data->y[0];
	vec_t* v = sim_data->y[1];
	// Transform the bodies coordinates and velocities
	for (int j = 0; j < n_bodies->get_n_total(); j++ ) {
		r[j].x -= R0.x;		r[j].y -= R0.y;		r[j].z -= R0.z;
		v[j].x -= V0.x;		v[j].y -= V0.y;		v[j].z -= V0.z;
	}

	cout << "done" << endl;
}

void pp_disk::load(string& path)
{
	cout << "Loading " << path << " ... ";

	ifstream input(path.c_str());
	if (input) 
	{
		int ns, ngp, nrp, npp, nspl, npl, ntp;
		input >> ns >> ngp >> nrp >> npp >> nspl >> npl >> ntp;
	}
	else 
	{
		throw string("Cannot open " + path + ".");
	}

	vec_t* r = sim_data->y[0];
	vec_t* v = sim_data->y[1];
	param_t* p = sim_data->p;
	body_metadata_t* body_md = sim_data->body_md;
	ttt_t* epoch = sim_data->epoch;

	if (input) {
		int_t	type = 0;
		string	dummy;
        		
		for (int i = 0; i < n_bodies->total; i++) { 
			// id
			input >> body_md[i].id;
			// name
			input >> dummy;
			body_names.push_back(dummy);
			// body type
			input >> type;
			body_md[i].body_type = static_cast<body_type_t>(type);
			// epoch
			input >> epoch[i];

			// mass
			input >> p[i].mass;
			// radius
			input >> p[i].radius;
			// density
			input >> p[i].density;
			// stokes constant
			input >> p[i].cd;

			// migration type
			input >> type;
			body_md[i].mig_type = static_cast<migration_type_t>(type);
			// migration stop at
			input >> body_md[i].mig_stop_at;

			// position
			input >> r[i].x;
			input >> r[i].y;
			input >> r[i].z;
			r[i].w = 0.0;
			// velocity
			input >> v[i].x;
			input >> v[i].y;
			input >> v[i].z;
			v[i].w = 0.0;
		}
        input.close();
	}
	else {
		throw string("Cannot open " + path + ".");
	}

	cout << "done" << endl;
}

void pp_disk::print_result(ostream& sout)
{
	vec_t* r = sim_data->y[0];
	vec_t* v = sim_data->y[1];
	param_t* p = sim_data->p;
	body_metadata_t* body_md = sim_data->body_md;
	ttt_t* epoch = sim_data->epoch;

	for (int i = 0; i < n_bodies->total; i++) {
		sout << body_md[i].id << SEP
			 << body_names[i] << SEP
			 << body_md[i].body_type << SEP 
			 << epoch[i] << SEP
			 << p[i].mass << SEP
			 << p[i].radius << SEP
			 << p[i].density << SEP
			 << p[i].cd << SEP
			 << body_md[i].mig_type << SEP
			 << body_md[i].mig_stop_at << SEP
			 << r[i].x << SEP
			 << r[i].y << SEP
			 << r[i].z << SEP
			 << v[i].x << SEP
			 << v[i].y << SEP
			 << v[i].z << endl;
    }
	sout.flush();
}
