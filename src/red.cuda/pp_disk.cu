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
	void	kernel_calculate_grav_accel(
										ttt_t t, 
										interaction_bound int_bound, 
										const body_metadata_t* body_md, 
										const param_t* params, 
										const posm_t* pos, 
										const velR_t* vel, 
										vec_t* acce
										)
{
	const int bodyIdx = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	if (body_md[bodyIdx].active && bodyIdx < int_bound.sink.y) {

		acce[bodyIdx].x = 0.0;
		acce[bodyIdx].y = 0.0;
		acce[bodyIdx].z = 0.0;

		vec_t ai = {0.0, 0.0, 0.0, 0.0};
		vec_t dVec;
		for (int j = int_bound.source.x; j < int_bound.source.y; j++) 
		{
			if (j == bodyIdx || !body_md[j].active)
			{
				continue;
			}
			// 3 FLOP
			dVec.x = pos[j].x - pos[bodyIdx].x;
			dVec.y = pos[j].y - pos[bodyIdx].y;
			dVec.z = pos[j].z - pos[bodyIdx].z;

			// 5 FLOP
			dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2
			// TODO: use rsqrt()
			// 20 FLOP
			var_t r = sqrt(dVec.w);								// = r

			// 2 FLOP
			dVec.w = pos[bodyIdx].m / (r*dVec.w);

			// 6 FLOP
			ai.x += dVec.w * dVec.x;
			ai.y += dVec.w * dVec.y;
			ai.z += dVec.w * dVec.z;
		}
		// 36 FLOP
		acce[bodyIdx].x = K2 * ai.x;
		acce[bodyIdx].y = K2 * ai.y;
		acce[bodyIdx].z = K2 * ai.z;
	}
}

static __global__
	void	kernel_transform_to(
								int n,
								int refBodyId,
								const param_t *params,
								posm_t* pos, 
								velR_t* vel
								)
{
	int	bodyIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (n > bodyIdx && refBodyId != bodyIdx)
	{
		vec_t rVec = vector_subtract((vec_t*)(&pos[bodyIdx]), (vec_t*)(&pos[refBodyId]));
		vec_t vVec = vector_subtract((vec_t*)(&vel[bodyIdx]), (vec_t*)(&vel[refBodyId]));
		pos[bodyIdx].x = rVec.x;
		pos[bodyIdx].y = rVec.y;
		pos[bodyIdx].z = rVec.z;
		vel[bodyIdx].x = vVec.x;
		vel[bodyIdx].y = vVec.y;
		vel[bodyIdx].z = vVec.z;
	}
}

__global__
	void kernel_print_position(int n, const posm_t* pos)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
	{
		printf("pos[%4d].x : %20.16lf\n", tid, pos[tid].x);
		printf("pos[%4d].y : %20.16lf\n", tid, pos[tid].y);
		printf("pos[%4d].z : %20.16lf\n", tid, pos[tid].z);
		printf("pos[%4d].m : %20.16lf\n", tid, pos[tid].m);
	}
}

__global__
	void kernel_print_velocity(int n, const velR_t* vel)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		printf("vel[%4d].x : %20.16lf\n", i, vel[i].x);
		printf("vel[%4d].y : %20.16lf\n", i, vel[i].y);
		printf("vel[%4d].z : %20.16lf\n", i, vel[i].z);
		printf("vel[%4d].R : %20.16lf\n", i, vel[i].R);
	}
}

__global__
	void kernel_print_parameters(int n, const param_t* par)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		printf("par[%4d].mass        : %20.16lf\n", i, par[i].mass);
		printf("par[%4d].density     : %20.16lf\n", i, par[i].density);
		printf("par[%4d].mig_stop_at : %20.16lf\n", i, par[i].mig_stop_at);
		printf("par[%4d].cd          : %20.16lf\n", i, par[i].cd);
	}
}

__global__
	void kernel_print_body_metadata(int n, const body_metadata_t* body_md)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		printf("par[%4d].id        : %10.4d\n", i, body_md[i].id);
		printf("par[%4d].active    : %10.4d\n", i, body_md[i].active);
		printf("par[%4d].body_type : %10.4d\n", i, body_md[i].body_type);
		printf("par[%4d].mig_type  : %10.4d\n", i, body_md[i].mig_type);
	}
}

__global__
	void kernel_print_epochs(int n, const ttt_t* epoch)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		printf("epoch[%4d] : %20.16lf\n", i, epoch[i]);
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

/****************** KERNEL functions ends  here  ******************/

/****************** TEST functions begins here ********************/

void test_print_position(int n, const posm_t* pos)
{
	for (int i = 0; i < n; i++)
	{
		printf("pos[%4d].x : %20.16lf\n", i, pos[i].x);
		printf("pos[%4d].y : %20.16lf\n", i, pos[i].y);
		printf("pos[%4d].z : %20.16lf\n", i, pos[i].z);
		printf("pos[%4d].m : %20.16lf\n", i, pos[i].m);
	}
}


// Test function: print out all the simulation data contained on the device
void pp_disk::test_call_kernel_print_sim_data()
{
	const int n = n_bodies->get_n_total();

	set_kernel_launch_param(n);

	kernel_print_position<<<grid, block>>>(n, sim_data->d_pos);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_print_position failed", cudaStatus);
	}
	cudaDeviceSynchronize();

	kernel_print_velocity<<<grid, block>>>(n, sim_data->d_vel);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_print_velocity failed", cudaStatus);
	}
	cudaDeviceSynchronize();

	kernel_print_parameters<<<grid, block>>>(n, sim_data->d_params);
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

void pp_disk::call_kernel_calculate_grav_accel(ttt_t curr_t, const posm_t* pos, const velR_t* vel, vec_t* dy)
{
	cudaError_t cudaStatus = cudaSuccess;
	int		nBodyToCalculate;
	
	nBodyToCalculate = n_bodies->get_n_self_interacting();
	if (0 < nBodyToCalculate) {
		interaction_bound int_bound = n_bodies->get_self_interacting();
		set_kernel_launch_param(nBodyToCalculate);

		kernel_calculate_grav_accel<<<grid, block>>>
			(curr_t, int_bound, sim_data->d_body_md, sim_data->d_params, pos, vel, dy);
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("kernel_calculate_grav_accel failed", cudaStatus);
		}
	}
}

void pp_disk::calculate_dy(int i, int r, ttt_t curr_t, const posm_t* pos, const velR_t* vel, vec_t* dy)
{
	cudaError_t cudaStatus = cudaSuccess;
	int n = n_bodies->get_n_total();

	switch (i)
	{
	case 0:
		// Copy velocities from previous step
		cudaMemcpy(dy, vel, n_bodies->get_n_total() * sizeof(vec_t), cudaMemcpyDeviceToDevice);
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("cudaMemcpy failed", cudaStatus);
		}
#if 0
		set_kernel_launch_param(n);
		kernel_print_vector<<<grid, block>>>(n, dy);
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("cudaMemcpy failed", cudaStatus);
		}
		cudaDeviceSynchronize();
#endif
		break;
	case 1:
		if (r == 0)
		{
			// Set the d field of the event_data_t struct to the threshold distance when collision must be looked for
			// This is set to the radius of the star enhanced by 1 %.
		}
		// Calculate accelerations originated from the gravitational force
		call_kernel_calculate_grav_accel(curr_t, pos, vel, dy);
#if 0
		set_kernel_launch_param(n);
		kernel_print_vector<<<grid, block>>>(n, dy);
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("cudaMemcpy failed", cudaStatus);
		}
		cudaDeviceSynchronize();
#endif
		break;
	}
}

void pp_disk::call_kernel_transform_to(int refBodyId)
{
	cudaError_t cudaStatus = cudaSuccess;
	int		nBodyToCalculate = n_bodies->get_n_total();

	set_kernel_launch_param(nBodyToCalculate);
	
	kernel_transform_to<<<grid, block>>>(nBodyToCalculate, refBodyId, sim_data->d_params, sim_data->d_pos, sim_data->d_vel);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_transform_to failed", cudaStatus);
	}
}

pp_disk::pp_disk(string& path, gas_disk *gd) :
	t(0.0),
	sim_data(0),
	n_bodies(0)
{
	get_number_of_bodies(path);

	cerr << n_bodies;

	allocate_storage();
	load(path);
	g_disk = gd;
}

pp_disk::~pp_disk()
{
	delete[] sim_data->pos;
	delete[] sim_data->vel;
	delete[] sim_data->params;
	delete[] sim_data->body_md;
	delete[] sim_data->epoch;

	cudaFree(sim_data->d_pos);
	cudaFree(sim_data->d_pos_out);
	cudaFree(sim_data->d_vel);
	cudaFree(sim_data->d_vel_out);
	cudaFree(sim_data->d_params);
	cudaFree(sim_data->d_body_md);
	cudaFree(sim_data->d_epoch);

	delete sim_data;
}

void pp_disk::get_number_of_bodies(string& path)
{
	ifstream input(path.c_str());
	if (input) 
	{
		int ns, ngp, nrp, npp, nspl, npl, ntp;
		ns = ngp = nrp = npp = nspl = npl = ntp = 0;
		input >> ns >> ngp >> nrp >> npp >> nspl >> npl >> ntp;
		n_bodies = new number_of_bodies(ns, ngp, nrp, npp, nspl, npl, ntp);
	}
	else 
	{
		throw string("Cannot open " + path + ".");
	}
	input.close();
}

void pp_disk::allocate_storage()
{
	const int n = n_bodies->total;

	sim_data = new sim_data_t;

	sim_data->pos = new posm_t[n];
	sim_data->vel = new velR_t[n];
	sim_data->params = new param_t[n];
	sim_data->body_md = new body_metadata_t[n];
	sim_data->epoch = new ttt_t[n];

	// Allocate device pointer.
	cudaMalloc((void**) &(sim_data->d_pos), n*sizeof(posm_t));
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMalloc failed", cudaStatus);
	}
	// Allocate device pointer.
	cudaMalloc((void**) &(sim_data->d_pos_out), n*sizeof(posm_t));
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMalloc failed", cudaStatus);
	}
	// Allocate device pointer.
	cudaMalloc((void**) &(sim_data->d_vel), n*sizeof(velR_t));
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMalloc failed", cudaStatus);
	}
	// Allocate device pointer.
	cudaMalloc((void**) &(sim_data->d_vel_out), n*sizeof(velR_t));
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMalloc failed", cudaStatus);
	}
	// Allocate device pointer.
	cudaMalloc((void**) &(sim_data->d_params), n*sizeof(param_t));
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMalloc failed", cudaStatus);
	}
	// Allocate device pointer.
	cudaMalloc((void**) &(sim_data->d_body_md), n*sizeof(body_metadata_t));
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMalloc failed", cudaStatus);
	}
	// Allocate device pointer.
	cudaMalloc((void**) &(sim_data->d_epoch), n*sizeof(ttt_t));
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMalloc failed", cudaStatus);
	}
}

void pp_disk::copy_to_device()
{
	const int n = n_bodies->total;

	// Copy pointer content (position and mass) from host to device.
	cudaMemcpy(sim_data->d_pos, sim_data->pos, n*sizeof(posm_t), cudaMemcpyHostToDevice);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}
	// Copy pointer content (velocity and radius) from host to device.
	cudaMemcpy(sim_data->d_vel, sim_data->vel, n*sizeof(velR_t), cudaMemcpyHostToDevice);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}
	// Copy pointer content (parameter) from host to device.
	cudaMemcpy(sim_data->d_params, sim_data->params, n*sizeof(param_t), cudaMemcpyHostToDevice);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}
	// Copy pointer content (body metadata) from host to device.
	cudaMemcpy(sim_data->d_body_md, sim_data->body_md, n*sizeof(body_metadata_t), cudaMemcpyHostToDevice);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}
	// Copy pointer content (epoch) from host to device.
	cudaMemcpy(sim_data->d_epoch, sim_data->epoch, n*sizeof(ttt_t), cudaMemcpyHostToDevice);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}
}

void pp_disk::copy_to_host()
{
	const int n = n_bodies->total;

	// Copy pointer content (position and mass) from device to host
	cudaMemcpy(sim_data->pos, sim_data->d_pos, n*sizeof(posm_t), cudaMemcpyDeviceToHost);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}
	// Copy pointer content (velocity and radius) from device to host
	cudaMemcpy(sim_data->vel, sim_data->d_vel, n*sizeof(velR_t), cudaMemcpyDeviceToHost);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}
	// Copy pointer content (parameter) from from device to host
	cudaMemcpy(sim_data->params, sim_data->d_params, n*sizeof(param_t), cudaMemcpyDeviceToHost);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}
	// Copy pointer content (body metadata) from device to host
	cudaMemcpy(sim_data->body_md, sim_data->d_body_md, n*sizeof(body_metadata_t), cudaMemcpyDeviceToHost);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}
	// Copy pointer content (epoch) from device to host
	cudaMemcpy(sim_data->epoch, sim_data->d_epoch, n*sizeof(ttt_t), cudaMemcpyDeviceToHost);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}
}

void pp_disk::copy_variables_to_host()
{
	const int n = n_bodies->total;

	// Copy pointer content (position and mass) from device to host
	cudaMemcpy(sim_data->pos, sim_data->d_pos, n*sizeof(posm_t), cudaMemcpyDeviceToHost);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}
	// Copy pointer content (velocity and radius) from device to host
	cudaMemcpy(sim_data->vel, sim_data->d_vel, n*sizeof(velR_t), cudaMemcpyDeviceToHost);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}
}

var_t pp_disk::get_mass_of_star()
{
	body_metadata_t* body_md = sim_data->body_md;
	for (int j = 0; j < n_bodies->get_n_massive(); j++ ) {
		if (body_md[j].body_type == BODY_TYPE_STAR)
		{
			return sim_data->params[j].mass;
		}
	}
	throw string("No star is included!");
}

var_t pp_disk::get_total_mass()
{
	var_t totalMass = 0.0;

	param_t* param = sim_data->params;
	for (int j = 0; j < n_bodies->get_n_massive(); j++ ) {
		totalMass += param[j].mass;
	}

	return totalMass;
}

void pp_disk::compute_bc(posm_t* R0, velR_t* V0)
{
	posm_t* coor = sim_data->pos;
	velR_t* velo = sim_data->vel;

	for (int j = 0; j < n_bodies->get_n_massive(); j++ ) {
		R0->x += coor[j].m * coor[j].x;
		R0->y += coor[j].m * coor[j].y;
		R0->z += coor[j].m * coor[j].z;

		V0->x += coor[j].m * velo[j].x;
		V0->y += coor[j].m * velo[j].y;
		V0->z += coor[j].m * velo[j].z;
	}
	var_t M0 = get_total_mass();

	R0->x /= M0;	R0->y /= M0;	R0->z /= M0;
	V0->x /= M0;	V0->y /= M0;	V0->z /= M0;
}

void pp_disk::transform_to_bc()
{
	cout << "Transforming to barycentric system ... ";

	// Position and velocity of the system's barycenter
	posm_t R0 = {0.0, 0.0, 0.0, 0.0};
	velR_t V0 = {0.0, 0.0, 0.0, 0.0};

	compute_bc(&R0, &V0);

	posm_t* coor = sim_data->pos;
	velR_t* velo = sim_data->vel;
	// Transform the bodies coordinates and velocities
	for (int j = 0; j < n_bodies->get_n_total(); j++ ) {
		coor[j].x -= R0.x;		coor[j].y -= R0.y;		coor[j].z -= R0.z;
		velo[j].x -= V0.x;		velo[j].y -= V0.y;		velo[j].z -= V0.z;
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

	posm_t* coor = sim_data->pos;
	velR_t* velo = sim_data->vel;
	param_t* param = sim_data->params;
	body_metadata_t* body_md = sim_data->body_md;
	ttt_t* epoch = sim_data->epoch;

	if (input) {
		int_t	type = 0;
		string	dummy;
        		
		for (int i = 0; i < n_bodies->total; i++) { 
			body_md[i].active = true;
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
			input >> param[i].mass;
			coor[i].m = param[i].mass;
			// radius
			input >> velo[i].R;
			// density
			input >> param[i].density;
			// stokes constant
			input >> param[i].cd;

			// migration type
			input >> type;
			body_md[i].mig_type = static_cast<migration_type_t>(type);
			// migration stop at
			input >> param[i].mig_stop_at;

			// position
			input >> coor[i].x;
			input >> coor[i].y;
			input >> coor[i].z;
			// velocity
			input >> velo[i].x;
			input >> velo[i].y;
			input >> velo[i].z;
		}
        input.close();
	}
	else {
		throw string("Cannot open " + path + ".");
	}

	cout << "done" << endl;
}

void pp_disk::print_body_data(ostream& sout)
{
	posm_t* coor = sim_data->pos;
	velR_t* velo = sim_data->vel;
	param_t* param = sim_data->params;
	body_metadata_t* body_md = sim_data->body_md;
	ttt_t* epoch = sim_data->epoch;

	for (int i = 0; i < n_bodies->total; i++) {
		sout << body_md[i].id << SEP
			 << body_names[i] << SEP
			 << body_md[i].body_type << SEP 
			 << epoch[i] << SEP
			 << param[i].mass << SEP
			 << velo[i].R << SEP
			 << param[i].density << SEP
			 << param[i].cd << SEP
			 << body_md[i].mig_type << SEP
			 << param[i].mig_stop_at << SEP
			 << coor[i].x << SEP
			 << coor[i].y << SEP
			 << coor[i].z << SEP
			 << velo[i].x << SEP
			 << velo[i].y << SEP
			 << velo[i].z << endl;
    }
}
