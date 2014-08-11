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

/****************** KERNEL functions begins here ******************/

__global__
	void kernel_print_position(int n, const posm_t* pos)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
	{
		printf("%4d\n", tid);
		var_t x = pos[tid].x;
		//printf("pos[%4d].x : %20.16lf\n", tid, pos[tid].x);
		//printf("pos[%4d].y : %20.16lf\n", tid, pos[tid].y);
		//printf("pos[%4d].z : %20.16lf\n", tid, pos[tid].z);
		//printf("pos[%4d].m : %20.16lf\n", tid, pos[tid].m);
	}
}

/****************** KERNEL functions ends  here  ******************/


pp_disk::pp_disk(string& path, gas_disk *gd) :
	t(0.0),
	sim_data(0),
	n_bodies(0)
{
	get_number_of_bodies(path);
	allocate_storage();
	load(path);
	copy_to_device();
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
	cudaFree(sim_data->d_vel);
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
	cudaMalloc((void**) &(sim_data->d_vel), n*sizeof(velR_t));
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

var_t pp_disk::get_mass_of_star()
{
	body_metadata_t* body_md = sim_data->body_md;
	for (int j = 0; j < n_bodies->n_massive(); j++ ) {
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
	for (int j = 0; j < n_bodies->n_massive(); j++ ) {
		totalMass += param[j].mass;
	}

	return totalMass;
}

void pp_disk::compute_bc(posm_t* R0, velR_t* V0)
{
	posm_t* coor = sim_data->pos;
	velR_t* velo = sim_data->vel;

	for (int j = 0; j < n_bodies->n_massive(); j++ ) {
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
	for (int j = 0; j < n_bodies->n_total(); j++ ) {
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
