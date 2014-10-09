// includes system
#include <iostream>
#include <iomanip>
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
#include "util.h"

using namespace std;
using namespace redutilcu;

__constant__ var_t dc_threshold[THRESHOLD_N];

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
	void kernel_dummy()
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = i*i;
}

static __global__
	void kernel_check_for_ejection_hit_centrum
	(
		ttt_t t, 
		interaction_bound int_bound, 
		const param_t* p, 
		const vec_t* r, 
		const vec_t* v, 
		body_metadata_t* body_md, 
		event_data_t* events,
		int *event_counter
	)
{
	const int i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	// Ignore the star and the inactive bodies
	if (i < int_bound.sink.y && body_md[i].id > 0 && body_md[i].body_type != BODY_TYPE_STAR)
	{
		unsigned int k = 0;
		vec_t dVec = {0.0, 0.0, 0.0, 0.0};

		// Calculate the distance from the star
		// 3 FLOP
		dVec.x = r[i].x - r[0].x;
		dVec.y = r[i].y - r[0].y;
		dVec.z = r[i].z - r[0].z;
		// 5 FLOP
		dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2
		if (dVec.w > SQR(dc_threshold[THRESHOLD_EJECTION_DISTANCE]))
		{
			k = atomicAdd(event_counter, 1);
			//printf("t = %20.10le d = %20.10le %d. EJECTION detected: id: %5d id: %5d\n", t, sqrt(dVec.w), k+1, body_md[0].id, body_md[i].id);

			events[k].event_name = EVENT_NAME_EJECTION;
			events[k].d = sqrt(dVec.w);
			events[k].t = t;
			events[k].id.x = body_md[0].id;
			events[k].id.y = body_md[i].id;
			events[k].idx.x = 0;
			events[k].idx.y = i;
			events[k].r1 = r[0];
			events[k].v1 = v[0];
			events[k].r2 = r[i];
			events[k].v2 = v[i];
			// Make the body inactive
			body_md[i].id *= -1;
		}
		else if (dVec.w < SQR(dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE]))
		{
			k = atomicAdd(event_counter, 1);
			//printf("t = %20.10le d = %20.10le %d. HIT_CENTRUM detected: id: %5d id: %5d\n", t, sqrt(dVec.w), k+1, body_md[0].id, body_md[i].id);

			events[k].event_name = EVENT_NAME_HIT_CENTRUM;
			events[k].d = sqrt(dVec.w);
			events[k].t = t;
			events[k].id.x = body_md[0].id;
			events[k].id.y = body_md[i].id;
			events[k].idx.x = 0;
			events[k].idx.y = i;
			events[k].r1 = r[0];
			events[k].v1 = v[0];
			events[k].r2 = r[i];
			events[k].v2 = v[i];
			// Make the body inactive
			body_md[i].id *= -1;
		}
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
				if (i > 0 && i > j && d < dc_threshold[THRESHOLD_COLLISION_FACTOR] * (p[i].radius + p[j].radius))
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

static __global__
	void kernel_print_constant_memory()
{
	printf("dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE] : %lf\n", dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE]);
	printf("dc_threshold[THRESHOLD_EJECTION_DISTANCE   ] : %lf\n", dc_threshold[THRESHOLD_EJECTION_DISTANCE]);
	printf("dc_threshold[THRESHOLD_COLLISION_FACTOR    ] : %lf\n", dc_threshold[THRESHOLD_COLLISION_FACTOR]);
}

/****************** KERNEL functions ends  here  ******************/

/****************** TEST functions begins here ********************/

void test_print_position(int n, const vec_t* r)
{
	for (int i = 0; i < n; i++)
	{
		printf("[%d]: (%20.16lf, %20.16lf, %20.16lf, %20.16lf)\n", i, r[i].x, r[i].y, r[i].z, r[i].w);
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

	kernel_print_constant_memory<<<1, 1>>>();
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_print_constant_memory failed", cudaStatus);
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
	
	int n_sink = n_bodies->get_n_self_interacting();
	if (0 < n_sink) {
		interaction_bound int_bound = n_bodies->get_self_interacting();
		set_kernel_launch_param(n_sink);

		kernel_calc_grav_accel<<<grid, block>>>
			(curr_t, int_bound, sim_data->d_body_md, sim_data->d_p, r, v, dy, d_events, d_event_counter);
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("kernel_calc_grav_accel failed", cudaStatus);
		}
	}

	n_sink = n_bodies->test_particle;
	if (0 < n_sink) {
		interaction_bound int_bound = n_bodies->get_non_interacting();
		set_kernel_launch_param(n_sink);

		kernel_calc_grav_accel<<<grid, block>>>
			(curr_t, int_bound, sim_data->d_body_md, sim_data->d_p, r, v, dy, d_events, d_event_counter);
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("kernel_calc_grav_accel failed", cudaStatus);
		}
	}
}

int pp_disk::call_kernel_check_for_ejection_hit_centrum()
{
	cudaError_t cudaStatus = cudaSuccess;
	
	int n_total = n_bodies->total;
	interaction_bound int_bound(0, n_total, 0, 0);
	set_kernel_launch_param(n_total);

	kernel_check_for_ejection_hit_centrum<<<grid, block>>>
		(t, int_bound, sim_data->d_p, sim_data->d_y[0], sim_data->d_y[1], sim_data->d_body_md, d_events, d_event_counter);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_check_for_ejection_hit_centrum failed", cudaStatus);
	}

	return get_n_event();
}

void pp_disk::calc_dy(int i, int rr, ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy)
{
	cudaError_t cudaStatus = cudaSuccess;
	const int n = n_bodies->get_n_total();

	switch (i)
	{
	case 0:
		// Copy velocities from previous step
		kernel_dummy<<<40, 256>>>();
#if 0 // NSIGHT CODE
		cudaMemcpy(dy, v, n * sizeof(vec_t), cudaMemcpyDeviceToDevice);
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("cudaMemcpy failed", cudaStatus);
		}
#endif
		break;
	case 1:
		if (rr == 0)
		{
		}
		// Calculate accelerations originated from the gravitational force
		call_kernel_calc_grav_accel(curr_t, r, v, dy);
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("call_kernel_calc_grav_accel failed", cudaStatus);
		}
		break;
	}
}

int pp_disk::handle_collision()
{
	create_sp_events();

	// TODO: implement collision graph: bredth-first search

	for (int i = 0; i < sp_events.size(); i++)
	{
		handle_collision_pair(i, &sp_events[i]);
		n_collision++;
	}

	return sp_events.size();
}

int2_t pp_disk::handle_ejection_hit_centrum()
{
	sp_events.resize(event_counter);
	survivors.resize(event_counter);

	int2_t number = {0, 0};

	for (int i = 0; i < event_counter; i++)
	{
		sp_events[i] = events[i];
		if (sp_events[i].event_name == EVENT_NAME_EJECTION)
		{
			n_ejection++;
			number.x++;
		}
		else
		{
			handle_collision_pair(i, &sp_events[i]);
			n_hit_centrum++;
			number.y++;
		}
	}

	return number;
}

void pp_disk::create_sp_events()
{
	sp_events.resize(event_counter);

	sp_events[0] = events[0];
	int k = 0;
	// Iterates over all collisions
	for (int i = 1; i < event_counter; i++)
	{
		if (sp_events[k].id.x == events[i].id.x && sp_events[k].id.y == events[i].id.y)
		{
			if (sp_events[k].t > events[i].t)
			{
				sp_events[k] = events[i];
			}
			continue;
		}
		else
		{
			k++;
			sp_events[k] = events[i];
		}
	}
	sp_events.resize(k+1);
	survivors.resize(k+1);
}

void pp_disk::handle_collision_pair(int i, event_data_t *collision)
{
	int survivIdx = collision->idx.x;
	int mergerIdx = collision->idx.y;

	// Calculate position and velocitiy of the new object
	vec_t r0 = {0.0, 0.0, 0.0, 0.0};
	vec_t v0 = {0.0, 0.0, 0.0, 0.0};

	var_t m_surviv = sim_data->p[survivIdx].mass;
	var_t m_merger = sim_data->p[mergerIdx].mass;
	calc_phase_after_collision(m_surviv, m_merger, &(collision->r1), &(collision->v1), &(collision->r2), &(collision->v2), r0, v0);

	if (BODY_TYPE_SUPERPLANETESIMAL == sim_data->body_md[mergerIdx].body_type)
	{
		// TODO: implement collision between a body and a super-planetesimal
		throw string("Collision between a massive body and a super-planetesimal is not yet implemented.");
	}
	// Calculate mass, volume, radius and density of the new object
	var_t mass	 = m_surviv + m_merger;
	// Calculate V = V1 + V2
	var_t volume = 4.188790204786391 * (CUBE(sim_data->p[mergerIdx].radius) + CUBE(sim_data->p[survivIdx].radius));
	var_t radius = pow(0.238732414637843 * volume, 1.0/3.0);
	var_t density= mass / volume;

	// Update mass, density and radius of survivor
	sim_data->p[survivIdx].mass	   = mass;
	sim_data->p[survivIdx].density = density;
	sim_data->p[survivIdx].radius  = radius;

	// Make the merged body inactive
	sim_data->body_md[mergerIdx].id *= -1;

	// Store the data of the survivor
	survivors[i].r = r0;
	survivors[i].v = v0;
	survivors[i].p = sim_data->p[survivIdx];
	survivors[i].body_md = sim_data->body_md[survivIdx];

	copy_vector_to_device((void **)&sim_data->d_y[0][survivIdx],	(void *)&r0,							sizeof(vec_t));
	copy_vector_to_device((void **)&sim_data->d_y[1][survivIdx],	(void *)&v0,							sizeof(vec_t));
	copy_vector_to_device((void **)&sim_data->d_p[survivIdx],		(void *)&sim_data->p[survivIdx],		sizeof(param_t));
	copy_vector_to_device((void **)&sim_data->d_body_md[mergerIdx],	(void *)&sim_data->body_md[mergerIdx],	sizeof(body_metadata_t));
}

void pp_disk::calc_phase_after_collision(var_t m1, var_t m2, const vec_t* r1, const vec_t* v1, const vec_t* r2, const vec_t* v2, vec_t& r0, vec_t& v0)
{
	const var_t M = m1 + m2;

	r0.x = (m1 * r1->x + m2 * r2->x) / M;
	r0.y = (m1 * r1->y + m2 * r2->y) / M;
	r0.z = (m1 * r1->z + m2 * r2->z) / M;

	v0.x = (m1 * v1->x + m2 * v2->x) / M;
	v0.y = (m1 * v1->y + m2 * v2->y) / M;
	v0.z = (m1 * v1->z + m2 * v2->z) / M;
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
	d_g_disk(0x0),
	t(0.0),
	sim_data(0x0),
	n_bodies(0x0),
	n_hit_centrum(0),
	n_ejection(0),
	n_collision(0),
	event_counter(0),
	d_event_counter(0x0)
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

void pp_disk::remove_inactive_bodies()
{
	sim_data_t *sim_data_temp = new sim_data_t;

	int n_save_nBody = n_bodies->total;
	int n_active_body = 0;
	int n_inactive_body = 0;
# if 1 // Find out the number of active/inactive bodies
	{
		int		star				= 0;
		int		giant_planet		= 0;
		int		rocky_planet		= 0;
		int		proto_planet		= 0;
		int		super_planetesimal	= 0;
		int		planetesimal		= 0;
		int		test_particle		= 0;

		for (int i = 0; i < n_bodies->total; i++)
		{
			if (sim_data->body_md[i].id > 0)
			{
				n_active_body++;
			}
			// Count the inactive bodies by type
			else
			{
				n_inactive_body++;
				switch (sim_data->body_md[i].body_type)
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

		n_bodies->star -= star;
		n_bodies->giant_planet -= giant_planet;
		n_bodies->rocky_planet -= rocky_planet;
		n_bodies->proto_planet -= proto_planet;
		n_bodies->super_planetesimal -= super_planetesimal;
		n_bodies->planetesimal -= planetesimal;
		n_bodies->test_particle -= test_particle;
		n_bodies->total = n_active_body;
	}
#endif

	sim_data_temp->y.resize(2);
	for (int i = 0; i < 2; i++)
	{
		sim_data_temp->y[i]	= new vec_t[n_bodies->total];
	}
	sim_data_temp->p		= new param_t[n_bodies->total];
	sim_data_temp->body_md	= new body_metadata_t[n_bodies->total];

	// Copy the data of active bodies to sim_data_temp
	int k = 0;
	for (int i = 0; i < n_save_nBody; i++)
	{
		if (sim_data->body_md[i].id > 0)
		{
			// Copy position
			sim_data_temp->y[0][k]		= sim_data->y[0][i];
			// Copy velocity
			sim_data_temp->y[1][k]		= sim_data->y[1][i];
			// Copy parameters
			sim_data_temp->p[k]			= sim_data->p[i];
			// Copy metadata
			sim_data_temp->body_md[k]	= sim_data->body_md[i];
			k++;
		}
	}

	// Copy the active bodies back to sim_data
	for (int i = 0; i < n_bodies->total; i++)
	{
		// Copy position
		sim_data->y[0][i]		= sim_data_temp->y[0][i];
		// Copy velocity
		sim_data->y[1][i]		= sim_data_temp->y[1][i];
		// Copy parameters
		sim_data->p[i]			= sim_data_temp->p[i];
		// Copy metadata
		sim_data->body_md[i]	= sim_data_temp->body_md[i];
	}

	// Copy the active bodies to the device
	copy_to_device();

	for (int i = 0; i < 2; i++)
	{
		delete[] sim_data_temp->y[i];
	}
	delete[] sim_data_temp->p;
	delete[] sim_data_temp->body_md;
	delete sim_data_temp;
}

void pp_disk::allocate_storage()
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
	copy_vector_to_device((void *)d_event_counter,		(void *)&event_counter,		1*sizeof(int));
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
	copy_vector_to_host((void *)&event_counter,			(void *)d_event_counter,	1*sizeof(int));
}

void pp_disk::copy_threshold_to_device(const var_t* threshold)
{
	// Calls the copy_constant_to_device in the util.cu
	copy_constant_to_device(dc_threshold, threshold, THRESHOLD_N*sizeof(var_t));
}

void pp_disk::copy_variables_to_host()
{
	const int n = n_bodies->total;

	for (int i = 0; i < 2; i++)
	{
		copy_vector_to_host((void *)sim_data->y[i],		(void *)sim_data->d_y[i],	n*sizeof(vec_t));
	}
}

void pp_disk::copy_event_data_to_host()
{
	copy_vector_to_host((void *)events, (void *)d_events, event_counter*sizeof(event_data_t));
}

int pp_disk::get_n_event()
{
	copy_vector_to_host((void *)&event_counter, (void *)d_event_counter, 1*sizeof(int));
	return event_counter;
}

int pp_disk::get_n_total_event()
{
	return n_collision + n_ejection + n_hit_centrum;
}

void pp_disk::clear_event_counter()
{
	event_counter = 0;
	copy_vector_to_device((void *)d_event_counter, (void *)&event_counter, 1*sizeof(int));
}

var_t pp_disk::get_mass_of_star()
{
	body_metadata_t* body_md = sim_data->body_md;
	for (int j = 0; j < n_bodies->get_n_massive(); j++ )
	{
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
	for (int j = 0; j < n_bodies->get_n_massive(); j++ )
	{
		totalMass += p[j].mass;
	}

	return totalMass;
}

void pp_disk::compute_bc(vec_t* R0, vec_t* V0)
{
	const param_t* p = sim_data->p;
	const vec_t* r = sim_data->y[0];
	const vec_t* v = sim_data->y[1];

	for (int j = 0; j < n_bodies->get_n_massive(); j++ )
	{
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
	for (int j = 0; j < n_bodies->total; j++ )
	{
		r[j].x -= R0.x;		r[j].y -= R0.y;		r[j].z -= R0.z;
		v[j].x -= V0.x;		v[j].y -= V0.y;		v[j].z -= V0.z;
	}

	cout << "done" << endl;
}

void pp_disk::transform_time()
{
	vec_t* v = sim_data->y[1];
	// Transform the bodies coordinates and velocities
	for (int j = 0; j < n_bodies->total; j++ )
	{
		sim_data->epoch[j] *= constants::Gauss;

		v[j].x /= constants::Gauss;
		v[j].y /= constants::Gauss;
		v[j].z /= constants::Gauss;
	}
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

void pp_disk::print_result_ascii(ostream& sout)
{
	vec_t* r = sim_data->y[0];
	vec_t* v = sim_data->y[1];
	param_t* p = sim_data->p;
	body_metadata_t* body_md = sim_data->body_md;

	for (int i = 0; i < n_bodies->total; i++) {
		// Skip inactive body
		if (body_md[i].id < 0)
		{
			continue;
		}
		sout << body_md[i].id << SEP
			 << body_names[i] << SEP
			 << body_md[i].body_type << SEP 
			 << t << SEP
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

void pp_disk::print_result_binary(ostream& sout)
{
	throw string("print_result_binary is not implemented");
}

void pp_disk::print_event_data(ostream& sout, ostream& log_f)
{
	static char sep = ' ';
	static char *e_names[] = {"NONE", "HIT_CENTRUM", "EJECTION", "CLOSE_ENCOUNTER", "COLLISION"};

	for (int i = 0; i < sp_events.size(); i++)
	{
		sout << setw(16) << e_names[sp_events[i].event_name] << sep
			 << setw(24) << setprecision(16) << sp_events[i].t << sep
			 << setw(24) << setprecision(16) << sp_events[i].d << sep
			 << setw( 8) << sp_events[i].id.x << sep
			 << setw( 8) << sp_events[i].id.y << sep
			 << setw(24) << setprecision(10) << sim_data->p[sp_events[i].idx.x].mass << sep
			 << setw(24) << setprecision(10) << sim_data->p[sp_events[i].idx.x].density << sep
			 << setw(24) << setprecision(10) << sim_data->p[sp_events[i].idx.x].radius << sep
			 << setw(24) << setprecision(10) << sp_events[i].r1.x << sep
			 << setw(24) << setprecision(10) << sp_events[i].r1.y << sep
			 << setw(24) << setprecision(10) << sp_events[i].r1.z << sep
			 << setw(24) << setprecision(10) << sp_events[i].v1.x << sep
			 << setw(24) << setprecision(10) << sp_events[i].v1.y << sep
			 << setw(24) << setprecision(10) << sp_events[i].v1.z << sep
			 << setw(24) << setprecision(10) << sim_data->p[sp_events[i].idx.y].mass << sep
			 << setw(24) << setprecision(10) << sim_data->p[sp_events[i].idx.y].density << sep
			 << setw(24) << setprecision(10) << sim_data->p[sp_events[i].idx.y].radius << sep
			 << setw(24) << setprecision(10) << sp_events[i].r2.x << sep
			 << setw(24) << setprecision(10) << sp_events[i].r2.y << sep
			 << setw(24) << setprecision(10) << sp_events[i].r2.z << sep
			 << setw(24) << setprecision(10) << sp_events[i].v2.x << sep
			 << setw(24) << setprecision(10) << sp_events[i].v2.y << sep
			 << setw(24) << setprecision(10) << sp_events[i].v2.z << sep
			 << setw( 8) << survivors[i].body_md.id << sep
			 << setw(24) << setprecision(10) << survivors[i].p.mass << sep
			 << setw(24) << setprecision(10) << survivors[i].p.density << sep
			 << setw(24) << setprecision(10) << survivors[i].p.radius << sep
			 << setw(24) << setprecision(10) << survivors[i].r.x << sep
			 << setw(24) << setprecision(10) << survivors[i].r.y << sep
			 << setw(24) << setprecision(10) << survivors[i].r.z << sep
			 << setw(24) << setprecision(10) << survivors[i].v.x << sep
			 << setw(24) << setprecision(10) << survivors[i].v.y << sep
			 << setw(24) << setprecision(10) << survivors[i].v.z << sep << endl;

		if (log_f)
		{
			log_f << tools::get_time_stamp() << sep 
				  << e_names[sp_events[i].event_name] << sep
			      << setw( 8) << sp_events[i].id.x << sep
			      << setw( 8) << sp_events[i].id.y << sep << endl;
		}
	}
}
