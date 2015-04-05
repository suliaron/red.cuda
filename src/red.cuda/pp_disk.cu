// includes system
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>

// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes project
#include "nbody_exception.h"
#include "pp_disk.h"
#include "redutilcu.h"
#include "red_constants.h"
#include "red_macro.h"
#include "red_type.h"

using namespace std;
using namespace redutilcu;

#define GAS_REDUCTION_THRESHOLD 1.0e-6
#define GAS_INNER_EDGE 0.1              // [AU]


__constant__ var_t dc_threshold[THRESHOLD_N];
__constant__ analytic_gas_disk_params_t dc_anal_gd_params;
__constant__ fargo_gas_disk_params_t dc_fargo_gd_params;

///****************** DEVICE functions begins here ******************/

/****************** KERNEL functions begins here ******************/

namespace pp_disk_utility
{
static __host__ __device__ 
	void store_event_data
	(
		event_name_t name,
		ttt_t t,
		var_t d,
		int idx1,
		int idx2,
		const param_t* p,
		const vec_t* r,
		const vec_t* v,
		const body_metadata_t* body_md,
		event_data_t *evnt)
{
	evnt->event_name = name;
	evnt->d = d;
	evnt->t = t;
	evnt->id1 = body_md[idx1].id;
	evnt->id2 = body_md[idx2].id;
	evnt->idx1 = idx1;
	evnt->idx2 = idx2;
	evnt->r1 = r[idx1];
	evnt->v1 = v[idx1];
	evnt->r2 = r[idx2];
	evnt->v2 = v[idx2];

	if (EVENT_NAME_EJECTION == name)
	{
		evnt->p1 = p[idx1];
		evnt->p2 = p[idx2];

		evnt->rs = evnt->r1;
		evnt->vs = evnt->v1;
		evnt->ps = evnt->p1;
	}
}


__host__ __device__
	var_t reduction_factor(gas_decrease_t gas_decrease, ttt_t t0, ttt_t t1, ttt_t e_folding_time, ttt_t t)
{
	switch (gas_decrease) 
	{
	case GAS_DENSITY_CONSTANT:
		return 1.0;
	case GAS_DENSITY_DECREASE_LINEAR:
		if (t <= t0)
		{
			return 1.0;
		}
		else if (t0 < t && t <= t1 && t0 != t1)
		{
			return 1.0 - (t - t0)/(t1 - t0);
		}
		else
		{
			return 0.0;
		}
	case GAS_DENSITY_DECREASE_EXPONENTIAL:
		return exp(-(t - t0)/e_folding_time);
	default:
		return 1.0;
	}
}

__host__ __device__
	var_t get_density(var2_t sch, var2_t rho, const vec_t* rVec)
{
	var_t density = 0.0;

	var_t r		= sqrt(SQR(rVec->x) + SQR(rVec->y));
	var_t h		= sch.x * pow(r, sch.y);
	var_t arg	= SQR(rVec->z/h);
	if (GAS_INNER_EDGE < r)
	{
		density	= rho.x * pow(r, rho.y) * exp(-arg);
	}
	else
	{
		var_t a	= rho.x * pow(GAS_INNER_EDGE, rho.y - 4.0);
		density	= a * SQR(SQR(r)) * exp(-arg);
	}

	return density;
}


__host__ __device__
	vec_t circular_velocity(var_t mu, const vec_t* rVec)
{
	vec_t result = {0.0, 0.0, 0.0, 0.0};

	var_t r  = sqrt(SQR(rVec->x) + SQR(rVec->y));
	var_t vc = sqrt(mu/r);

	var_t p = 0.0;
	if (rVec->x == 0.0 && rVec->y == 0.0)
	{
		return result;
	}
	else if (rVec->y == 0.0)
	{
		result.y = rVec->x > 0.0 ? vc : -vc;
	}
	else if (rVec->x == 0.0)
	{
		result.x = rVec->y > 0.0 ? -vc : vc;
	}
	else if (rVec->x >= rVec->y)
	{
		p = rVec->y / rVec->x;
		result.y = rVec->x >= 0 ? vc/sqrt(1.0 + SQR(p)) : -vc/sqrt(1.0 + SQR(p));
		result.x = -result.y*p;
	}
	else
	{
		p = rVec->x / rVec->y;
		result.x = rVec->y >= 0 ? -vc/sqrt(1.0 + SQR(p)) : vc/sqrt(1.0 + SQR(p));
		result.y = -result.x*p;
	}

	return result;
}

__host__ __device__
	vec_t get_velocity(var_t mu, var2_t eta, const vec_t* rVec)
{
	vec_t v_gas = circular_velocity(mu, rVec);
	var_t r = sqrt(SQR(rVec->x) + SQR(rVec->y));

	var_t v = sqrt(1.0 - 2.0*eta.x * pow(r, eta.y));
	v_gas.x *= v;
	v_gas.y *= v;
	
	return v_gas;
}
} /* pp_disk_utility */

namespace kernel_pp_disk
{
static __global__
	void check_for_ejection_hit_centrum
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

	// Ignore the star, the padding particles (whose id = 0) and the inactive bodies (whose id < 0)
	if (i < int_bound.sink.y && body_md[i].id > 0 && body_md[i].body_type != BODY_TYPE_STAR)
	{
		unsigned int k = 0;

		// Calculate the distance from the barycenter
		var_t r2 = SQR(r[i].x) + SQR(r[i].y) + SQR(r[i].z);
		if (0.0 < dc_threshold[THRESHOLD_EJECTION_DISTANCE] && dc_threshold[THRESHOLD_EJECTION_DISTANCE_SQUARED] < r2)
		{
			k = atomicAdd(event_counter, 1);
			pp_disk_utility::store_event_data(EVENT_NAME_EJECTION, t, sqrt(r2), 0, i, p, r, v, body_md, &events[k]);
			//printf("t = %20.10le d = %20.10le %d. EJECTION detected: id: %5d id: %5d\n", t, sqrt(dVec.w), k+1, body_md[0].id, body_md[i].id);

			//events[k].event_name = EVENT_NAME_EJECTION;
			//events[k].d = sqrt(r2); //sqrt(dVec.w);
			//events[k].t = t;
			//events[k].id1 = body_md[0].id;
			//events[k].id2 = body_md[i].id;
			//events[k].idx1 = 0;
			//events[k].idx2 = i;
			//events[k].r1 = r[0];
			//events[k].v1 = v[0];
			//events[k].r2 = r[i];
			//events[k].v2 = v[i];

			//events[k].p1 = p[0];
			//events[k].p2 = p[i];
			//events[k].ps = p[0];
			//events[k].rs = r[0];
			//events[k].vs = v[0];

			// Make the body inactive
			body_md[i].id *= -1;
		}
		else if (0.0 < dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE] && dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE_SQUARED] > r2)
		{
			k = atomicAdd(event_counter, 1);
			pp_disk_utility::store_event_data(EVENT_NAME_HIT_CENTRUM, t, sqrt(r2), 0, i, p, r, v, body_md, &events[k]);
			//printf("t = %20.10le d = %20.10le %d. HIT_CENTRUM detected: id: %5d id: %5d\n", t, sqrt(dVec.w), k+1, body_md[0].id, body_md[i].id);

			//events[k].event_name = EVENT_NAME_HIT_CENTRUM;
			//events[k].d = sqrt(r2); //sqrt(dVec.w);
			//events[k].t = t;
			//events[k].id1 = body_md[0].id;
			//events[k].id2 = body_md[i].id;
			//events[k].idx1 = 0;
			//events[k].idx2 = i;
			//events[k].r1 = r[0];
			//events[k].v1 = v[0];
			//events[k].r2 = r[i];
			//events[k].v2 = v[i];
			// Make the body inactive
			body_md[i].id *= -1;
		}
	}
}

static __global__
	void calc_grav_accel_int_mul_of_thread_per_block
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

		if (i > 0 && i > j && d < dc_threshold[THRESHOLD_RADII_ENHANCE_FACTOR] * (p[i].radius + p[j].radius))
		{
			unsigned int k = atomicAdd(event_counter, 1);

			int survivIdx = i;
			int mergerIdx = j;
			if (p[mergerIdx].mass > p[survivIdx].mass)
			{
				int m = survivIdx;
				survivIdx = mergerIdx;
				mergerIdx = m;
			}
			//printf("t = %20.10le d = %20.10le %d. COLLISION detected: id: %5d id: %5d\n", t, d, k+1, body_md[survivIdx].id, body_md[mergerIdx].id);

			pp_disk_utility::store_event_data(EVENT_NAME_COLLISION, t, d, survivIdx, mergerIdx, p, r, v, body_md, &events[k]);

			//events[k].event_name = EVENT_NAME_COLLISION;
			//events[k].d = d;
			//events[k].t = t;
			//events[k].id1 = body_md[survivIdx].id;
			//events[k].id2 = body_md[mergerIdx].id;
			//events[k].idx1 = survivIdx;
			//events[k].idx2 = mergerIdx;
			//events[k].r1 = r[survivIdx];
			//events[k].v1 = v[survivIdx];
			//events[k].r2 = r[mergerIdx];
			//events[k].v2 = v[mergerIdx];
		}
	}
}

static __global__
	void calc_grav_accel
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
		a[i].x = a[i].y = a[i].z = a[i].w = 0.0;
		if (0 < body_md[i].id)
		{
			vec_t dVec = {0.0, 0.0, 0.0, 0.0};
			for (int j = int_bound.source.x; j < int_bound.source.y; j++) 
			{
				/* Skip the body with the same index and those which are inactive ie. id < 0 */
				if (i == j || 0 > body_md[j].id)
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
				if (0 < i && i > j && d < dc_threshold[THRESHOLD_RADII_ENHANCE_FACTOR] * (p[i].radius + p[j].radius))
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

					pp_disk_utility::store_event_data(EVENT_NAME_COLLISION, t, d, survivIdx, mergerIdx, p, r, v, body_md, &events[k]);

					//events[k].event_name = EVENT_NAME_COLLISION;
					//events[k].d = d;
					//events[k].t = t;
					//events[k].id1 = body_md[survivIdx].id;
					//events[k].id2 = body_md[mergerIdx].id;
					//events[k].idx1 = survivIdx;
					//events[k].idx2 = mergerIdx;
					//events[k].r1 = r[survivIdx];
					//events[k].v1 = v[survivIdx];
					//events[k].r2 = r[mergerIdx];
					//events[k].v2 = v[mergerIdx];
				}
			} // 36 FLOP
		}
	}
}

static __global__
	void calc_drag_accel_NSI
	(
		ttt_t curr_t,
		interaction_bound int_bound, 
		const body_metadata_t* body_md, 
		const param_t* p, 
		const vec_t* r, 
		const vec_t* v, 
		vec_t* a
	)
{
	const int i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	if (i < int_bound.sink.y)
	{
		var_t decr_fact = pp_disk_utility::reduction_factor(dc_anal_gd_params.gas_decrease, dc_anal_gd_params.t0, dc_anal_gd_params.t1, dc_anal_gd_params.e_folding_time, curr_t);
		// TODO: export 1.0e-6 into the gas disk description file
		if (GAS_REDUCTION_THRESHOLD > decr_fact)
		{
			return;
		}

		var_t m_star = p[0].mass;
		var_t mu        = 1.0 * (m_star + p[i].mass);
		vec_t v_g       = pp_disk_utility::get_velocity(mu, dc_anal_gd_params.eta, &r[i]);
		vec_t u         = {v_g.x - v[i].x, v_g.y - v[i].y, v_g.z - v[i].z, 0.0};
		var_t u_n       = sqrt(SQR(u.x) + SQR(u.y) + SQR(u.z));
		var_t density_g = pp_disk_utility::get_density(dc_anal_gd_params.sch, dc_anal_gd_params.rho, &r[i]);

		var_t f = decr_fact * (3.0 * p[i].cd * density_g * u_n) / (8.0 * p[i].radius * p[i].density);

		a[i].x += f * u.x;
		a[i].y += f * u.y;
		a[i].z += f * u.z;
	}
}

} /* kernel_pp_disk */

namespace kernel_utility
{
static __global__
	void print_body_metadata(int n, const body_metadata_t* body_md)
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
	void print_epochs(int n, const ttt_t* epoch)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		printf("epoch[%4d] : %20.16lf\n", i, epoch[i]);
	}
}

static __global__
	void print_vector(int n, const vec_t* v)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
	{
		printf("[%d]: (%20.16lf, %20.16lf, %20.16lf, %20.16lf)\n", tid, v[tid].x, v[tid].y, v[tid].z, v[tid].w);
	}
}

static __global__
	void print_constant_memory()
{
	printf("dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE        ] : %lf\n", dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE]);
	printf("dc_threshold[THRESHOLD_EJECTION_DISTANCE           ] : %lf\n", dc_threshold[THRESHOLD_EJECTION_DISTANCE]);
	printf("dc_threshold[THRESHOLD_RADII_ENHANCE_FACTOR        ] : %lf\n", dc_threshold[THRESHOLD_RADII_ENHANCE_FACTOR]);
	printf("dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE_SQUARED] : %lf\n", dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE_SQUARED]);
	printf("dc_threshold[THRESHOLD_EJECTION_DISTANCE_SQUARED   ] : %lf\n", dc_threshold[THRESHOLD_EJECTION_DISTANCE_SQUARED]);
}
} /* kernel_utility */


void pp_disk::test_call_kernel_print_sim_data()
{
	const int n_total = use_padded_storage ? n_bodies->get_n_prime_total() : n_bodies->get_n_total();

	set_kernel_launch_param(n_total);

	kernel_utility::print_vector<<<grid, block>>>(n_total, sim_data->d_y[0]);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_utility::print_vector failed", cudaStatus);
	}
	cudaDeviceSynchronize();

	kernel_utility::print_vector<<<grid, block>>>(n_total, sim_data->d_y[1]);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_utility::print_vector failed", cudaStatus);
	}
	cudaDeviceSynchronize();

	kernel_utility::print_vector<<<grid, block>>>(n_total, (vec_t*)sim_data->d_p);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_utility::print_vector failed", cudaStatus);
	}
	cudaDeviceSynchronize();

	kernel_utility::print_body_metadata<<<grid, block>>>(n_total, sim_data->d_body_md);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_utility::print_body_metadata failed", cudaStatus);
	}
	cudaDeviceSynchronize();

	kernel_utility::print_epochs<<<grid, block>>>(n_total, sim_data->d_epoch);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_utility::print_epochs failed", cudaStatus);
	}
	cudaDeviceSynchronize();

	kernel_utility::print_constant_memory<<<1, 1>>>();
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("kernel_utility::print_constant_memory failed", cudaStatus);
	}
	cudaDeviceSynchronize();
}

void pp_disk::set_kernel_launch_param(int n_data)
{
	int n_thread = min(n_tpb, n_data);
	int n_block = (n_data + n_thread - 1)/n_thread;

	grid.x	= n_block;
	block.x = n_thread;
}

void pp_disk::cpu_calc_drag_accel(ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy)
{
	int n_sink = n_bodies->get_n_NSI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_GD();
		cpu_calc_drag_accel_NSI(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy);
	}
}


void pp_disk::cpu_calc_drag_accel_NSI(ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a)
{
	var_t decr_fact = pp_disk_utility::reduction_factor(a_gd->params.gas_decrease, a_gd->params.t0, a_gd->params.t1, a_gd->params.e_folding_time, curr_t);
	// TODO: export 1.0e-6 into the gas disk description file
	if (GAS_REDUCTION_THRESHOLD > decr_fact)
	{
		return;
	}

	var_t m_star = p[0].mass;
	for (int i = int_bound.sink.x; i < int_bound.sink.y; i++)
	{
		var_t mu        = 1.0 * (m_star + p[i].mass);
		vec_t v_g       = pp_disk_utility::get_velocity(mu, a_gd->params.eta, &r[i]);
		vec_t u         = {v_g.x - v[i].x, v_g.y - v[i].y, v_g.z - v[i].z, 0.0};
		var_t u_n       = sqrt(SQR(u.x) + SQR(u.y) + SQR(u.z));
		var_t density_g = pp_disk_utility::get_density(a_gd->params.sch, a_gd->params.rho, &r[i]);

		var_t f = decr_fact * (3.0 * p[i].cd * density_g * u_n) / (8.0 * p[i].radius * p[i].density);

		a[i].x += f * u.x;
		a[i].y += f * u.y;
		a[i].z += f * u.z;

		//var_t rhoGas = rFactor * gas_density_at(gasDisk, (vec_t*)&coor[bodyIdx]);
		//var_t r = norm((vec_t*)&coor[bodyIdx]);

		//vec_t u;
		//u.x	= velo[bodyIdx].x - vGas.x;
		//u.y	= velo[bodyIdx].y - vGas.y;
		//u.z	= velo[bodyIdx].z - vGas.z;
		//var_t C	= 0.0;

		//var_t lambda = gasDisk->mfp.x * pow(r, gasDisk->mfp.y);
		//// Epstein-regime:
		//if (     params[bodyIdx].radius <= 0.1 * lambda)
		//{
		//	var_t vth = mean_thermal_speed_CMU(gasDisk, r);
		//	C = params[bodyIdx].gamma_epstein * vth * rhoGas;
		//}
		//// Stokes-regime:
		//else if (params[bodyIdx].radius >= 10.0 * lambda)
		//{
		//	C = params[bodyIdx].gamma_stokes * norm(&u) * rhoGas;
		//}
		//// Transition-regime:
		//else
		//{

		//}

		//acce[tid].x = -C * u.x;
		//acce[tid].y = -C * u.y;
		//acce[tid].z = -C * u.z;
		//acce[tid].w = 0.0;

		//printf("acce[tid].x: %10le\n", acce[tid].x);
		//printf("acce[tid].y: %10le\n", acce[tid].y);
		//printf("acce[tid].z: %10le\n", acce[tid].z);
	}
}

void pp_disk::cpu_calc_grav_accel_SI(ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a, event_data_t* events, int *event_counter)
{
	for (int i = int_bound.sink.x; i < int_bound.sink.y; i++)
	{
		a[i].x = a[i].y = a[i].z = a[i].w = 0.0;
		if (0 < body_md[i].id)
		{
			vec_t dVec = {0.0, 0.0, 0.0, 0.0};
			for (int j = int_bound.source.x; j < int_bound.source.y; j++) 
			{
				/* Skip the body with the same index and those which are inactive ie. id < 0 */
				if (i == j || 0 > body_md[j].id)
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
				if (i > 0 && i > j && d < threshold[THRESHOLD_RADII_ENHANCE_FACTOR] * (p[i].radius + p[j].radius))
				{
					int k = *event_counter;

					int survivIdx = i;
					int mergerIdx = j;
					if (p[mergerIdx].mass > p[survivIdx].mass)
					{
						int t = survivIdx;
						survivIdx = mergerIdx;
						mergerIdx = t;
					}
					printf("t = %20.10le d = %20.10le %d. COLLISION detected: id: %5d id: %5d\n", t, d, k+1, body_md[survivIdx].id, body_md[mergerIdx].id);

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

					(*event_counter)++;
				}
			} // 36 FLOP
		}
	}
}

void pp_disk::cpu_calc_grav_accel_NI( ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a, event_data_t* events, int *event_counter)
{
	cpu_calc_grav_accel_SI(t, int_bound, body_md, p, r, v, a, events, event_counter);
}

void pp_disk::cpu_calc_grav_accel_NSI(ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a, event_data_t* events, int *event_counter)
{
	cpu_calc_grav_accel_SI(t, int_bound, body_md, p, r, v, a, events, event_counter);
}

void pp_disk::cpu_calc_grav_accel(ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy)
{
	int n_sink = n_bodies->get_n_SI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_SI();
		cpu_calc_grav_accel_SI(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy, events, &event_counter);
	}

	n_sink = n_bodies->get_n_NSI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_NSI();
		cpu_calc_grav_accel_NSI(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy, events, &event_counter);
	}

	n_sink = n_bodies->get_n_NI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_NI();
		cpu_calc_grav_accel_NI(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy, events, &event_counter);
	}
}

void pp_disk::call_kernel_calc_grav_accel(ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy)
{
	cudaError_t cudaStatus = cudaSuccess;
	
	int n_sink = use_padded_storage ? n_bodies->get_n_prime_SI() : n_bodies->get_n_SI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_SI();
		set_kernel_launch_param(n_sink);

		if (use_padded_storage)
		{
			kernel_pp_disk::calc_grav_accel_int_mul_of_thread_per_block<<<grid, block>>>
				(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy, d_events, d_event_counter);
		}
		else
		{
			kernel_pp_disk::calc_grav_accel<<<grid, block>>>
				(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy, d_events, d_event_counter);
		}
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus)
		{
			throw nbody_exception("kernel_pp_disk::calc_grav_accel failed", cudaStatus);
		}
	}

	n_sink = use_padded_storage ? n_bodies->get_n_prime_NSI() : n_bodies->get_n_NSI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_NSI();
		set_kernel_launch_param(n_sink);

		if (use_padded_storage)
		{
			kernel_pp_disk::calc_grav_accel_int_mul_of_thread_per_block<<<grid, block>>>
				(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy, d_events, d_event_counter);
		}
		else
		{
			kernel_pp_disk::calc_grav_accel<<<grid, block>>>
				(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy, d_events, d_event_counter);
		}
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus)
		{
			throw nbody_exception("kernel_pp_disk::calc_grav_accel failed", cudaStatus);
		}
	}

	n_sink = use_padded_storage ? n_bodies->get_n_prime_NI() : n_bodies->get_n_NI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_NI();
		set_kernel_launch_param(n_sink);

		if (use_padded_storage)
		{
			kernel_pp_disk::calc_grav_accel_int_mul_of_thread_per_block<<<grid, block>>>
				(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy, d_events, d_event_counter);
		}
		else
		{
			kernel_pp_disk::calc_grav_accel<<<grid, block>>>
				(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy, d_events, d_event_counter);
		}
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus)
		{
			throw nbody_exception("kernel_pp_disk::calc_grav_accel failed", cudaStatus);
		}
	}
}

void pp_disk::call_kernel_calc_drag_accel(ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy)
{
	int n_sink = n_bodies->get_n_NSI();
	if (0 < n_sink)
	{
		set_kernel_launch_param(n_sink);
		interaction_bound int_bound = n_bodies->get_bound_GD();
		kernel_pp_disk::calc_drag_accel_NSI<<<grid, block>>>(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy);
	}
}

bool pp_disk::check_for_ejection_hit_centrum()
{
	// Number of ejection + hit centrum events
	int n_event = 0;
	switch (comp_dev)
	{
	case COMPUTING_DEVICE_CPU:
		n_event = cpu_check_for_ejection_hit_centrum();
		break;
	case COMPUTING_DEVICE_GPU:
		n_event = call_kernel_check_for_ejection_hit_centrum();
		break;
	}

	if (0 < n_event)
	{
		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			copy_event_data_to_host();
		}
		// handle_ejection_hit_centrum() will create sp_events vector which will explicitly written to the disk via print_event_data()
		handle_ejection_hit_centrum();
		cout << n_ejection[   EVENT_COUNTER_NAME_LAST_STEP] << " ejection ";
		cout << n_hit_centrum[EVENT_COUNTER_NAME_LAST_STEP] << " hit_centrum event(s) occured" << endl;

		n_ejection[   EVENT_COUNTER_NAME_LAST_STEP] = 0;
		n_hit_centrum[EVENT_COUNTER_NAME_LAST_STEP] = 0;

		return true;
	}

	return false;
}

bool pp_disk::check_for_collision()
{
	// Number of collision
	int n_event = get_n_event();

	if (0 < n_event)
	{
		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			copy_event_data_to_host();
		}
		// handle_collision() will create sp_events vector which will explicitly written to the disk via print_event_data()
		handle_collision();
		cout << n_collision[EVENT_COUNTER_NAME_LAST_STEP] << " collision event(s) occurred" << endl;

		n_collision[EVENT_COUNTER_NAME_LAST_STEP] = 0;

		return true;
	}

	return false;
}

bool pp_disk::check_for_rebuild_vectors(int n)
{
	if (n_event[EVENT_COUNTER_NAME_LAST_CLEAR] >= n)
	{
		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			copy_to_host();
		}
		// Rebuild the vectors and remove inactive bodies
		remove_inactive_bodies();
		set_event_counter(EVENT_COUNTER_NAME_LAST_CLEAR, 0);
		return true;
	}
	return false;
}

void pp_disk::store_event_data(event_name_t name, ttt_t t, var_t d, int idx1, int idx2, event_data_t *evnt)
{
	evnt->event_name = name;
	evnt->d = d;
	evnt->t = t;
	evnt->id1 = sim_data->body_md[idx1].id;
	evnt->id2 = sim_data->body_md[idx2].id;
	evnt->idx1 = idx1;
	evnt->idx2 = idx2;
	evnt->r1 = sim_data->y[0][idx1];
	evnt->v1 = sim_data->y[1][idx1];
	evnt->r2 = sim_data->y[0][idx2];
	evnt->v2 = sim_data->y[1][idx2];

	if (EVENT_NAME_EJECTION == name)
	{
		evnt->p1 = sim_data->h_p[idx1];
		evnt->p2 = sim_data->h_p[idx2];

		evnt->rs = evnt->r1;
		evnt->vs = evnt->v1;
		evnt->ps = evnt->p1;
	}
}

int pp_disk::cpu_check_for_ejection_hit_centrum()
{
	const vec_t* r = sim_data->y[0];
	const vec_t* v = sim_data->y[1];
	body_metadata_t* body_md = sim_data->body_md;
	
	int n_total = n_bodies->get_n_total();
	interaction_bound int_bound(0, n_total, 0, 0);

	for (int i = int_bound.sink.x; i < int_bound.sink.y; i++)
	{
		// Ignore the star and the inactive bodies (whose id < 0)
		if (0 < sim_data->body_md[i].id && BODY_TYPE_STAR != sim_data->body_md[i].body_type)
		{
			//int k = 0;

			// Calculate the distance from the barycenter
			var_t r2 = SQR(r[i].x) + SQR(r[i].y) + SQR(r[i].z);
			if (0.0 < threshold[THRESHOLD_EJECTION_DISTANCE] && threshold[THRESHOLD_EJECTION_DISTANCE_SQUARED] < r2)
			{
				//k = event_counter;
				//printf("t = %20.10le d = %20.10le %d. EJECTION detected: id: %5d id: %5d\n", t, sqrt(dVec.w), k+1, body_md[0].id, body_md[i].id);
				store_event_data(EVENT_NAME_EJECTION, t, sqrt(r2), 0, i, &events[event_counter]);

				// Make the body inactive
				body_md[i].id *= -1;
				event_counter++;
			}
			else if (0.0 < threshold[THRESHOLD_HIT_CENTRUM_DISTANCE] && threshold[THRESHOLD_HIT_CENTRUM_DISTANCE_SQUARED] > r2)
			{
				//k = event_counter;
				//printf("t = %20.10le d = %20.10le %d. HIT_CENTRUM detected: id: %5d id: %5d\n", t, sqrt(dVec.w), k+1, body_md[0].id, body_md[i].id);
				store_event_data(EVENT_NAME_HIT_CENTRUM, t, sqrt(r2), 0, i, &events[event_counter]);

				// Make the body inactive
				body_md[i].id *= -1;
				event_counter++;
			}
		}
	}

	return event_counter;
}

int pp_disk::call_kernel_check_for_ejection_hit_centrum()
{
	cudaError_t cudaStatus = cudaSuccess;
	
	int n_total = use_padded_storage ? n_bodies->get_n_prime_total() : n_bodies->get_n_total();
	interaction_bound int_bound(0, n_total, 0, 0);
	set_kernel_launch_param(n_total);

	kernel_pp_disk::check_for_ejection_hit_centrum<<<grid, block>>>
		(t, int_bound, sim_data->p, sim_data->y[0], sim_data->y[1], sim_data->body_md, d_events, d_event_counter);

	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw nbody_exception("kernel_pp_disk::check_for_ejection_hit_centrum failed", cudaStatus);
	}

	return get_n_event();
}

void pp_disk::calc_dydx(int i, int rr, ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy)
{
	cudaError_t cudaStatus = cudaSuccess;

	const int n_total = use_padded_storage ? n_bodies->get_n_prime_total() : n_bodies->get_n_total();

	switch (i)
	{
	case 0:
		if (COMPUTING_DEVICE_CPU == comp_dev)
		{
			memcpy(dy, v, n_total * sizeof(vec_t));
		}
		else
		{
			cudaMemcpy(dy, v, n_total * sizeof(vec_t), cudaMemcpyDeviceToDevice);
			cudaStatus = HANDLE_ERROR(cudaGetLastError());
			if (cudaSuccess != cudaStatus)
			{
				throw nbody_exception("cudaMemcpy failed", cudaStatus);
			}
		}
		break;
	case 1:  // Calculate accelerations originated from the gravitational force, drag force etc.
		if (COMPUTING_DEVICE_CPU == comp_dev)
		{
			/*
			 * SORREND:
			 * 1. Gravity
			 * 2. other forces
			 */
			cpu_calc_grav_accel(curr_t, r, v, dy);
			// This if will be used to speed-up the calculation when gas drag is also acting on the bodies.
			// (BUT early optimization is the root of much evil)
			if (rr == 0)
			{
			}
			if (GAS_DISK_MODEL_NONE != g_disk_model)
			{
				cpu_calc_drag_accel(curr_t, r, v, dy);
			}
		}
		else
		{
			call_kernel_calc_grav_accel(curr_t, r, v, dy);
			if (rr == 0)
			{
			}
			if (GAS_DISK_MODEL_NONE != g_disk_model)
			{
				call_kernel_calc_drag_accel(curr_t, r, v, dy);
			}
	// DEBUG CODE
	//		cudaDeviceSynchronize();
	// END DEBUG CODE
		}
		break;
	}
}

void pp_disk::swap()
{
	for (int i = 0; i < 2; i++)
	{
		::swap(sim_data->yout[i], sim_data->y[i]);
	}
}

void pp_disk::increment_event_counter(int *event_counter)
{
	for (int i = 0; i < EVENT_COUNTER_NAME_N; i++)
	{
		event_counter[i]++;
		// Increment the total number of events
		n_event[i]++;
	}
}

void pp_disk::set_event_counter(event_counter_name_t field, int value)
{
	n_hit_centrum[field] = value;
	n_ejection[field]    = value;
	n_collision[field]   = value;
	n_event[field]       = value;
}

void pp_disk::handle_collision()
{
	create_sp_events();

	// TODO: implement collision graph: bredth-first search
	for (unsigned int i = 0; i < sp_events.size(); i++)
	{
		handle_collision_pair(i, &sp_events[i]);
		increment_event_counter(n_collision);
	}
}

void pp_disk::handle_ejection_hit_centrum()
{
	sp_events.resize(event_counter);

	for (int i = 0; i < event_counter; i++)
	{
		// The events must be copied into sp_events since the print_event_data() write the content of the sp_events to disk.
		sp_events[i] = events[i];
		if (sp_events[i].event_name == EVENT_NAME_EJECTION)
		{
			increment_event_counter(n_ejection);
		}
		else
		{
			handle_collision_pair(i, &sp_events[i]);
			increment_event_counter(n_hit_centrum);
		}
	}
}

void pp_disk::create_sp_events()
{
	sp_events.resize(event_counter);

	bool *processed = new bool[event_counter];
	for (int i = 0; i < event_counter; i++)
	{
		processed[i] = false;
	}

	int n = 0;
	for (int k = 0; k < event_counter; k++)
	{
		if (processed[k] == false)
		{
			processed[k] = true;
			sp_events[n] = events[k];
		}
		else
		{
			continue;
		}
		for (int i = k + 1; i < event_counter; i++)
		{
			if (sp_events[n].id1 == events[i].id1 && sp_events[n].id2 == events[i].id2)
			{
				processed[i] = true;
				if (sp_events[n].t > events[i].t)
				{
					sp_events[n] = events[i];
				}
			}
		}
		n++;
	}
	delete[] processed;

	sp_events.resize(n);
}

void pp_disk::handle_collision_pair(int i, event_data_t *collision)
{
	int survivIdx = collision->idx1;
	int mergerIdx = collision->idx2;

	if (BODY_TYPE_SUPERPLANETESIMAL == sim_data->h_body_md[mergerIdx].body_type)
	{
		// TODO: implement collision between a body and a super-planetesimal
		throw string("Collision between a massive body and a super-planetesimal is not yet implemented.");
	}

	collision->p1 = sim_data->h_p[survivIdx];
	collision->p2 = sim_data->h_p[mergerIdx];

	// Calculate position and velocitiy of the new object
	tools::calc_position_after_collision(collision->p1.mass, collision->p2.mass, &(collision->r1), &(collision->r2), collision->rs);
	tools::calc_velocity_after_collision(collision->p1.mass, collision->p2.mass, &(collision->v1), &(collision->v2), collision->vs);
	// Update position and velocity of survivor
	sim_data->h_y[0][survivIdx] = collision->rs;
	sim_data->h_y[1][survivIdx] = collision->vs;

	// Calculate physical properties of the new object
	tools::calc_physical_properties(collision->p1, collision->p2, collision->ps);
	// Update physical properties of survivor
	sim_data->h_p[survivIdx] = collision->ps;

	// Make the merged body inactive 
	sim_data->h_body_md[mergerIdx].id *= -1;
	// Set its parameters to zero
	sim_data->h_p[mergerIdx].mass    = 0.0;
	sim_data->h_p[mergerIdx].density = 0.0;
	sim_data->h_p[mergerIdx].radius  = 0.0;
	// and push it radialy extremly far away with zero velocity
	sim_data->h_y[0][mergerIdx].x = 1.0e20;
	sim_data->h_y[0][mergerIdx].y = 0.0;
	sim_data->h_y[0][mergerIdx].z = 0.0;
	sim_data->h_y[1][mergerIdx].x = 0.0;
	sim_data->h_y[1][mergerIdx].y = 0.0;
	sim_data->h_y[1][mergerIdx].z = 0.0;

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		copy_vector_to_device((void **)&sim_data->d_y[0][survivIdx],	(void *)&sim_data->h_y[0][survivIdx],	 sizeof(vec_t));
		copy_vector_to_device((void **)&sim_data->d_y[1][survivIdx],	(void *)&sim_data->h_y[1][survivIdx],	 sizeof(vec_t));
		copy_vector_to_device((void **)&sim_data->d_p[survivIdx],		(void *)&sim_data->h_p[survivIdx],       sizeof(param_t));

		copy_vector_to_device((void **)&sim_data->d_y[0][mergerIdx],	(void *)&sim_data->h_y[0][mergerIdx],    sizeof(vec_t));
		copy_vector_to_device((void **)&sim_data->d_y[1][mergerIdx],	(void *)&sim_data->h_y[1][mergerIdx],    sizeof(vec_t));
		copy_vector_to_device((void **)&sim_data->d_p[mergerIdx],		(void *)&sim_data->h_p[mergerIdx],       sizeof(param_t));
		copy_vector_to_device((void **)&sim_data->d_body_md[mergerIdx],	(void *)&sim_data->h_body_md[mergerIdx], sizeof(body_metadata_t));
	}
}

pp_disk::pp_disk(string& path, int n_tpb, bool use_padded_storage, gas_disk_model_t g_disk_model, computing_device_t comp_dev) :
	n_tpb(n_tpb),
	use_padded_storage(use_padded_storage),
	g_disk_model(g_disk_model),
	comp_dev(comp_dev)
{
	initialize();
	n_bodies = get_number_of_bodies(path);
	allocate_storage();
	redutilcu::create_aliases(comp_dev, sim_data);
	load(path);
}

pp_disk::~pp_disk()
{
	deallocate_host_storage(sim_data);
	FREE_HOST_VECTOR((void **)&events);

	deallocate_device_storage(sim_data);
	FREE_DEVICE_VECTOR((void **)&d_events);
	FREE_DEVICE_VECTOR((void **)&d_event_counter);
	delete sim_data;

	//FREE_HOST_VECTOR(  (void **)&g_disk);
	//FREE_DEVICE_VECTOR((void **)&d_g_disk);
}

void pp_disk::initialize()
{
	t               = 0.0;
	sim_data        = 0x0;
	n_bodies        = 0x0;
	event_counter   = 0;
	d_event_counter = 0x0;
	events          = 0x0;
	d_events        = 0x0;

	for (int i = 0; i < EVENT_COUNTER_NAME_N; i++)
	{
		n_hit_centrum[i] = 0;
		n_ejection[i]    = 0;
		n_collision[i]   = 0;
		n_event[i]       = 0;
	}
}

void pp_disk::allocate_storage()
{
	int n_total = use_padded_storage ? n_bodies->get_n_prime_total() : n_bodies->get_n_total();

	sim_data = new sim_data_t;
	
	// These will be only aliases to the actual storage space either in the HOST or DEVICE memory
	sim_data->y.resize(2);
	sim_data->yout.resize(2);

	allocate_host_storage(sim_data, n_total);
	ALLOCATE_HOST_VECTOR((void **)&events, n_total*sizeof(event_data_t));

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		allocate_device_storage(sim_data, n_total);
		ALLOCATE_DEVICE_VECTOR((void **)&d_events,        n_total*sizeof(event_data_t));
		ALLOCATE_DEVICE_VECTOR((void **)&d_event_counter,       1*sizeof(int));
	}
}

void pp_disk::set_computing_device(computing_device_t device)
{
	// If the execution is already on the requested device than nothing to do
	if (this->comp_dev == device)
	{
		return;
	}

	int n_total = use_padded_storage ? n_bodies->get_n_prime_total() : n_bodies->get_n_total();

	switch (device)
	{
	case COMPUTING_DEVICE_CPU:
		copy_to_host();
		clear_event_counter();
		deallocate_device_storage(sim_data);
		FREE_DEVICE_VECTOR((void **)&d_events);
		FREE_DEVICE_VECTOR((void **)&d_event_counter);
		break;
	case COMPUTING_DEVICE_GPU:
		allocate_device_storage(sim_data, n_total);
		ALLOCATE_DEVICE_VECTOR((void **)&d_events,        n_total*sizeof(event_data_t));
		ALLOCATE_DEVICE_VECTOR((void **)&d_event_counter,       1*sizeof(int));

		copy_to_device();
		copy_disk_params_to_device();
		copy_constant_to_device(dc_threshold, this->threshold, THRESHOLD_N*sizeof(var_t));
		copy_vector_to_device((void *)d_event_counter, (void *)&event_counter, 1*sizeof(int));
		break;
	default:
		throw string ("Invalid parameter: computing device was out of range.");
	}
	redutilcu::create_aliases(device, sim_data);

	this->comp_dev = device;
}

number_of_bodies* pp_disk::get_number_of_bodies(string& path)
{
	int ns, ngp, nrp, npp, nspl, npl, ntp;

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
    return new number_of_bodies(ns, ngp, nrp, npp, nspl, npl, ntp, n_tpb, use_padded_storage);
}

void pp_disk::remove_inactive_bodies()
{
	int old_n_total = use_padded_storage ? n_bodies->get_n_prime_total() : n_bodies->get_n_total();
	// Update the numbers after counting the eliminated bodies
	n_bodies->update_numbers(sim_data->h_body_md);

	sim_data_t *sim_data_temp = new sim_data_t;
	// Only the data of the active bodies will be temporarily stored
	allocate_host_storage(sim_data_temp, n_bodies->get_n_total());

	// Copy the data of active bodies to sim_data_temp
	int i = 0;
	int k = 0;
	for ( ; i < old_n_total; i++)
	{
		if (0 < sim_data->h_body_md[i].id && BODY_TYPE_PADDINGPARTICLE > sim_data->h_body_md[i].body_type)
		{
			sim_data_temp->h_y[0][k]    = sim_data->h_y[0][i];
			sim_data_temp->h_y[1][k]    = sim_data->h_y[1][i];
			sim_data_temp->h_p[k]       = sim_data->h_p[i];
			sim_data_temp->h_body_md[k] = sim_data->h_body_md[i];
			k++;
		}
	}
	if (n_bodies->get_n_total() != k)
	{
		throw string("Error: number of copied bodies does not equal to the number of active bodies.");
	}

	int n_SI		= n_bodies->get_n_SI();
	int n_NSI		= n_bodies->get_n_NSI();
	int n_total		= n_bodies->get_n_total();
	int n_prime_SI	= n_bodies->get_n_prime_SI();
	int n_prime_NSI	= n_bodies->get_n_prime_NSI();
	int n_prime_total=n_bodies->get_n_prime_total();

	k = 0;
	i = 0;
	for ( ; i < n_SI; i++, k++)
	{
		sim_data->h_y[0][k]    = sim_data_temp->h_y[0][i];
		sim_data->h_y[1][k]    = sim_data_temp->h_y[1][i];
		sim_data->h_p[k]       = sim_data_temp->h_p[i];
		sim_data->h_body_md[k] = sim_data_temp->h_body_md[i];
	}
    while (use_padded_storage && k < n_prime_SI)
    {
		create_padding_particle(k, sim_data->h_epoch, sim_data->h_body_md, sim_data->h_p, sim_data->h_y[0], sim_data->h_y[1]);
        k++;
    }

	for ( ; i < n_SI + n_NSI; i++, k++)
	{
		sim_data->h_y[0][k]    = sim_data_temp->h_y[0][i];
		sim_data->h_y[1][k]    = sim_data_temp->h_y[1][i];
		sim_data->h_p[k]       = sim_data_temp->h_p[i];
		sim_data->h_body_md[k] = sim_data_temp->h_body_md[i];
	}
    while (use_padded_storage && k < n_prime_SI + n_prime_NSI)
    {
		create_padding_particle(k, sim_data->h_epoch, sim_data->h_body_md, sim_data->h_p, sim_data->h_y[0], sim_data->h_y[1]);
        k++;
    }

	for ( ; i < n_total; i++, k++)
	{
		sim_data->h_y[0][k]    = sim_data_temp->h_y[0][i];
		sim_data->h_y[1][k]    = sim_data_temp->h_y[1][i];
		sim_data->h_p[k]       = sim_data_temp->h_p[i];
		sim_data->h_body_md[k] = sim_data_temp->h_body_md[i];
	}
    while (use_padded_storage && k < n_prime_total)
    {
		create_padding_particle(k, sim_data->h_epoch, sim_data->h_body_md, sim_data->h_p, sim_data->h_y[0], sim_data->h_y[1]);
        k++;
    }

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		// Copy the active bodies to the device
		copy_to_device();
	}
	deallocate_host_storage(sim_data_temp);

	delete sim_data_temp;
}

void pp_disk::copy_to_device()
{
	int n_body = use_padded_storage ? n_bodies->get_n_prime_total() : n_bodies->get_n_total();

	for (int i = 0; i < 2; i++)
	{
		copy_vector_to_device((void *)sim_data->d_y[i],	(void *)sim_data->h_y[i],	 n_body*sizeof(vec_t));
	}
	copy_vector_to_device((void *)sim_data->d_p,		(void *)sim_data->h_p,		 n_body*sizeof(param_t));
	copy_vector_to_device((void *)sim_data->d_body_md,	(void *)sim_data->h_body_md, n_body*sizeof(body_metadata_t));
	copy_vector_to_device((void *)sim_data->d_epoch,	(void *)sim_data->h_epoch,	 n_body*sizeof(ttt_t));

	copy_vector_to_device((void *)d_event_counter,		(void *)&event_counter,		      1*sizeof(int));
}

void pp_disk::copy_to_host()
{
	int n_body = use_padded_storage ? n_bodies->get_n_prime_total() : n_bodies->get_n_total();

	for (int i = 0; i < 2; i++)
	{
		copy_vector_to_host((void *)sim_data->h_y[i],	(void *)sim_data->d_y[i],	 n_body*sizeof(vec_t));
	}
	copy_vector_to_host((void *)sim_data->h_p,			(void *)sim_data->d_p,		 n_body*sizeof(param_t));
	copy_vector_to_host((void *)sim_data->h_body_md,	(void *)sim_data->d_body_md, n_body*sizeof(body_metadata_t));
	copy_vector_to_host((void *)sim_data->h_epoch,		(void *)sim_data->d_epoch,	 n_body*sizeof(ttt_t));

	copy_vector_to_host((void *)&event_counter,			(void *)d_event_counter,	      1*sizeof(int));
}

void pp_disk::copy_threshold(const var_t* thrshld)
{
	memcpy(threshold, thrshld, THRESHOLD_N * sizeof(var_t));

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		copy_constant_to_device(dc_threshold, thrshld, THRESHOLD_N*sizeof(var_t));
	}
}

void pp_disk::copy_disk_params_to_device()
{
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		switch (g_disk_model)
		{
		case GAS_DISK_MODEL_NONE:
			break;
		case GAS_DISK_MODEL_ANALYTIC:
			copy_constant_to_device((void*)&dc_anal_gd_params,  (void*)&(this->a_gd->params), sizeof(analytic_gas_disk_params_t));
			break;
		case GAS_DISK_MODEL_FARGO:
			copy_constant_to_device((void*)&dc_fargo_gd_params, (void*)&(this->f_gd->params), sizeof(fargo_gas_disk_params_t));
			break;
		}
	}
}

void pp_disk::copy_event_data_to_host()
{
	copy_vector_to_host((void *)events, (void *)d_events, event_counter*sizeof(event_data_t));
}

int pp_disk::get_n_event()
{
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		copy_vector_to_host((void *)&event_counter, (void *)d_event_counter, 1*sizeof(int));
	}

	return event_counter;
}

void pp_disk::clear_event_counter()
{
	event_counter = 0;
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		copy_vector_to_device((void *)d_event_counter, (void *)&event_counter, 1*sizeof(int));
	}
}

int pp_disk::get_n_total_event()
{
	return (n_collision[EVENT_COUNTER_NAME_TOTAL] + n_ejection[EVENT_COUNTER_NAME_TOTAL] + n_hit_centrum[EVENT_COUNTER_NAME_TOTAL]);
}

var_t pp_disk::get_mass_of_star()
{
	int n = use_padded_storage ? n_bodies->get_n_prime_massive() : n_bodies->get_n_massive();

	body_metadata_t* body_md = sim_data->h_body_md;
	for (int j = 0; j < n; j++ )
	{
		if (body_md[j].body_type == BODY_TYPE_STAR)
		{
			return sim_data->h_p[j].mass;
		}
	}
	throw string("No star is included!");
}

void pp_disk::transform_to_bc(bool verbose)
{
	int n = use_padded_storage ? n_bodies->get_n_prime_total() : n_bodies->get_n_total();

	tools::transform_to_bc(n, verbose, sim_data);
}

void pp_disk::transform_time(bool verbose)
{
	if (verbose)
	{
		cout << "Transforming the time ... ";
	}

	// Transform the bodies' epochs
	int n = use_padded_storage ? n_bodies->get_n_prime_total() : n_bodies->get_n_total();
	for (int j = 0; j < n; j++ )
	{
		sim_data->h_epoch[j] *= constants::Gauss;
	}

	if (verbose)
	{
		cout << "done" << endl;
	}
}

void pp_disk::transform_velocity(bool verbose)
{
	if (verbose)
	{
		cout << "Transforming the velocity ... ";
	}

	vec_t* v = sim_data->h_y[1];
	// Transform the bodies' velocities
	int n = use_padded_storage ? n_bodies->get_n_prime_total() : n_bodies->get_n_total();
	for (int j = 0; j < n; j++ )
	{
		v[j].x /= constants::Gauss;		v[j].y /= constants::Gauss;		v[j].z /= constants::Gauss;
	}

	if (verbose)
	{
		cout << "done" << endl;
	}
}

void pp_disk::create_padding_particle(int k, ttt_t* epoch, body_metadata_t* body_md, param_t* p, vec_t* r, vec_t* v)
{
	body_md[k].id = 0;
	body_names.push_back("Pad_Part");

	body_md[k].body_type = static_cast<body_type_t>(BODY_TYPE_PADDINGPARTICLE);
	epoch[k] = 0.0;
	p[k].mass = 0.0;
	p[k].radius = 0.0;
	p[k].density = 0.0;
	p[k].cd = 0.0;

	body_md[k].mig_type = static_cast<migration_type_t>(MIGRATION_TYPE_NO);
	body_md[k].mig_stop_at = 0.0;

	r[k].x = 1.0e9 + (var_t)rand() / RAND_MAX * 1.0e9;
	r[k].y = r[k].x + (var_t)rand() / RAND_MAX * 1.0e9;
	r[k].z = 0.0;
	r[k].w = 0.0;

	v[k].x = v[k].y = v[k].z = v[k].w = 0.0;
}

void pp_disk::read_body_record(ifstream& input, int k, ttt_t* epoch, body_metadata_t* body_md, param_t* p, vec_t* r, vec_t* v)
{
	int_t	type = 0;
	string	dummy;

	// id
	input >> body_md[k].id;
	// name
	input >> dummy;
	// The names must be less than or equal to 30 chars
	if (dummy.length() > 30)
	{
		dummy = dummy.substr(0, 30);
	}
	body_names.push_back(dummy);
	// body type
	input >> type;
	body_md[k].body_type = static_cast<body_type_t>(type);
	// epoch
	input >> epoch[k];

	// mass, radius density and stokes coefficient
	input >> p[k].mass >> p[k].radius >> p[k].density >> p[k].cd;

	// migration type
	input >> type;
	body_md[k].mig_type = static_cast<migration_type_t>(type);
	// migration stop at
	input >> body_md[k].mig_stop_at;

	// position
	input >> r[k].x >> r[k].y >> r[k].z;
	// velocity
	input >> v[k].x >> v[k].y >> v[k].z;

	r[k].w = v[k].w = 0.0;
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

	vec_t* r = sim_data->h_y[0];
	vec_t* v = sim_data->h_y[1];
	param_t* p = sim_data->h_p;
	body_metadata_t* body_md = sim_data->h_body_md;
	ttt_t* epoch = sim_data->h_epoch;

	int n_SI		= n_bodies->get_n_SI();
	int n_NSI		= n_bodies->get_n_NSI();
	int n_total		= n_bodies->get_n_total();
	int n_prime_SI	= n_bodies->get_n_prime_SI();
	int n_prime_NSI	= n_bodies->get_n_prime_NSI();
	int n_prime_total=n_bodies->get_n_prime_total(); 

    if (input) 
    {
		int i = 0;
		int k = 0;
		for ( ; i < n_SI; i++, k++)
		{
			read_body_record(input, k, epoch, body_md, p, r, v);
		}
        while (use_padded_storage && k < n_prime_SI)
        {
			create_padding_particle(k, epoch, body_md, p, r, v);
            k++;
        }

		for ( ; i < n_SI + n_NSI; i++, k++)
		{
			read_body_record(input, k, epoch, body_md, p, r, v);
		}
		while (use_padded_storage && k < n_prime_SI + n_prime_NSI)
		{
			create_padding_particle(k, epoch, body_md, p, r, v);
			k++;
		}

		for ( ; i < n_total; i++, k++)
		{
			read_body_record(input, k, epoch, body_md, p, r, v);
		}
		while (use_padded_storage && k < n_prime_total)
		{
			create_padding_particle(k, epoch, body_md, p, r, v);
			k++;
		}
        input.close();
	}
	else
    {
		throw string("Cannot open " + path + ".");
	}

	cout << "done" << endl;
}

void pp_disk::print_result_ascii(ostream& sout)
{
	static int int_t_w  =  8;
	static int var_t_w  = 25;

	sout.precision(16);
	sout.setf(ios::right);
	sout.setf(ios::scientific);

	vec_t* r = sim_data->h_y[0];
	vec_t* v = sim_data->h_y[1];
	param_t* p = sim_data->h_p;
	body_metadata_t* body_md = sim_data->h_body_md;

	int n = use_padded_storage ? n_bodies->get_n_prime_total() : n_bodies->get_n_total();
	for (int i = 0; i < n; i++)
    {
		// Skip inactive bodies and padding particles and alike
		if (body_md[i].id <= 0 || body_md[i].body_type >= BODY_TYPE_PADDINGPARTICLE)
		{
			continue;
		}
		sout << setw(int_t_w) << body_md[i].id << SEP                /* id of the body starting from 1                                (int)              */
			 << setw(     30) << body_names[i] << SEP                /* name of the body                                              (string = 30 char) */ 
			 << setw(      2) << body_md[i].body_type << SEP         /* type of the body                                              (int)              */
			 << setw(var_t_w) << t / constants::Gauss << SEP         /* time of the record                           [day]            (double)           */
			 << setw(var_t_w) << p[i].mass << SEP                    /* mass of the body                             [solar mass]     (double)           */
			 << setw(var_t_w) << p[i].radius << SEP                  /* radius of the body                           [AU]             (double)           */
			 << setw(var_t_w) << p[i].density << SEP                 /* density of the body in                       [solar mass/AU3] (double)           */
			 << setw(var_t_w) << p[i].cd << SEP                      /* Stokes drag coefficeint dimensionless                         (double)           */
			 << setw(      2) << body_md[i].mig_type << SEP          /* migration type of the body                                    (int)              */
			 << setw(var_t_w) << body_md[i].mig_stop_at << SEP       /* migration stops at this barycentric distance [AU]             (double)           */
			 << setw(var_t_w) << r[i].x << SEP                       /* body's x-coordiante in barycentric system    [AU]             (double)           */
			 << setw(var_t_w) << r[i].y << SEP                       /* body's y-coordiante in barycentric system    [AU]             (double)           */
			 << setw(var_t_w) << r[i].z << SEP                       /* body's z-coordiante in barycentric system    [AU]             (double)           */
			 << setw(var_t_w) << v[i].x * constants::Gauss << SEP    /* body's x-velocity in baryentric system       [AU/day]         (double)           */
			 << setw(var_t_w) << v[i].y * constants::Gauss << SEP    /* body's y-velocity in barycentric system      [AU/day]         (double)           */
			 << setw(var_t_w) << v[i].z * constants::Gauss << endl;  /* body's z-velocity in barycentric system      [AU/day]         (double)           */
    }
	sout.flush();
}

void pp_disk::print_result_binary(ostream& sout)
{
	throw string("print_result_binary() is not implemented");
}

void pp_disk::print_event_data(ostream& sout, ostream& log_f)
{
	static int int_t_w =  8;
	static int var_t_w = 25;
	static char *e_names[] = {"NONE", "HIT_CENTRUM", "EJECTION", "CLOSE_ENCOUNTER", "COLLISION"};

	sout.precision(16);
	sout.setf(ios::right);
	sout.setf(ios::scientific);

	log_f.precision(16);
	log_f.setf(ios::right);
	log_f.setf(ios::scientific);

	for (unsigned int i = 0; i < sp_events.size(); i++)
	{
		sout << setw(16)      << e_names[sp_events[i].event_name] << SEP
			 << setw(var_t_w) << sp_events[i].t / constants::Gauss << SEP /* time of the event in day */
			 << setw(var_t_w) << sp_events[i].d << SEP
			 << setw(int_t_w) << sp_events[i].id1 << SEP		/* id of the survivor */
			 << setw(int_t_w) << sp_events[i].id2 << SEP		/* id of the merger */
			 << setw(var_t_w) << sp_events[i].p1.mass << SEP	/* parameters of the survivor before the event */
			 << setw(var_t_w) << sp_events[i].p1.density << SEP
			 << setw(var_t_w) << sp_events[i].p1.radius << SEP
			 << setw(var_t_w) << sp_events[i].r1.x << SEP		/* position of the survivor before the event */
			 << setw(var_t_w) << sp_events[i].r1.y << SEP
			 << setw(var_t_w) << sp_events[i].r1.z << SEP
			 << setw(var_t_w) << sp_events[i].v1.x * constants::Gauss << SEP		/* velocity of the survivor before the event */
			 << setw(var_t_w) << sp_events[i].v1.y * constants::Gauss << SEP
			 << setw(var_t_w) << sp_events[i].v1.z * constants::Gauss << SEP
			 << setw(var_t_w) << sp_events[i].p2.mass << SEP	/* parameters of the merger before the event */
			 << setw(var_t_w) << sp_events[i].p2.density << SEP
			 << setw(var_t_w) << sp_events[i].p2.radius << SEP
			 << setw(var_t_w) << sp_events[i].r2.x << SEP		/* position of the merger before the event */
			 << setw(var_t_w) << sp_events[i].r2.y << SEP
			 << setw(var_t_w) << sp_events[i].r2.z << SEP
			 << setw(var_t_w) << sp_events[i].v2.x * constants::Gauss << SEP		/* velocity of the merger before the event */
			 << setw(var_t_w) << sp_events[i].v2.y * constants::Gauss<< SEP
			 << setw(var_t_w) << sp_events[i].v2.z * constants::Gauss<< SEP
			 << setw(var_t_w) << sp_events[i].ps.mass << SEP	/* parameters of the survivor after the event */
			 << setw(var_t_w) << sp_events[i].ps.density << SEP
			 << setw(var_t_w) << sp_events[i].ps.radius << SEP
			 << setw(var_t_w) << sp_events[i].rs.x << SEP		/* position of the survivor after the event */
			 << setw(var_t_w) << sp_events[i].rs.y << SEP
			 << setw(var_t_w) << sp_events[i].rs.z << SEP
			 << setw(var_t_w) << sp_events[i].vs.x * constants::Gauss << SEP		/* velocity of the survivor after the event */
			 << setw(var_t_w) << sp_events[i].vs.y * constants::Gauss << SEP
			 << setw(var_t_w) << sp_events[i].vs.z * constants::Gauss << SEP << endl;
		if (log_f)
		{
			log_f << tools::get_time_stamp() << SEP 
				  << e_names[sp_events[i].event_name] << SEP
			      << setw(int_t_w) << sp_events[i].id1 << SEP
			      << setw(int_t_w) << sp_events[i].id2 << SEP << endl;
		}
	}
	sout.flush();
	log_f.flush();
}

#undef GAS_REDUCTION_THRESHOLD
#undef GAS_INNER_EDGE
