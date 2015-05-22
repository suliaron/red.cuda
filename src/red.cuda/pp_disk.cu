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

#define GAS_REDUCTION_THRESHOLD    1.0e-6
#define GAS_INNER_EDGE             0.1    // [AU]


__constant__ var_t                      dc_threshold[THRESHOLD_N];
__constant__ analytic_gas_disk_params_t dc_anal_gd_params;
__constant__ fargo_gas_disk_params_t    dc_fargo_gd_params;

///****************** DEVICE functions begins here ******************/

/****************** KERNEL functions begins here ******************/

namespace pp_disk_utility
{

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

__host__ __device__
	int calc_linear_index(const vec_t& rVec, var_t* used_rad, int n_sec, int n_rad)
{
	const var_t dalpha = TWOPI / n_sec;
	
	var_t r = sqrt(SQR(rVec.x) + SQR(rVec.y));
	if (     used_rad[0] > r)
	{
		return 0;
	}
	else if (used_rad[n_rad] < r)
	{
		return n_rad * n_sec - 1;
	}
	else
	{
		// TODO: implement a fast search for the cell
		// IDEA: populate the used_rad with the square of the distance, since it is much faster to calculate r^2
		// Determine which ring contains r
		int i_rad = 0;
		int i_sec = 0;
		for (int k = 0; k < n_rad; k++)
		{
			if (used_rad[k] <= r && r < used_rad[k+1])
			{
				i_rad = k;
				break;
			}
		}

		var_t alpha = (rVec.y >= 0.0 ? atan2(rVec.y, rVec.x) : TWOPI + atan2(rVec.y, rVec.x));
		i_sec =  (int)(alpha / dalpha);
		int i_linear = i_rad * n_sec + i_sec;

		return i_linear;
	}
}

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

	switch (name)
	{
	case EVENT_NAME_COLLISION:
		printf("COLLISION   t = %20.10le [d] d = %20.10le [AU] ids: %5d %5d\n", t/K, d, body_md[idx1].id, body_md[idx2].id);
		break;
	case EVENT_NAME_EJECTION:
		printf("EJECTION    t = %20.10le [d] d = %20.10le [AU] ids: %5d %5d\n", t/K, d, body_md[idx1].id, body_md[idx2].id);
		break;
	case EVENT_NAME_HIT_CENTRUM:
		printf("HIT CENTRUM t = %20.10le [d] d = %20.10le [AU] ids: %5d %5d\n", t/K, d, body_md[idx1].id, body_md[idx2].id);
		break;
	}
}

} /* pp_disk_utility */

namespace kernel_pp_disk
{
static __global__
	void check_for_ejection_hit_centrum
	(
		ttt_t curr_t, 
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
		if (     dc_threshold[THRESHOLD_EJECTION_DISTANCE_SQUARED] < r2)
		{
			k = atomicAdd(event_counter, 1);
			pp_disk_utility::store_event_data(EVENT_NAME_EJECTION, curr_t, sqrt(r2), 0, i, p, r, v, body_md, &events[k]);
		}
		else if (dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE_SQUARED] > r2)
		{
			k = atomicAdd(event_counter, 1);
			pp_disk_utility::store_event_data(EVENT_NAME_HIT_CENTRUM, curr_t, sqrt(r2), 0, i, p, r, v, body_md, &events[k]);
		}
	}
}

static __global__
	void check_for_collision
	(
		ttt_t curr_t, 
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
	const double f = SQR(dc_threshold[THRESHOLD_RADII_ENHANCE_FACTOR]);

	if (i < int_bound.sink.y && 0 < body_md[i].id)
	{
		vec_t dVec = {0.0, 0.0, 0.0, 0.0};
		for (int j = i + 1; j < int_bound.source.y; j++) 
		{
			/* Skip inactive bodies, i.e. id < 0 */
			if (0 > body_md[j].id)
			{
				continue;
			}
			// 3 FLOP
			dVec.x = r[j].x - r[i].x;
			dVec.y = r[j].y - r[i].y;
			dVec.z = r[j].z - r[i].z;
			// 5 FLOP
			dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2

			// The data of the collision will be stored for the body with the greater index (test particles can collide with massive bodies)
			// If i < j is the condition than test particles can not collide with massive bodies
			if (dVec.w < f * SQR((p[i].radius + p[j].radius)))
			{
				unsigned int k = atomicAdd(event_counter, 1);

				int survivIdx = i;
				int mergerIdx = j;
				if (p[mergerIdx].mass > p[survivIdx].mass)
				{
					int idx = survivIdx;
					survivIdx = mergerIdx;
					mergerIdx = idx;
				}
				pp_disk_utility::store_event_data(EVENT_NAME_COLLISION, curr_t, sqrt(dVec.w), survivIdx, mergerIdx, p, r, v, body_md, &events[k]);
			}
		}
	}
}

static __global__
	void calc_grav_accel_int_mul_of_thread_per_block
	(
		ttt_t curr_t, 
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

		//if (i > 0 && i > j && d < dc_threshold[THRESHOLD_RADII_ENHANCE_FACTOR] * (p[i].radius + p[j].radius))
		//{
		//	unsigned int k = atomicAdd(event_counter, 1);

		//	int survivIdx = i;
		//	int mergerIdx = j;
		//	if (p[mergerIdx].mass > p[survivIdx].mass)
		//	{
		//		int m = survivIdx;
		//		survivIdx = mergerIdx;
		//		mergerIdx = m;
		//	}
		//	pp_disk_utility::store_event_data(EVENT_NAME_COLLISION, curr_t, d, survivIdx, mergerIdx, p, r, v, body_md, &events[k]);
		//}
	}
}

static __global__
	void calc_grav_accel
	(
		ttt_t curr_t, 
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
				//if (0 < i && i > j && d < dc_threshold[THRESHOLD_RADII_ENHANCE_FACTOR] * (p[i].radius + p[j].radius))
				//{
				//	unsigned int k = atomicAdd(event_counter, 1);

				//	int survivIdx = i;
				//	int mergerIdx = j;
				//	if (p[mergerIdx].mass > p[survivIdx].mass)
				//	{
				//		int t = survivIdx;
				//		survivIdx = mergerIdx;
				//		mergerIdx = t;
				//	}
				//	pp_disk_utility::store_event_data(EVENT_NAME_COLLISION, curr_t, d, survivIdx, mergerIdx, p, r, v, body_md, &events[k]);
				//}
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

	// TODO: decr_fact should be a parameter to the function
	var_t decr_fact = pp_disk_utility::reduction_factor(dc_anal_gd_params.gas_decrease, dc_anal_gd_params.t0, dc_anal_gd_params.t1, dc_anal_gd_params.e_folding_time, curr_t);
	// TODO: export 1.0e-6 into the gas disk description file
	if (GAS_REDUCTION_THRESHOLD > decr_fact)
	{
		return;
	}

	if (i < int_bound.sink.y)
	{
		var_t m_star = p[0].mass;
		vec_t v_g       = pp_disk_utility::get_velocity(m_star, dc_anal_gd_params.eta, &r[i]);
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
	const int n_total = get_n_total_body();

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
		interaction_bound int_bound = n_bodies->get_bound_GD(ups, n_tpb);
		cpu_calc_drag_accel_NSI(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy);
	}
}

void pp_disk::cpu_calc_drag_accel_NSI(ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a)
{
	switch (g_disk_model)
	{
	case GAS_DISK_MODEL_NONE:
		break;
	case GAS_DISK_MODEL_ANALYTIC:
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
			vec_t v_g       = pp_disk_utility::get_velocity(m_star, a_gd->params.eta, &r[i]);
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
		} /* for */
		} /* case block */
		break;
	case GAS_DISK_MODEL_FARGO:
		{
		var_t m_star = p[0].mass;
		for (int i = int_bound.sink.x; i < int_bound.sink.y; i++)
		{
			int linear_index= pp_disk_utility::calc_linear_index(r[i], f_gd->used_rad[0], f_gd->params.n_sec,  f_gd->params.n_rad);
			var_t r_norm    = sqrt(SQR(r[i].x) + SQR(r[i].y));

			// Massless body's circular velocity at r_norm distance from the barycenter
			var_t vc_theta  = sqrt(m_star/r_norm);

			// Get gas parcel's velocity at r[i]
			var_t v_g_theta = vc_theta + f_gd->vtheta[0][linear_index];
			var_t v_g_rad   = f_gd->vrad[0][linear_index];
			vec_t v_g_polar = {v_g_rad, v_g_theta, 0.0, 0.0};
			
			// Calculate the angle between the r position vector and the x-axis
			var_t theta     = (r[i].y >= 0.0 ? atan2(r[i].y, r[i].x) : TWOPI + atan2(r[i].y, r[i].x));

			// TODO: calculate the x and y components of the gas velocity
			vec_t v_g       = redutilcu::rotate_2D_vector(theta, v_g_polar);
			
			// Get gas parcel's density at r[i]
			var_t density_g = f_gd->density[0][linear_index];

			// Compute the solid body's relative velocity
			vec_t u         = {v_g.x - v[i].x, v_g.y - v[i].y, v_g.z - v[i].z, 0.0};
			var_t u_n       = sqrt(SQR(u.x) + SQR(u.y) + SQR(u.z));

			var_t f = (3.0 * p[i].cd * density_g * u_n) / (8.0 * p[i].radius * p[i].density);

			a[i].x += f * u.x;
			a[i].y += f * u.y;
			a[i].z += f * u.z;
		} /* for */
		} /* case block */
		break;
	default:
		throw string("Parameter 'g_disk_model' is out of range.");
	} /* switch */
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
				//if (i > 0 && i > j && d < threshold[THRESHOLD_RADII_ENHANCE_FACTOR] * (p[i].radius + p[j].radius))
				//{
				//	int k = *event_counter;

				//	int survivIdx = i;
				//	int mergerIdx = j;
				//	if (p[mergerIdx].mass > p[survivIdx].mass)
				//	{
				//		int t = survivIdx;
				//		survivIdx = mergerIdx;
				//		mergerIdx = t;
				//	}
				//	//printf("t = %20.10le d = %20.10le %d. COLLISION detected: id: %5d id: %5d\n", curr_t / constants::Gauss, d, k+1, body_md[survivIdx].id, body_md[mergerIdx].id);
				//	pp_disk_utility::store_event_data(EVENT_NAME_COLLISION, curr_t, d, survivIdx, mergerIdx, p, r, v, body_md, &events[k]);

				//	(*event_counter)++;
				//}
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
		interaction_bound int_bound = n_bodies->get_bound_SI(ups, n_tpb);
		cpu_calc_grav_accel_SI(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy, events, &event_counter);
	}

	n_sink = n_bodies->get_n_NSI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_NSI(ups, n_tpb);
		cpu_calc_grav_accel_NSI(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy, events, &event_counter);
	}

	n_sink = n_bodies->get_n_NI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_NI(ups, n_tpb);
		cpu_calc_grav_accel_NI(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy, events, &event_counter);
	}
}

float pp_disk::wrapper_kernel_pp_disk_calc_grav_accel(ttt_t curr_t, int n_sink, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const vec_t* r, const vec_t* v, vec_t* a)
{
	cudaError_t cudaStatus = cudaSuccess;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw nbody_exception("cudaEventCreate failed", cudaStatus);
	}

	set_kernel_launch_param(n_sink);

	cudaEventRecord(start, 0);
	kernel_pp_disk::calc_grav_accel<<<grid, block>>>(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, a, d_events, d_event_counter);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw nbody_exception("kernel_pp_disk::calc_grav_accel failed", cudaStatus);
	}

	float elapsed_time = 0.0f;
	cudaEventElapsedTime(&elapsed_time, start, stop);

	return elapsed_time;
}

void pp_disk::call_kernel_calc_grav_accel(ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy)
{
	cudaError_t cudaStatus = cudaSuccess;
	
	int n_sink = ups ? n_bodies->get_n_prime_SI(n_tpb) : n_bodies->get_n_SI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_SI(ups, n_tpb);
		set_kernel_launch_param(n_sink);

		if (ups)
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

	n_sink = ups ? n_bodies->get_n_prime_NSI(n_tpb) : n_bodies->get_n_NSI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_NSI(ups, n_tpb);
		set_kernel_launch_param(n_sink);

		if (ups)
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

	n_sink = ups ? n_bodies->get_n_prime_NI(n_tpb) : n_bodies->get_n_NI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_NI(ups, n_tpb);
		set_kernel_launch_param(n_sink);

		if (ups)
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
		interaction_bound int_bound = n_bodies->get_bound_GD(ups, n_tpb);
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
	default:
		throw string("Parameter 'comp_dev' is out of range.");
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
	//int n_event = get_n_event();

	int n_event = 0;
	switch (comp_dev)
	{
	case COMPUTING_DEVICE_CPU:
		n_event = cpu_check_for_collision();
		break;
	case COMPUTING_DEVICE_GPU:
		n_event = call_kernel_check_for_collision();
		break;
	default:
		throw string("Parameter 'comp_dev' is out of range.");
	}

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

int pp_disk::cpu_check_for_ejection_hit_centrum()
{
	int n_total = n_bodies->get_n_total_playing();
	interaction_bound int_bound(0, n_total, 0, 0);

	const vec_t* r = sim_data->y[0];
	const vec_t* v = sim_data->y[1];
	const param_t* p = sim_data->h_p;
	body_metadata_t* body_md = sim_data->body_md;
	
	for (int i = int_bound.sink.x; i < int_bound.sink.y; i++)
	{
		// Ignore the star and the inactive bodies (whose id < 0)
		if (0 < sim_data->body_md[i].id && BODY_TYPE_STAR != sim_data->body_md[i].body_type)
		{
			// Calculate the distance from the barycenter
			var_t r2 = SQR(r[i].x) + SQR(r[i].y) + SQR(r[i].z);
			if (     threshold[THRESHOLD_EJECTION_DISTANCE_SQUARED] < r2)
			{
				pp_disk_utility::store_event_data(EVENT_NAME_EJECTION, t, sqrt(r2), 0, i, p, r, v, body_md, &events[event_counter]);
				event_counter++;
			}
			else if (threshold[THRESHOLD_HIT_CENTRUM_DISTANCE_SQUARED] > r2)
			{
				pp_disk_utility::store_event_data(EVENT_NAME_HIT_CENTRUM, t, sqrt(r2), 0, i, p, r, v, body_md, &events[event_counter]);
				event_counter++;
			}
		}
	}

	return event_counter;
}

int pp_disk::cpu_check_for_collision()
{
	int n_total = n_bodies->get_n_total_playing();
	interaction_bound int_bound(0, n_total, 0, n_total);

	int n_event = cpu_check_for_collision(int_bound, false, false, false, false);

	return n_event;
}

int pp_disk::cpu_check_for_collision(interaction_bound int_bound, bool SI_NSI, bool SI_TP, bool NSI, bool NSI_TP)
{
	const double f = SQR(threshold[THRESHOLD_RADII_ENHANCE_FACTOR]);

	const vec_t* r = sim_data->y[0];
	const vec_t* v = sim_data->y[1];
	const param_t* p = sim_data->h_p;
	body_metadata_t* body_md = sim_data->body_md;

	for (int i = int_bound.sink.x; i < int_bound.sink.y; i++)
	{
		if (0 < body_md[i].id)
		{
			vec_t dVec = {0.0, 0.0, 0.0, 0.0};
			for (int j = i + 1; j < int_bound.source.y; j++) 
			{
				/* Skip inactive bodies, i.e. id < 0 */
				if (0 > body_md[j].id)
				{
					continue;
				}
				// 3 FLOP
				dVec.x = r[j].x - r[i].x;
				dVec.y = r[j].y - r[i].y;
				dVec.z = r[j].z - r[i].z;
				// 5 FLOP
				dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2

				// The data of the collision will be stored for the body with the greater index (test particles can collide with massive bodies)
				// If i < j is the condition than test particles can not collide with massive bodies
				if (dVec.w < f * SQR((p[i].radius + p[j].radius)))
				{
					int survivIdx = i;
					int mergerIdx = j;
					if (p[mergerIdx].mass > p[survivIdx].mass)
					{
						int idx = survivIdx;
						survivIdx = mergerIdx;
						mergerIdx = idx;
					}
					pp_disk_utility::store_event_data(EVENT_NAME_COLLISION, t, sqrt(dVec.w), survivIdx, mergerIdx, p, r, v, body_md, &events[event_counter]);
					event_counter++;
				}
			}
		}
	}

	return event_counter;
}

int pp_disk::call_kernel_check_for_ejection_hit_centrum()
{
	cudaError_t cudaStatus = cudaSuccess;
	
	int n_total = get_n_total_body();
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

int pp_disk::call_kernel_check_for_collision()
{
	cudaError_t cudaStatus = cudaSuccess;
	
	int n_total = get_n_total_body();
	interaction_bound int_bound(0, n_total, 0, n_total);
	set_kernel_launch_param(n_total);

	kernel_pp_disk::check_for_collision<<<grid, block>>>
		(t, int_bound, sim_data->p, sim_data->y[0], sim_data->y[1], sim_data->body_md, d_events, d_event_counter);

	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw nbody_exception("kernel_pp_disk::check_for_collision failed", cudaStatus);
	}

	return get_n_event();
}

void pp_disk::calc_dydx(int i, int rr, ttt_t curr_t, const vec_t* r, const vec_t* v, vec_t* dy)
{
	cudaError_t cudaStatus = cudaSuccess;

	const int n_total = get_n_total_body();

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
	default:
		throw string("Parameter 'i' is out of range.");
	}
}

void pp_disk::swap()
{
	for (int i = 0; i < 2; i++)
	{
		::swap(sim_data->yout[i], sim_data->y[i]);
	}

	int tmp = (int)aps + 1;
	if (ACTUAL_PHASE_STORAGE_N > tmp)
	{
		aps = (actual_phase_storage_t)tmp;
	}
	else
	{
		aps = ACTUAL_PHASE_STORAGE_Y;
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
		int ictv_idx = sp_events[i].idx2;

		handle_collision_pair(i, &sp_events[i]);
		increment_event_counter(n_collision);
		// Make the merged body inactive 
		sim_data->h_body_md[ictv_idx].id *= -1;
		// Copy it up to GPU 
		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			copy_vector_to_device((void **)&sim_data->d_body_md[ictv_idx], (void *)&sim_data->h_body_md[ictv_idx], sizeof(body_metadata_t));
		}
		// Update number of inactive bodies
		n_bodies->inactive[sim_data->h_body_md[ictv_idx].body_type]++;
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

void pp_disk::handle_ejection_hit_centrum()
{
	sp_events.resize(event_counter);

	for (int i = 0; i < event_counter; i++)
	{
		// The events must be copied into sp_events since the print_event_data() write the content of the sp_events to disk.
		sp_events[i] = events[i];
		int ictv_idx = sp_events[i].idx2;
		if (EVENT_NAME_EJECTION == sp_events[i].event_name)
		{
			increment_event_counter(n_ejection);
		}
		else
		{
			handle_collision_pair(i, &sp_events[i]);
			increment_event_counter(n_hit_centrum);
		}
		// Make the body which has hitted the center or ejected inactive 
		sim_data->h_body_md[ictv_idx].id *= -1;
		// Copy it up to GPU 
		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			copy_vector_to_device((void **)&sim_data->d_body_md[ictv_idx], (void *)&sim_data->h_body_md[ictv_idx], sizeof(body_metadata_t));
		}
		// Update number of inactive bodies
		n_bodies->inactive[sim_data->h_body_md[ictv_idx].body_type]++;
	}
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
	switch (aps)
	{
	case ACTUAL_PHASE_STORAGE_Y:
		sim_data->h_y[0][survivIdx] = collision->rs;
		sim_data->h_y[1][survivIdx] = collision->vs;
		break;
	case ACTUAL_PHASE_STORAGE_YOUT:
		sim_data->h_yout[0][survivIdx] = collision->rs;
		sim_data->h_yout[1][survivIdx] = collision->vs;
		break;
	default:
		throw string("Parameter 'aps' is out of range.");
	}

	// Calculate physical properties of the new object
	tools::calc_physical_properties(collision->p1, collision->p2, collision->ps);
	// Update physical properties of survivor
	sim_data->h_p[survivIdx] = collision->ps;

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		switch (aps)
		{
		case ACTUAL_PHASE_STORAGE_Y:
			copy_vector_to_device((void **)&sim_data->d_y[0][survivIdx],	(void *)&sim_data->h_y[0][survivIdx],	 sizeof(vec_t));
			copy_vector_to_device((void **)&sim_data->d_y[1][survivIdx],	(void *)&sim_data->h_y[1][survivIdx],	 sizeof(vec_t));
			break;
		case ACTUAL_PHASE_STORAGE_YOUT:
			copy_vector_to_device((void **)&sim_data->d_yout[0][survivIdx],	(void *)&sim_data->h_yout[0][survivIdx], sizeof(vec_t));
			copy_vector_to_device((void **)&sim_data->d_yout[1][survivIdx],	(void *)&sim_data->h_yout[1][survivIdx], sizeof(vec_t));
			break;
		default:
			throw string("Parameter 'aps' is out of range.");
		}
		copy_vector_to_device((void **)&sim_data->d_p[survivIdx], (void *)&sim_data->h_p[survivIdx], sizeof(param_t));
	}
}

pp_disk::pp_disk(number_of_bodies *n_bodies, int n_tpb, bool ups, gas_disk_model_t g_disk_model, computing_device_t comp_dev) :
	n_bodies(n_bodies),
	n_tpb(n_tpb),
	ups(ups),
	g_disk_model(g_disk_model),
	comp_dev(comp_dev)
{
	initialize();
	allocate_storage();
	redutilcu::create_aliases(comp_dev, sim_data);
	tools::populate_data(n_bodies->initial, sim_data);
}

pp_disk::pp_disk(string& path, bool continue_simulation, int n_tpb, bool ups, gas_disk_model_t g_disk_model, computing_device_t comp_dev) :
	n_tpb(n_tpb),
	continue_simulation(continue_simulation),
	ups(ups),
	g_disk_model(g_disk_model),
	comp_dev(comp_dev)
{
	initialize();

	data_representation_t repres = (file::get_extension(path) == "txt" ? DATA_REPRESENTATION_ASCII : DATA_REPRESENTATION_BINARY);

	n_bodies = load_number_of_bodies(path, repres);
	allocate_storage();
	redutilcu::create_aliases(comp_dev, sim_data);
	load(path, repres);
}

pp_disk::~pp_disk()
{
	deallocate_host_storage(sim_data);
	FREE_HOST_VECTOR((void **)&events);

	deallocate_device_storage(sim_data);
	FREE_DEVICE_VECTOR((void **)&d_events);
	FREE_DEVICE_VECTOR((void **)&d_event_counter);

	delete sim_data;
}

void pp_disk::initialize()
{
	aps             = ACTUAL_PHASE_STORAGE_Y;

	t               = 0.0;
	sim_data        = 0x0;
	event_counter   = 0;
	d_event_counter = 0x0;
	events          = 0x0;
	d_events        = 0x0;

	a_gd            = 0x0;
	f_gd            = 0x0;

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
	int n_total = get_n_total_body();

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

	int n_total = get_n_total_body();

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
		throw string("Parameter 'device' is out of range.");
	}
	redutilcu::create_aliases(device, sim_data);

	this->comp_dev = device;
}

void pp_disk::remove_inactive_bodies()
{
	int n_total_players = get_n_total_body();

	sim_data_t* sim_data_temp = new sim_data_t;
	// Only the data of the active bodies will be temporarily stored
	allocate_host_storage(sim_data_temp, n_bodies->get_n_total_active());

	// Create aliases:
	vec_t* r = (aps == ACTUAL_PHASE_STORAGE_Y ? sim_data->h_y[0] : sim_data->h_yout[0]);
	vec_t* v = (aps == ACTUAL_PHASE_STORAGE_Y ? sim_data->h_y[1] : sim_data->h_yout[1]);
	param_t *p = sim_data->h_p;
	body_metadata_t *bmd = sim_data->h_body_md;

	// Copy the data of active bodies to sim_data_temp
	int i = 0;
	int k = 0;
	for ( ; i < n_total_players; i++)
	{
		if (0 < sim_data->h_body_md[i].id && BODY_TYPE_PADDINGPARTICLE != sim_data->h_body_md[i].body_type)
		{
			sim_data_temp->h_y[0][k]    = r[i];
			sim_data_temp->h_y[1][k]    = v[i];
			sim_data_temp->h_p[k]       = p[i];     //sim_data->h_p[i];
			sim_data_temp->h_body_md[k] = bmd[i];   //sim_data->h_body_md[i];
			k++;
		}
	}
	if (n_bodies->get_n_total_active() != k)
	{
		throw string("Number of copied bodies does not equal to the number of active bodies.");
	}
	this->n_bodies->playing[BODY_TYPE_PADDINGPARTICLE] = 0;
	this->n_bodies->update();

	int n_SI		= n_bodies->get_n_SI();
	int n_NSI		= n_bodies->get_n_NSI();
	int n_total		= n_bodies->get_n_total_active();
	int n_prime_SI	= n_bodies->get_n_prime_SI(n_tpb);
	int n_prime_NSI	= n_bodies->get_n_prime_NSI(n_tpb);
	int n_prime_total=n_bodies->get_n_prime_total(n_tpb);

	k = 0;
	i = 0;
	for ( ; i < n_SI; i++, k++)
	{
		r[k]   = sim_data_temp->h_y[0][i];
		v[k]   = sim_data_temp->h_y[1][i];
		p[k]   = sim_data_temp->h_p[i];
		bmd[k] = sim_data_temp->h_body_md[i];
	}
    while (ups && k < n_prime_SI)
    {
		create_padding_particle(k, sim_data->h_epoch, bmd, p, r, v);
		this->n_bodies->playing[BODY_TYPE_PADDINGPARTICLE]++;
        k++;
    }

	for ( ; i < n_SI + n_NSI; i++, k++)
	{
		r[k]   = sim_data_temp->h_y[0][i];
		v[k]   = sim_data_temp->h_y[1][i];
		p[k]   = sim_data_temp->h_p[i];
		bmd[k] = sim_data_temp->h_body_md[i];
	}
    while (ups && k < n_prime_SI + n_prime_NSI)
    {
		create_padding_particle(k, sim_data->h_epoch, bmd, p, r, v);
		this->n_bodies->playing[BODY_TYPE_PADDINGPARTICLE]++;
        k++;
    }

	for ( ; i < n_total; i++, k++)
	{
		r[k]   = sim_data_temp->h_y[0][i];
		v[k]   = sim_data_temp->h_y[1][i];
		p[k]   = sim_data_temp->h_p[i];
		bmd[k] = sim_data_temp->h_body_md[i];
	}
    while (ups && k < n_prime_total)
    {
		create_padding_particle(k, sim_data->h_epoch, bmd, p, r, v);
		this->n_bodies->playing[BODY_TYPE_PADDINGPARTICLE]++;
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
	int n_body = get_n_total_body();

	for (int i = 0; i < 2; i++)
	{
		switch (aps)
		{
		case ACTUAL_PHASE_STORAGE_Y:
			copy_vector_to_device((void *)sim_data->d_y[i],	    (void *)sim_data->h_y[i],    n_body*sizeof(vec_t));
			break;
		case ACTUAL_PHASE_STORAGE_YOUT:
			copy_vector_to_device((void *)sim_data->d_yout[i],	(void *)sim_data->h_yout[i], n_body*sizeof(vec_t));
			break;
		default:
			throw string("Parameter 'aps' is out of range.");
		}
	}
	copy_vector_to_device((void *)sim_data->d_p,		(void *)sim_data->h_p,		 n_body*sizeof(param_t));
	copy_vector_to_device((void *)sim_data->d_body_md,	(void *)sim_data->h_body_md, n_body*sizeof(body_metadata_t));
	copy_vector_to_device((void *)sim_data->d_epoch,	(void *)sim_data->h_epoch,	 n_body*sizeof(ttt_t));

	copy_vector_to_device((void *)d_event_counter,		(void *)&event_counter,		      1*sizeof(int));
}

void pp_disk::copy_to_host()
{
	int n_body = get_n_total_body();

	for (int i = 0; i < 2; i++)
	{
		switch (aps)
		{
		case ACTUAL_PHASE_STORAGE_Y:
			copy_vector_to_host((void *)sim_data->h_y[i],    (void *)sim_data->d_y[i],    n_body*sizeof(vec_t));
			break;
		case ACTUAL_PHASE_STORAGE_YOUT:
			copy_vector_to_host((void *)sim_data->h_yout[i], (void *)sim_data->d_yout[i], n_body*sizeof(vec_t));
			break;
		default:
			throw string("Parameter 'aps' is out of range.");
		}
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
	default:
		throw string("Parameter 'g_disk_model' is out of range.");
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

int pp_disk::get_n_total_body()
{
	return (ups ? n_bodies->get_n_prime_total(n_tpb) : n_bodies->get_n_total_playing());
}

var_t pp_disk::get_mass_of_star()
{
	int n_massive = ups ? n_bodies->get_n_prime_massive(n_tpb) : n_bodies->get_n_massive();

	body_metadata_t* body_md = sim_data->h_body_md;
	for (int j = 0; j < n_massive; j++ )
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
	int n = get_n_total_body();

	tools::transform_to_bc(n, verbose, sim_data);
}

void pp_disk::transform_time(bool verbose)
{
	if (verbose)
	{
		cout << "Transforming the time ... ";
	}

	// Transform the bodies' epochs
	int n = get_n_total_body();
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
	int n = get_n_total_body();
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

number_of_bodies* pp_disk::load_number_of_bodies(string& path, data_representation_t repres)
{
	unsigned int ns, ngp, nrp, npp, nspl, npl, ntp;
	ns = ngp = nrp = npp = nspl = npl = ntp = 0;

	ifstream input;
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		input.open(path.c_str());
		if (input) 
		{
			input >> ns >> ngp >> nrp >> npp >> nspl >> npl >> ntp;
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	case DATA_REPRESENTATION_BINARY:
		input.open(path.c_str(), ios::in | ios::binary);
		if (input) 
		{
			input.read((char*)&ns,   sizeof(ns));
			input.read((char*)&ngp,  sizeof(ngp));
			input.read((char*)&nrp,  sizeof(nrp));
			input.read((char*)&npp,  sizeof(npp));
			input.read((char*)&nspl, sizeof(nspl));
			input.read((char*)&npl,  sizeof(npl));
			input.read((char*)&ntp,  sizeof(ntp));
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	}
	input.close();

    return new number_of_bodies(ns, ngp, nrp, npp, nspl, npl, ntp);
}

void pp_disk::load_body_record(ifstream& input, int k, ttt_t* epoch, body_metadata_t* body_md, param_t* p, vec_t* r, vec_t* v)
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

void pp_disk::load(string& path, data_representation_t repres)
{
	cout << "Loading " << path << " ... ";

	ifstream input;
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		input.open(path.c_str());
		if (input) 
		{
			load_ascii(input);
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	case DATA_REPRESENTATION_BINARY:
		input.open(path.c_str(), ios::in | ios::binary);
		if (input) 
		{
			load_binary(input);
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	}
	input.close();

	cout << "done" << endl;
}

void pp_disk::load_ascii(ifstream& input)
{
	int ns, ngp, nrp, npp, nspl, npl, ntp;
	input >> ns >> ngp >> nrp >> npp >> nspl >> npl >> ntp;

	vec_t* r = sim_data->h_y[0];
	vec_t* v = sim_data->h_y[1];
	param_t* p = sim_data->h_p;
	body_metadata_t* body_md = sim_data->h_body_md;
	ttt_t* epoch = sim_data->h_epoch;

	int n_SI		= n_bodies->get_n_SI();
	int n_NSI		= n_bodies->get_n_NSI();
	int n_total		= n_bodies->get_n_total_playing();
	int n_prime_SI	= n_bodies->get_n_prime_SI(n_tpb);
	int n_prime_NSI	= n_bodies->get_n_prime_NSI(n_tpb);
	int n_prime_total=n_bodies->get_n_prime_total(n_tpb); 

	int i = 0;
	int k = 0;
	for ( ; i < n_SI; i++, k++)
	{
		load_body_record(input, k, epoch, body_md, p, r, v);
	}
    while (ups && k < n_prime_SI)
    {
		create_padding_particle(k, epoch, body_md, p, r, v);
		this->n_bodies->playing[BODY_TYPE_PADDINGPARTICLE]++;
        k++;
    }

	for ( ; i < n_SI + n_NSI; i++, k++)
	{
		load_body_record(input, k, epoch, body_md, p, r, v);
	}
	while (ups && k < n_prime_SI + n_prime_NSI)
	{
		create_padding_particle(k, epoch, body_md, p, r, v);
		this->n_bodies->playing[BODY_TYPE_PADDINGPARTICLE]++;
		k++;
	}

	for ( ; i < n_total; i++, k++)
	{
		load_body_record(input, k, epoch, body_md, p, r, v);
	}
	while (ups && k < n_prime_total)
	{
		create_padding_particle(k, epoch, body_md, p, r, v);
		this->n_bodies->playing[BODY_TYPE_PADDINGPARTICLE]++;
		k++;
	}
}

void pp_disk::load_binary(ifstream& input)
{
	for (unsigned int type = 0; type < BODY_TYPE_N; type++)
	{
		if (BODY_TYPE_PADDINGPARTICLE == type)
		{
			continue;
		}
		unsigned int tmp = 0;
		input.read((char*)&tmp, sizeof(tmp));
	}

	char name_buffer[30];
	vec_t* r = sim_data->h_y[0];
	vec_t* v = sim_data->h_y[1];
	param_t* p = sim_data->h_p;
	body_metadata_t* bmd = sim_data->h_body_md;
	ttt_t* epoch = sim_data->h_epoch;

	unsigned int n_total = n_bodies->get_n_total_initial();
	for (unsigned int i = 0; i < n_total; i++)
	{
		memset(name_buffer, 0, sizeof(name_buffer));

		input.read((char*)&epoch[i],  1*sizeof(ttt_t));
		input.read(name_buffer,      30*sizeof(char));
		input.read((char*)&bmd[i],    1*sizeof(body_metadata_t));
		input.read((char*)&p[i],      1*sizeof(param_t));
		input.read((char*)&r[i],      1*sizeof(vec_t));
		input.read((char*)&v[i],      1*sizeof(vec_t));

		body_names.push_back(name_buffer);
	}
}

void pp_disk::print_dump(ostream& sout, data_representation_t repres)
{
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		for (unsigned int type = 0; type < BODY_TYPE_N; type++)
		{
			if (BODY_TYPE_PADDINGPARTICLE == type)
			{
				continue;
			}
			sout << n_bodies->get_n_active_by((body_type_t)type) << SEP;
		}
		sout << endl;
		print_result_ascii(sout);
		break;
	case DATA_REPRESENTATION_BINARY:
		for (unsigned int type = 0; type < BODY_TYPE_N; type++)
		{
			if (BODY_TYPE_PADDINGPARTICLE == type)
			{
				continue;
			}
			unsigned int _n = n_bodies->get_n_active_by((body_type_t)type);
			sout.write((char*)&_n, sizeof(_n));
		}
		print_result_binary(sout);
		break;
	}
}

void pp_disk::print_result_ascii(ostream& sout)
{
	static int int_t_w  =  8;
	static int var_t_w  = 25;

	sout.precision(16);
	sout.setf(ios::right);
	sout.setf(ios::scientific);

	vec_t* r = (aps == ACTUAL_PHASE_STORAGE_Y ? sim_data->h_y[0] : sim_data->h_yout[0]);
	vec_t* v = (aps == ACTUAL_PHASE_STORAGE_Y ? sim_data->h_y[1] : sim_data->h_yout[1]);
	param_t* p = sim_data->h_p;
	body_metadata_t* bmd = sim_data->h_body_md;

	int n = get_n_total_body();
	for (int i = 0; i < n; i++)
    {
		// Skip inactive bodies and padding particles and alike
		if (bmd[i].id <= 0 || bmd[i].body_type >= BODY_TYPE_PADDINGPARTICLE)
		{
			continue;
		}
		int orig_idx = bmd[i].id - 1;
		sout << setw(int_t_w) << bmd[i].id << SEP                    /* id of the body starting from 1                                (int)              */
			 << setw(     30) << body_names[orig_idx] << SEP         /* name of the body                                              (string = 30 char) */ 
			 << setw(      2) << bmd[i].body_type << SEP             /* type of the body                                              (int)              */
			 << setw(var_t_w) << t / constants::Gauss << SEP         /* time of the record                           [day]            (double)           */
			 << setw(var_t_w) << p[i].mass << SEP                    /* mass of the body                             [solar mass]     (double)           */
			 << setw(var_t_w) << p[i].radius << SEP                  /* radius of the body                           [AU]             (double)           */
			 << setw(var_t_w) << p[i].density << SEP                 /* density of the body in                       [solar mass/AU3] (double)           */
			 << setw(var_t_w) << p[i].cd << SEP                      /* Stokes drag coefficeint dimensionless                         (double)           */
			 << setw(      2) << bmd[i].mig_type << SEP              /* migration type of the body                                    (int)              */
			 << setw(var_t_w) << bmd[i].mig_stop_at << SEP           /* migration stops at this barycentric distance [AU]             (double)           */
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
	char name_buffer[30];

	vec_t* r = (aps == ACTUAL_PHASE_STORAGE_Y ? sim_data->h_y[0] : sim_data->h_yout[0]);
	vec_t* v = (aps == ACTUAL_PHASE_STORAGE_Y ? sim_data->h_y[1] : sim_data->h_yout[1]);
	param_t* p = sim_data->h_p;
	body_metadata_t* bmd = sim_data->h_body_md;

	int n = get_n_total_body();
	for (int i = 0; i < n; i++)
    {
		// Skip inactive bodies and padding particles and alike
		if (bmd[i].id <= 0 || bmd[i].body_type >= BODY_TYPE_PADDINGPARTICLE)
		{
			continue;
		}
		int orig_idx = bmd[i].id - 1;
		memset(name_buffer, 0, sizeof(name_buffer));
		strcpy(name_buffer, body_names[orig_idx].c_str());

		sout.write((char*)&(this->t), sizeof(ttt_t));
		sout.write(name_buffer,    sizeof(name_buffer));
		sout.write((char*)&bmd[i], sizeof(body_metadata_t));
		sout.write((char*)&p[i],   sizeof(param_t));
		sout.write((char*)&r[i],   sizeof(vec_t));
		sout.write((char*)&v[i],   sizeof(vec_t));
	}
}

void pp_disk::print_event_data(ostream& sout, ostream& log_f)
{
	static int int_t_w =  8;
	static int var_t_w = 25;
	string e_names[] = {"NONE", "HIT_CENTRUM", "EJECTION", "CLOSE_ENCOUNTER", "COLLISION"};

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
			 << setw(var_t_w) << sp_events[i].v2.y * constants::Gauss << SEP
			 << setw(var_t_w) << sp_events[i].v2.z * constants::Gauss << SEP
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
