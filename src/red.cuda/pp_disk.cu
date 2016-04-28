#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <float.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "nbody_exception.h"
#include "pp_disk.h"
#include "redutilcu.h"
#include "red_constants.h"
#include "red_macro.h"
#include "red_type.h"

using namespace std;
using namespace redutilcu;
using namespace pp_disk_t;

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
	var_t norm(const var4_t* r)
{
	return sqrt(SQR(r->x) + SQR(r->y) + SQR(r->z));
}

__host__ __device__
	var_t calc_dot_product(const var4_t& u, const var4_t& v)
{
    return (u.x * v.x + u.y * v.y + u.z * v.z);
}

__host__ __device__
	var4_t calc_cross_product(const var4_t& u, const var4_t& v)
{
    var4_t result = {0.0, 0.0, 0.0, 0.0};
    
    result.x = u.y * v.z - u.z * v.y;
    result.y = u.z * v.x - u.x * v.z;
    result.z = u.x * v.y - u.y * v.x;

    return result;
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
	var_t get_density(var2_t sch, var2_t rho, const var4_t* rVec)
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
	var4_t circular_velocity(var_t mu, const var4_t* rVec)
{
	var4_t result = {0.0, 0.0, 0.0, 0.0};

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
	var4_t get_velocity(var_t mu, var2_t eta, const var4_t* rVec)
{
	var4_t v_gas = circular_velocity(mu, rVec);
	var_t r = sqrt(SQR(rVec->x) + SQR(rVec->y));

	var_t v = sqrt(1.0 - 2.0*eta.x * pow(r, eta.y));
	v_gas.x *= v;
	v_gas.y *= v;
	
	return v_gas;
}

__host__ __device__
	uint32_t calc_linear_index(const var4_t& rVec, var_t* used_rad, uint32_t n_sec, uint32_t n_rad)
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
		uint32_t i_rad = 0;
		uint32_t i_sec = 0;
		for (uint32_t k = 0; k < n_rad; k++)
		{
			if (used_rad[k] <= r && r < used_rad[k+1])
			{
				i_rad = k;
				break;
			}
		}

		var_t alpha = (rVec.y >= 0.0 ? atan2(rVec.y, rVec.x) : TWOPI + atan2(rVec.y, rVec.x));
		i_sec =  (uint32_t)(alpha / dalpha);
		uint32_t i_linear = i_rad * n_sec + i_sec;

		return i_linear;
	}
}

__host__ __device__ 
	void store_event_data
	(
		event_name_t name,
		ttt_t t,
		var_t d,
		uint32_t idx1,     /* survivor/target   */
		uint32_t idx2,     /* merger/projectile */
		const param_t* p,
		const var4_t* r,
		const var4_t* v,
		const body_metadata_t* bmd,
		event_data_t *evnt)
{
	evnt->event_name = name;
	evnt->d = d;
	evnt->t = t;
	evnt->id1 = bmd[idx1].id;
	evnt->id2 = bmd[idx2].id;
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
		printf("COLLISION,   %20.10le, [d], %20.10le, [AU], %5d, %5d\n", t/K, d, bmd[idx1].id, bmd[idx2].id);
		break;
	case EVENT_NAME_EJECTION:
		printf("EJECTION,    %20.10le, [d], %20.10le, [AU], %5d, %5d\n", t/K, d, bmd[idx1].id, bmd[idx2].id);
		break;
	case EVENT_NAME_HIT_CENTRUM:
		printf("HIT_CENTRUM, %20.10le, [d], %20.10le, [AU], %5d, %5d\n", t/K, d, bmd[idx1].id, bmd[idx2].id);
		break;
	}
}

inline __host__ __device__
	var4_t body_body_interaction(var4_t riVec, var4_t rjVec, var_t mj, var4_t aiVec)
{
	var4_t dVec = {0.0, 0.0, 0.0, 0.0};

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

} /* pp_disk_utility */

namespace kernel_pp_disk
{
static __global__
	void check_for_ejection_hit_centrum
	(
		ttt_t curr_t, 
		interaction_bound int_bound, 
		const param_t* p, 
		const var4_t* r, 
		const var4_t* v, 
		const body_metadata_t* bmd, 
		event_data_t* events,
		uint32_t *event_counter
	)
{
	const uint32_t i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	var_t ed2  = (0.0 < dc_threshold[THRESHOLD_EJECTION_DISTANCE] ? SQR(dc_threshold[THRESHOLD_EJECTION_DISTANCE]) : DBL_MAX);
	var_t hcd2 = SQR(dc_threshold[THRESHOLD_HIT_CENTRUM_DISTANCE]);
	// Ignore the inactive bodies (whose id < 0) and the star
	if (i < int_bound.sink.y && bmd[i].id > 0 && bmd[i].body_type != BODY_TYPE_STAR)
	{
		uint32_t k = 0;

		// Calculate the distance from the barycenter
		var_t r2 = SQR(r[i].x) + SQR(r[i].y) + SQR(r[i].z);
		if (ed2 < r2)
		{
			k = atomicAdd(event_counter, 1);
			pp_disk_utility::store_event_data(EVENT_NAME_EJECTION,    curr_t, sqrt(r2), 0, i, p, r, v, bmd, &events[k]);
		}
		if (hcd2 > r2)
		{
			k = atomicAdd(event_counter, 1);
			pp_disk_utility::store_event_data(EVENT_NAME_HIT_CENTRUM, curr_t, sqrt(r2), 0, i, p, r, v, bmd, &events[k]);
		}
	}
}

static __global__
	void check_for_collision
	(
		ttt_t curr_t, 
		interaction_bound int_bound, 
		const param_t* p, 
		const var4_t* r, 
		const var4_t* v, 
		const body_metadata_t* bmd, 
		event_data_t* events,
		uint32_t *event_counter
	)
{
	const uint32_t i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;
	const double f = SQR(dc_threshold[THRESHOLD_RADII_ENHANCE_FACTOR]);

	if (i < int_bound.sink.y && 0 < bmd[i].id)
	{
		var4_t dVec = {0.0, 0.0, 0.0, 0.0};
		for (uint32_t j = i + 1; j < int_bound.source.y; j++) 
		{
			/* Skip inactive bodies, i.e. id < 0 */
			if (0 > bmd[j].id)
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
				uint32_t k = atomicAdd(event_counter, 1);

				uint32_t survivIdx = i;
				uint32_t mergerIdx = j;
				if (p[mergerIdx].mass > p[survivIdx].mass)
				{
					uint32_t idx = survivIdx;
					survivIdx = mergerIdx;
					mergerIdx = idx;
				}
				pp_disk_utility::store_event_data(EVENT_NAME_COLLISION, curr_t, sqrt(dVec.w), survivIdx, mergerIdx, p, r, v, bmd, &events[k]);
			}
		}
	}
}

static __global__
	void calc_grav_accel_naive
	(
		interaction_bound int_bound, 
		const body_metadata_t* bmd, 
		const param_t* p, 
		const var4_t* r, 
		var4_t* a
	)
{
	const uint32_t i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	if (i < int_bound.sink.y)
	{
		a[i].x = a[i].y = a[i].z = a[i].w = 0.0;
		if (0 < bmd[i].id)
		{
			var4_t dVec = {0.0, 0.0, 0.0, 0.0};
			for (uint32_t j = int_bound.source.x; j < int_bound.source.y; j++) 
			{
				/* Skip the body with the same index and those which are inactive ie. id < 0 */
				if (i == j || 0 > bmd[j].id)
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
			} // 36 FLOP
		}
	}
}

static __global__
	void calc_grav_accel_naive
	(
		ttt_t curr_t, 
		interaction_bound int_bound, 
		const body_metadata_t* bmd, 
		const param_t* p, 
		const var4_t* r, 
		const var4_t* v, 
		var4_t* a,
		event_data_t* events,
		uint32_t *event_counter
	)
{
	const uint32_t i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	if (i < int_bound.sink.y)
	{
		a[i].x = a[i].y = a[i].z = a[i].w = 0.0;
		if (0 < bmd[i].id)
		{
			var4_t dVec = {0.0, 0.0, 0.0, 0.0};
			for (uint32_t j = int_bound.source.x; j < int_bound.source.y; j++) 
			{
				/* Skip the body with the same index and those which are inactive ie. id < 0 */
				if (i == j || 0 > bmd[j].id)
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
					uint32_t k = atomicAdd(event_counter, 1);

					uint32_t survivIdx = i;
					uint32_t mergerIdx = j;
					if (p[mergerIdx].mass > p[survivIdx].mass)
					{
						uint32_t t = survivIdx;
						survivIdx = mergerIdx;
						mergerIdx = t;
					}
					pp_disk_utility::store_event_data(EVENT_NAME_COLLISION, curr_t, d, survivIdx, mergerIdx, p, r, v, bmd, &events[k]);
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
		const body_metadata_t* bmd, 
		const param_t* p, 
		const var4_t* r, 
		const var4_t* v, 
		var4_t* a
	)
{
	const uint32_t i = int_bound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

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
		var4_t v_g       = pp_disk_utility::get_velocity(m_star, dc_anal_gd_params.eta, &r[i]);
		var4_t u         = {v_g.x - v[i].x, v_g.y - v[i].y, v_g.z - v[i].z, 0.0};
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
	void print_body_metadata(uint32_t n, const body_metadata_t* bmd)
{
	uint32_t tid = 0;
	while (tid < n)
	{
		printf("%5d %6d %2d %2d %24.16le\n", tid, bmd[tid].id, bmd[tid].body_type, bmd[tid].mig_type, bmd[tid].mig_stop_at);
		tid++;
	}
}


static __global__
	void print_vector(uint32_t n, const var4_t* v)
{
	uint32_t tid = 0;
	while (tid < n)
	{
		printf("%5d %24.16le %24.16le %24.16le %24.16le\n", tid, v[tid].x, v[tid].y, v[tid].z, v[tid].w);
		tid++;
	}
}

} /* kernel_utility */


uint32_t pp_disk::benchmark()
{
	// Create aliases
	var4_t* r   = sim_data->y[0];
	var4_t* v   = sim_data->y[1];
	param_t* p = sim_data->p;
	body_metadata_t* bmd = sim_data->body_md;

	size_t size = n_bodies->get_n_total_playing() * sizeof(var4_t);
	var4_t* d_dy = 0x0;
	ALLOCATE_DEVICE_VECTOR((void**)&d_dy, size);

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, this->id_dev));

	uint32_t half_warp_size = deviceProp.warpSize/2;
	vector<float2> execution_time;

	uint32_t n_sink = n_bodies->get_n_SI();
	uint32_t n_pass = 0;
	uint32_t min_idx = 0;
	if (0 < n_sink)
	{
		for (uint32_t n_tpb = half_warp_size; n_tpb <= (uint32_t)deviceProp.maxThreadsPerBlock/2; n_tpb += half_warp_size)
		{
			set_n_tpb(n_tpb);
			interaction_bound int_bound = n_bodies->get_bound_SI();

			clock_t t_start = clock();
			float cu_elt = benchmark_calc_grav_accel(t, n_sink, int_bound, bmd, p, r, v, d_dy);
			clock_t elapsed_time = clock() - t_start;

			if (cudaSuccess != cudaGetLastError())
			{
				break;
			}
			float2 exec_t = {(float)elapsed_time, cu_elt};
			execution_time.push_back(exec_t);
			n_pass++;
		} /* for */

		float min_y = 1.0e10;
		for (uint32_t i = 0; i < n_pass; i++)
		{
			if (min_y > execution_time[i].y)
			{
				min_y = execution_time[i].y;
				min_idx = i;
			}
		}
	} /* if */
	FREE_DEVICE_VECTOR((void**)&d_dy);

	return ((min_idx + 1) * half_warp_size);
}

float pp_disk::benchmark_calc_grav_accel(ttt_t curr_t, uint32_t n_sink, interaction_bound int_bound, const body_metadata_t* body_md, const param_t* p, const var4_t* r, const var4_t* v, var4_t* a)
{
	cudaEvent_t start, stop;

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	set_kernel_launch_param(n_sink);

	CUDA_SAFE_CALL(cudaEventRecord(start, 0));
	kernel_pp_disk::calc_grav_accel_naive<<<grid, block>>>(int_bound, sim_data->body_md, sim_data->p, r, a);
	CUDA_CHECK_ERROR();
	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float elapsed_time = 0.0f;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));

	return elapsed_time;
}

void pp_disk::print_sim_data(computing_device_t cd)
{
	uint32_t n_total = n_bodies->get_n_total_playing();

	switch (cd)
	{
	case COMPUTING_DEVICE_CPU:
		{
		printf("**********************************************************************\n");
		printf("                 DATA ON THE HOST                                     \n");
		printf("**********************************************************************\n");
		printf("\nPosition:\n");
		const var4_t* v = sim_data->h_y[0];
		for (uint32_t i = 0; i < n_total; i++)
		{
			printf("%5d %24.16le %24.16le %24.16le %24.16le\n", i, v[i].x, v[i].y, v[i].z, v[i].w);
		}

		printf("\nBody metaddata:\n");
		const body_metadata_t* bmd = sim_data->h_body_md;
		for (uint32_t i = 0; i < n_total; i++)
		{
			printf("%5d %6d %2d %2d %20.16le\n", i, bmd[i].id, bmd[i].body_type, bmd[i].mig_type, bmd[i].mig_stop_at);
		}

		}
		break;
	case COMPUTING_DEVICE_GPU:
		{
		printf("**********************************************************************\n");
		printf("                 DATA ON THE DEVICE                                   \n");
		printf("**********************************************************************\n");
		printf("\nPosition:\n");
		kernel_utility::print_vector<<<1, 1>>>(n_total, sim_data->d_y[0]);
		CUDA_CHECK_ERROR();

		//printf("\nVelocity:\n");
		//kernel_utility::print_vector<<<1, 1>>>(n_total, sim_data->d_y[1]);
		//CUDA_CHECK_ERROR();

		//kernel_utility::print_vector<<<1, 1>>>(n_total, (var4_t*)sim_data->d_p);
		//CUDA_CHECK_ERROR();

		kernel_utility::print_body_metadata<<<1, 1>>>(n_total, sim_data->d_body_md);
		CUDA_CHECK_ERROR();

		//kernel_utility::print_epochs<<<1, 1>>>(n_total, sim_data->d_epoch);
		//CUDA_CHECK_ERROR();

		//kernel_utility::print_constant_memory<<<1, 1>>>();
		//CUDA_CHECK_ERROR();
		break;
		}
	default:
		throw string("Parameter cd is out of range.");
	}
}

void pp_disk::set_kernel_launch_param(uint32_t n_data)
{
	uint32_t n_thread = min(n_tpb, n_data);
	uint32_t n_block = (n_data + n_thread - 1)/n_thread;

	grid.x	= n_block;
	block.x = n_thread;
}

void pp_disk::cpu_calc_drag_accel(ttt_t curr_t, const var4_t* r, const var4_t* v, var4_t* dy)
{
	uint32_t n_sink = n_bodies->get_n_NSI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_GD();
		cpu_calc_drag_accel_NSI(curr_t, int_bound, sim_data->body_md, sim_data->p, r, v, dy);
	}
}

void pp_disk::cpu_calc_drag_accel_NSI(ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* bmd, const param_t* p, const var4_t* r, const var4_t* v, var4_t* a)
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
		for (uint32_t i = int_bound.sink.x; i < int_bound.sink.y; i++)
		{
			var4_t v_g       = pp_disk_utility::get_velocity(m_star, a_gd->params.eta, &r[i]);
			var4_t u         = {v_g.x - v[i].x, v_g.y - v[i].y, v_g.z - v[i].z, 0.0};
			var_t u_n       = sqrt(SQR(u.x) + SQR(u.y) + SQR(u.z));
			var_t density_g = pp_disk_utility::get_density(a_gd->params.sch, a_gd->params.rho, &r[i]);

			var_t f = decr_fact * (3.0 * p[i].cd * density_g * u_n) / (8.0 * p[i].radius * p[i].density);

			a[i].x += f * u.x;
			a[i].y += f * u.y;
			a[i].z += f * u.z;

			//var_t rhoGas = rFactor * gas_density_at(gasDisk, (var4_t*)&coor[bodyIdx]);
			//var_t r = norm((var4_t*)&coor[bodyIdx]);

			//var4_t u;
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
		for (uint32_t i = int_bound.sink.x; i < int_bound.sink.y; i++)
		{
			uint32_t linear_index= pp_disk_utility::calc_linear_index(r[i], f_gd->used_rad[0], f_gd->params.n_sec,  f_gd->params.n_rad);
			var_t r_norm    = sqrt(SQR(r[i].x) + SQR(r[i].y));

			// Massless body's circular velocity at r_norm distance from the barycenter
			var_t vc_theta  = sqrt(m_star/r_norm);

			// Get gas parcel's velocity at r[i]
			var_t v_g_theta = vc_theta + f_gd->vtheta[0][linear_index];
			var_t v_g_rad   = f_gd->vrad[0][linear_index];
			var4_t v_g_polar = {v_g_rad, v_g_theta, 0.0, 0.0};
			
			// Calculate the angle between the r position vector and the x-axis
			var_t theta     = (r[i].y >= 0.0 ? atan2(r[i].y, r[i].x) : TWOPI + atan2(r[i].y, r[i].x));

			// TODO: calculate the x and y components of the gas velocity
			var4_t v_g       = redutilcu::rotate_2D_vector(theta, v_g_polar);
			
			// Get gas parcel's density at r[i]
			var_t density_g = f_gd->density[0][linear_index];

			// Compute the solid body's relative velocity
			var4_t u         = {v_g.x - v[i].x, v_g.y - v[i].y, v_g.z - v[i].z, 0.0};
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



void pp_disk::cpu_calc_grav_accel_SI( ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* bmd, const param_t* p, const var4_t* r, const var4_t* v, var4_t* a)
{
	for (uint32_t i = int_bound.sink.x; i < int_bound.sink.y; i++)
	{
		if (0 < bmd[i].id)
		{
			var4_t dVec = {0.0, 0.0, 0.0, 0.0};
			//for (uint32_t j = int_bound.source.x; j < int_bound.source.y; j++) 
			for (uint32_t j = i+1; j < int_bound.source.y; j++) 
			{
				/* Skip the body with the same index and those which are inactive ie. id < 0 */
				//if (i == j || 0 > bmd[j].id)
				if (0 > bmd[j].id)
				{
					continue;
				}
				// r_ij 3 FLOP
				dVec.x = r[j].x - r[i].x;
				dVec.y = r[j].y - r[i].y;
				dVec.z = r[j].z - r[i].z;
				// 5 FLOP
				dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2

				// 20 FLOP
				var_t d = sqrt(dVec.w);								// = r
				// 2 FLOP
				var_t r_3 = 1.0 / (d*dVec.w);
				// 1 FLOP
				dVec.w = p[j].mass * r_3;
				// 6 FLOP
				a[i].x += dVec.w * dVec.x;
				a[i].y += dVec.w * dVec.y;
				a[i].z += dVec.w * dVec.z;

				// 2 FLOP
				dVec.w = p[i].mass * r_3;
				// 6 FLOP
				a[j].x -= dVec.w * dVec.x;
				a[j].y -= dVec.w * dVec.y;
				a[j].z -= dVec.w * dVec.z;
			} // 36 + 8 = 44 FLOP
		}
	}
}

void pp_disk::cpu_calc_grav_accel_SI( ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* bmd, const param_t* p, const var4_t* r, const var4_t* v, var4_t* a, event_data_t* events, uint32_t *event_counter)
{
	memset(a, 0, (int_bound.sink.y - int_bound.sink.x)*sizeof(var4_t));

	for (uint32_t i = int_bound.sink.x; i < int_bound.sink.y; i++)
	{
		if (0 < bmd[i].id)
		{
			var4_t dVec = {0.0, 0.0, 0.0, 0.0};
			//for (uint32_t j = int_bound.source.x; j < int_bound.source.y; j++)
			for (uint32_t j = i+1; j < int_bound.source.y; j++) 
			{
				/* Skip the body with the same index and those which are inactive ie. id < 0 */
				//if (i == j || 0 > bmd[j].id)
				if (0 > bmd[j].id)
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
				var_t r_3 = 1.0 / (d*dVec.w);
				// 1 FLOP
				dVec.w = p[j].mass * r_3;
				// 6 FLOP
				a[i].x += dVec.w * dVec.x;
				a[i].y += dVec.w * dVec.y;
				a[i].z += dVec.w * dVec.z;

				// 2 FLOP
				dVec.w = p[i].mass * r_3;
				// 6 FLOP
				a[j].x -= dVec.w * dVec.x;
				a[j].y -= dVec.w * dVec.y;
				a[j].z -= dVec.w * dVec.z;

				// Check for collision - ignore the star (i > 0 criterium)
				// The data of the collision will be stored for the body with the greater index (test particles can collide with massive bodies)
				// If i < j is the condition than test particles can not collide with massive bodies
				//if (i > 0 && i > j && d < threshold[THRESHOLD_RADII_ENHANCE_FACTOR] * (p[i].radius + p[j].radius))
				if (i > 0 && d < threshold[THRESHOLD_RADII_ENHANCE_FACTOR] * (p[i].radius + p[j].radius))
				{
					uint32_t k = (*event_counter)++; // Tested - works as needed

					uint32_t survivIdx = i;
					uint32_t mergerIdx = j;
					if (p[mergerIdx].mass > p[survivIdx].mass)
					{
						uint32_t t = survivIdx;
						survivIdx = mergerIdx;
						mergerIdx = t;
					}
					pp_disk_utility::store_event_data(EVENT_NAME_COLLISION, curr_t, d, survivIdx, mergerIdx, p, r, v, bmd, &events[k]);
				}
			} // 36 FLOP
		}
	}
}

void pp_disk::cpu_calc_grav_accel_NI( ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* bmd, const param_t* p, const var4_t* r, const var4_t* v, var4_t* a)
{
	for (uint32_t i = int_bound.sink.x; i < int_bound.sink.y; i++)
	{
		if (0 < bmd[i].id)
		{
			var4_t dVec = {0.0, 0.0, 0.0, 0.0};
			for (uint32_t j = int_bound.source.x; j < int_bound.source.y; j++) 
			{
				/* Skip the body with the same index and those which are inactive ie. id < 0 */
				if (i == j || 0 > bmd[j].id)
				{
					continue;
				}
				// r_ij 3 FLOP
				dVec.x = r[j].x - r[i].x;
				dVec.y = r[j].y - r[i].y;
				dVec.z = r[j].z - r[i].z;
				// 5 FLOP
				dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2

				// 20 FLOP
				var_t d = sqrt(dVec.w);								// = r
				// 2 FLOP
				var_t r_3 = 1.0 / (d*dVec.w);
				// 1 FLOP
				dVec.w = p[j].mass * r_3;
				// 6 FLOP
				a[i].x += dVec.w * dVec.x;
				a[i].y += dVec.w * dVec.y;
				a[i].z += dVec.w * dVec.z;
			} // 36 44 FLOP
		}
	}
}

void pp_disk::cpu_calc_grav_accel_NI( ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* bmd, const param_t* p, const var4_t* r, const var4_t* v, var4_t* a, event_data_t* events, uint32_t *event_counter)
{
	throw string("cpu_calc_grav_accel_NI() is not implemented.");
}

void pp_disk::cpu_calc_grav_accel_NSI(ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* bmd, const param_t* p, const var4_t* r, const var4_t* v, var4_t* a)
{
	throw string("cpu_calc_grav_accel_NSI() is not implemented.");
}

void pp_disk::cpu_calc_grav_accel_NSI(ttt_t curr_t, interaction_bound int_bound, const body_metadata_t* bmd, const param_t* p, const var4_t* r, const var4_t* v, var4_t* a, event_data_t* events, uint32_t *event_counter)
{
	throw string("cpu_calc_grav_accel_NSI() is not implemented.");
}

void pp_disk::cpu_calc_grav_accel(ttt_t curr_t, const var4_t* r, const var4_t* v, var4_t* dy)
{
	const param_t* p = sim_data->p;
	const body_metadata_t* bmd = sim_data->body_md;

	uint32_t n_body = n_bodies->get_n_total_playing();
	memset(dy, 0, n_body*sizeof(var4_t));

	uint32_t n_sink = n_bodies->get_n_SI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_SI();
		cpu_calc_grav_accel_SI(curr_t, int_bound, bmd, p, r, v, dy);
	}

	n_sink = n_bodies->get_n_NSI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_NSI();
		cpu_calc_grav_accel_NSI(curr_t, int_bound, bmd, p, r, v, dy);
	}

	n_sink = n_bodies->get_n_NI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_NI();
		cpu_calc_grav_accel_NI(curr_t, int_bound, bmd, p, r, v, dy);
	}
}

void pp_disk::gpu_calc_grav_accel(ttt_t curr_t, const var4_t* r, const var4_t* v, var4_t* dy)
{
	const param_t* p = sim_data->p;
	const body_metadata_t* bmd = sim_data->body_md;

	uint32_t n_sink = n_bodies->get_n_SI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_SI();
		set_kernel_launch_param(n_sink);
		kernel_pp_disk::calc_grav_accel_naive<<<grid, block>>>(int_bound, bmd, p, r, dy);
		CUDA_CHECK_ERROR();
	}

	n_sink = n_bodies->get_n_NSI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_NSI();
		set_kernel_launch_param(n_sink);
		kernel_pp_disk::calc_grav_accel_naive<<<grid, block>>>(int_bound, bmd, p, r, dy);
		CUDA_CHECK_ERROR();
	}

	n_sink = n_bodies->get_n_NI();
	if (0 < n_sink)
	{
		interaction_bound int_bound = n_bodies->get_bound_NI();
		set_kernel_launch_param(n_sink);
		kernel_pp_disk::calc_grav_accel_naive<<<grid, block>>>(int_bound, bmd, p, r, dy);
		CUDA_CHECK_ERROR();
	}
}

void pp_disk::gpu_calc_drag_accel(ttt_t curr_t, const var4_t* r, const var4_t* v, var4_t* dy)
{
	const param_t* p = sim_data->p;
	const body_metadata_t* bmd = sim_data->body_md;

	uint32_t n_sink = n_bodies->get_n_NSI();
	if (0 < n_sink)
	{
		set_kernel_launch_param(n_sink);
		interaction_bound int_bound = n_bodies->get_bound_GD();
		kernel_pp_disk::calc_drag_accel_NSI<<<grid, block>>>(curr_t, int_bound, bmd, p, r, v, dy);
		CUDA_CHECK_ERROR();
	}
}



bool pp_disk::check_for_ejection_hit_centrum()
{
	switch (comp_dev)
	{
	case COMPUTING_DEVICE_CPU:
		cpu_check_for_ejection_hit_centrum();
		break;
	case COMPUTING_DEVICE_GPU:
		gpu_check_for_ejection_hit_centrum();
		break;
	default:
		throw string("Parameter 'comp_dev' is out of range.");
	}
	return (0 < event_counter ? true : false);
}

bool pp_disk::check_for_collision()
{
	switch (comp_dev)
	{
	case COMPUTING_DEVICE_CPU:
		cpu_check_for_collision();
		break;
	case COMPUTING_DEVICE_GPU:
		gpu_check_for_collision();
		break;
	default:
		throw string("Parameter 'comp_dev' is out of range.");
	}
	return (0 < event_counter ? true : false);
}

void pp_disk::cpu_check_for_ejection_hit_centrum()
{
	uint32_t n_total = n_bodies->get_n_total_playing();
	interaction_bound int_bound(0, n_total, 0, 0);

	const var4_t* r = sim_data->y[0];
	const var4_t* v = sim_data->y[1];
	const param_t* p = sim_data->h_p;
	body_metadata_t* bmd = sim_data->body_md;
	
	var_t ed2  = (0.0 < threshold[THRESHOLD_EJECTION_DISTANCE] ? SQR(threshold[THRESHOLD_EJECTION_DISTANCE]) : DBL_MAX);
	var_t hcd2 = SQR(threshold[THRESHOLD_HIT_CENTRUM_DISTANCE]);
	for (uint32_t i = int_bound.sink.x; i < int_bound.sink.y; i++)
	{
		// Ignore the star and the inactive bodies (whose id < 0)
		if (0 < bmd[i].id && BODY_TYPE_STAR != bmd[i].body_type)
		{
			// Calculate the distance from the barycenter
			var_t r2 = SQR(r[i].x) + SQR(r[i].y) + SQR(r[i].z);
			if (ed2 < r2)
			{
				pp_disk_utility::store_event_data(EVENT_NAME_EJECTION,    t, sqrt(r2), 0, i, p, r, v, bmd, &events[event_counter]);
				event_counter++;
			}
			if (hcd2 > r2)
			{
				pp_disk_utility::store_event_data(EVENT_NAME_HIT_CENTRUM, t, sqrt(r2), 0, i, p, r, v, bmd, &events[event_counter]);
				event_counter++;
			}
		}
	}
}

void pp_disk::cpu_check_for_collision()
{
	uint32_t n_total = n_bodies->get_n_total_playing();
	interaction_bound int_bound(0, n_total, 0, n_total);

	cpu_check_for_collision(int_bound, false, false, false, false);
}

void pp_disk::cpu_check_for_collision(interaction_bound int_bound, bool SI_NSI, bool SI_TP, bool NSI, bool NSI_TP)
{
	const double f = SQR(threshold[THRESHOLD_RADII_ENHANCE_FACTOR]);

	const var4_t* r = sim_data->y[0];
	const var4_t* v = sim_data->y[1];
	const param_t* p = sim_data->h_p;
	const body_metadata_t* bmd = sim_data->body_md;

	for (uint32_t i = int_bound.sink.x; i < int_bound.sink.y; i++)
	{
		/* Skip inactive bodies, i.e. id < 0 and the star type bodies*/
		if (0 < bmd[i].id && BODY_TYPE_STAR != bmd[i].body_type)
		{
			var4_t dVec = {0.0, 0.0, 0.0, 0.0};
			for (uint32_t j = i + 1; j < int_bound.source.y; j++) 
			{
				/* Skip inactive bodies, i.e. id < 0 and the star type bodies*/
				if (0 > bmd[j].id && BODY_TYPE_STAR != bmd[j].body_type)
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
					uint32_t survivIdx = i;
					uint32_t mergerIdx = j;
					if (p[mergerIdx].mass > p[survivIdx].mass)
					{
						uint32_t idx = survivIdx;
						survivIdx = mergerIdx;
						mergerIdx = idx;
					}
					pp_disk_utility::store_event_data(EVENT_NAME_COLLISION, t, sqrt(dVec.w), survivIdx, mergerIdx, p, r, v, bmd, &events[event_counter]);
					event_counter++;
				}
			}
		}
	}
}

void pp_disk::gpu_check_for_ejection_hit_centrum()
{
	const var4_t* r = sim_data->y[0];
	const var4_t* v = sim_data->y[1];
	const param_t* p = sim_data->p;
	const body_metadata_t* bmd = sim_data->body_md;

	uint32_t n_total = n_bodies->get_n_total_playing();
	interaction_bound int_bound(0, n_total, 0, 0);
	set_kernel_launch_param(n_total);

	kernel_pp_disk::check_for_ejection_hit_centrum<<<grid, block>>>(t, int_bound, p, r, v, bmd, d_events, d_event_counter);
	CUDA_CHECK_ERROR();

	copy_vector_to_host((void *)&event_counter, (void *)d_event_counter, 1*sizeof(uint32_t));
}

void pp_disk::gpu_check_for_collision()
{
	const var4_t* r = sim_data->y[0];
	const var4_t* v = sim_data->y[1];
	const param_t* p = sim_data->p;
	const body_metadata_t* bmd = sim_data->body_md;

	uint32_t n_total = n_bodies->get_n_total_playing();
	interaction_bound int_bound(0, n_total, 0, n_total);
	set_kernel_launch_param(n_total);

	kernel_pp_disk::check_for_collision<<<grid, block>>>(t, int_bound, p, r, v, bmd, d_events, d_event_counter);
	CUDA_CHECK_ERROR();

	copy_vector_to_host((void *)&event_counter, (void *)d_event_counter, 1*sizeof(uint32_t));
}

void pp_disk::handle_collision()
{
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		copy_event_data_to_host();
	}
	populate_sp_events();

	// TODO: implement collision graph: bredth-first search
	for (uint32_t i = 0; i < sp_events.size(); i++)
	{
		uint32_t ictv_idx = sp_events[i].idx2;

		handle_collision_pair(i, &sp_events[i]);
		increment_event_counter(n_collision);
		// Make the merged body inactive 
		sim_data->h_body_md[ictv_idx].id *= -1;
		// Copy it up to GPU 
		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			copy_vector_to_device((void **)&sim_data->d_body_md[ictv_idx], (void **)&sim_data->h_body_md[ictv_idx], sizeof(body_metadata_t));
		}
		// Update number of inactive bodies
		n_bodies->inactive[sim_data->h_body_md[ictv_idx].body_type]++;
	}
	cout << n_collision[EVENT_COUNTER_NAME_LAST_STEP] << " collision event(s) occurred" << endl;

	n_collision[EVENT_COUNTER_NAME_LAST_STEP] = 0;
}

void pp_disk::populate_sp_events()
{
	sp_events.resize(event_counter);

	bool *processed = new bool[event_counter];
	for (uint32_t i = 0; i < event_counter; i++)
	{
		processed[i] = false;
	}

	uint32_t n = 0;
	for (uint32_t k = 0; k < event_counter; k++)
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
		for (uint32_t i = k + 1; i < event_counter; i++)
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
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		copy_event_data_to_host();
	}
	sp_events.resize(event_counter);

	for (uint32_t i = 0; i < event_counter; i++)
	{
		// The events must be copied into sp_events since the print_event_data() write the content of the sp_events to disk.
		sp_events[i] = events[i];
		uint32_t ictv_idx = sp_events[i].idx2;
		if (EVENT_NAME_EJECTION == sp_events[i].event_name)
		{
			increment_event_counter(n_ejection);
		}
		else
		{
			handle_collision_pair(i, &sp_events[i]);
			increment_event_counter(n_hit_centrum);
		}
		// Make the body inactive 
		sim_data->h_body_md[ictv_idx].id *= -1;
		// Copy it up to GPU 
		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			copy_vector_to_device((void **)&sim_data->d_body_md[ictv_idx], (void **)&sim_data->h_body_md[ictv_idx], sizeof(body_metadata_t));
		}
		// Update number of inactive bodies
		n_bodies->inactive[sim_data->h_body_md[ictv_idx].body_type]++;
	}
	cout << n_ejection[EVENT_COUNTER_NAME_LAST_STEP] << " ejection " << n_hit_centrum[EVENT_COUNTER_NAME_LAST_STEP] << " hit_centrum event(s) occured" << endl;

	n_ejection[   EVENT_COUNTER_NAME_LAST_STEP] = 0;
	n_hit_centrum[EVENT_COUNTER_NAME_LAST_STEP] = 0;
}

void pp_disk::handle_collision_pair(uint32_t i, event_data_t *collision)
{
	uint32_t survivIdx = collision->idx1;
	uint32_t mergerIdx = collision->idx2;

	if (BODY_TYPE_SUPERPLANETESIMAL == sim_data->h_body_md[mergerIdx].body_type)
	{
		// TODO: implement collision between a body and a super-planetesimal
		throw string("Collision between a massive body and a super-planetesimal is not yet implemented.");
	}

	collision->p1 = sim_data->h_p[survivIdx];
	collision->p2 = sim_data->h_p[mergerIdx];

	// Compute the kinetic energy of the two bodies before the collision
	var_t T0 = 0.5 * (collision->p1.mass * (SQR(collision->v1.x) + SQR(collision->v1.y) + SQR(collision->v1.z)) + 
		              collision->p2.mass * (SQR(collision->v2.x) + SQR(collision->v2.y) + SQR(collision->v2.z)));

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

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		copy_vector_to_device((void **)&sim_data->d_y[0][survivIdx],	(void *)&sim_data->h_y[0][survivIdx],	 sizeof(var4_t));
		copy_vector_to_device((void **)&sim_data->d_y[1][survivIdx],	(void *)&sim_data->h_y[1][survivIdx],	 sizeof(var4_t));
		copy_vector_to_device((void **)&sim_data->d_p[survivIdx], (void *)&sim_data->h_p[survivIdx], sizeof(param_t));
	}

	// Compute the kinetic energy of the surviver
	var_t T1 = 0.5 * (collision->ps.mass * (SQR(collision->vs.x) + SQR(collision->vs.y) + SQR(collision->vs.z)));

	var_t dT = T1 - T0;
}

void pp_disk::rebuild_vectors()
{
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		copy_to_host();
	}
	uint32_t n_total_players = n_bodies->get_n_total_playing();
	uint32_t n_total_active  = n_bodies->get_n_total_active();

	sim_data_t* sim_data_temp = new sim_data_t;
	// Only the data of the active bodies will be temporarily stored
	allocate_host_storage(sim_data_temp, n_total_active);
	vector<string> names(n_total_active);

	// Create aliases:
	var4_t* r = sim_data->h_y[0];
	var4_t* v = sim_data->h_y[1];
	param_t *p = sim_data->h_p;
	body_metadata_t *bmd = sim_data->h_body_md;

	// Copy the data of active bodies to sim_data_temp
	uint32_t i = 0;
	uint32_t k = 0;
	for ( ; i < n_total_players; i++)
	{
		// Active? i.e. its id is positive?
		if (0 < bmd[i].id && n_total_active > k)
		{
			sim_data_temp->h_y[0][k]    = r[i];
			sim_data_temp->h_y[1][k]    = v[i];
			sim_data_temp->h_p[k]       = p[i];
			sim_data_temp->h_body_md[k] = bmd[i];
			names[k]                    = body_names[i];
			k++;
		}
	}
	if (n_total_active != k)
	{
		throw string("Number of copied bodies does not equal to the number of active bodies.");
	}
	this->n_bodies->update();

	for (i = 0; i < n_total_active; i++)
	{
		r[i]   = sim_data_temp->h_y[0][i];
		v[i]   = sim_data_temp->h_y[1][i];
		p[i]   = sim_data_temp->h_p[i];
		bmd[i] = sim_data_temp->h_body_md[i];

		body_names[i] = names[i];
	}

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		// Copy the active bodies to the device
		copy_to_device();
	}
	deallocate_host_storage(sim_data_temp);

	delete sim_data_temp;
}

void pp_disk::increment_event_counter(uint32_t *event_counter)
{
	for (uint32_t i = 0; i < EVENT_COUNTER_NAME_N; i++)
	{
		event_counter[i]++;
		// Increment the total number of events
		n_event[i]++;
	}
}

void pp_disk::set_event_counter(event_counter_name_t field, uint32_t value)
{
	n_hit_centrum[field] = value;
	n_ejection[field]    = value;
	n_collision[field]   = value;
	n_event[field]       = value;
}

void pp_disk::calc_dydx(uint32_t i, uint32_t rr, ttt_t curr_t, const var4_t* r, const var4_t* v, var4_t* dy)
{
	uint32_t n_total = n_bodies->get_n_total_playing();

	switch (i)
	{
	case 0:
		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			CUDA_SAFE_CALL(cudaMemcpy(dy, v, n_total * sizeof(var4_t), cudaMemcpyDeviceToDevice));
		}
		else
		{
			memcpy(dy, v, n_total * sizeof(var4_t));
		}
		break;
	case 1:  // Calculate accelerations originating from the gravitational force, drag force etc.
		/*
		* SORREND:
		* 1. Gravity
		* 2. other forces
		*/
		if (COMPUTING_DEVICE_GPU == comp_dev)
		{
			gpu_calc_grav_accel(curr_t, r, v, dy);
			// This if will be used to speed-up the calculation when gas drag is also acting on the bodies.
			// (BUT early optimization is the root of much evil)
			if (0 == rr)
			{
			}
			if (GAS_DISK_MODEL_NONE != g_disk_model)
			{
				gpu_calc_drag_accel(curr_t, r, v, dy);
			}
		}
		else
		{
			cpu_calc_grav_accel(curr_t, r, v, dy);
			// This if will be used to speed-up the calculation when gas drag is also acting on the bodies.
			// (BUT early optimization is the root of much evil)
			if (0 == rr)
			{
			}
			if (GAS_DISK_MODEL_NONE != g_disk_model)
			{
				cpu_calc_drag_accel(curr_t, r, v, dy);
			}
		}
		break;
	default:
		throw string("Parameter 'i' is out of range.");
	}
}

void pp_disk::calc_integral(bool cpy_to_HOST, integral_t& integrals)
{
	const uint32_t n = n_bodies->get_n_total_playing();

	if (cpy_to_HOST && COMPUTING_DEVICE_GPU == this->comp_dev)
	{
		copy_to_host();
	}
	var_t M0 = tools::get_total_mass(n, sim_data);
	tools::calc_bc(n, sim_data, M0, &(integrals.R), &(integrals.V));
	integrals.C = tools::calc_angular_momentum(n, sim_data);
	integrals.E = tools::calc_total_energy(n, sim_data);
}

void pp_disk::swap()
{
	for (uint32_t i = 0; i < 2; i++)
	{
		::swap(sim_data->yout[i],   sim_data->y[i]);
		::swap(sim_data->h_yout[i], sim_data->h_y[i]);
		if (COMPUTING_DEVICE_GPU == this->comp_dev)
		{
			::swap(sim_data->d_yout[i], sim_data->d_y[i]);
		}
	}
}



pp_disk::pp_disk(n_objects_t* n_bodies, gas_disk_model_t g_disk_model, uint32_t id_dev, computing_device_t comp_dev) :
	n_bodies(n_bodies),
	g_disk_model(g_disk_model),
	id_dev(id_dev),
	comp_dev(comp_dev)
{
	initialize();

	allocate_storage();
	redutilcu::create_aliases(comp_dev, sim_data);
	tools::populate_data(n_bodies->initial, sim_data);
}

pp_disk::pp_disk(string& path_data, string& path_data_info, gas_disk_model_t g_disk_model, uint32_t id_dev, computing_device_t comp_dev, const var_t* thrshld) :
	n_bodies(0x0),
	g_disk_model(g_disk_model),
	id_dev(id_dev),
	comp_dev(comp_dev)
{
	initialize();
	memcpy(threshold, thrshld, THRESHOLD_N*sizeof(var_t));

	data_rep_t repres;
	string ext = file::get_extension(path_data);
//START DEBUG CODE
DBG();printf("ext: %s\n", ext.c_str());
//END   DEBUG CODE

	{
		uint16_t len = ext.length() + 1;
		char* _ext = (char*)malloc(len);
		strcpy(_ext, ext.c_str());
		DBG(); printf("_ext = %s\n", _ext);
		if (     0 == strcmp("txt", _ext))
		{
			repres = DATA_REPRESENTATION_ASCII;
		}
		else if (0 == strcmp("dat", _ext))
		{
			repres = DATA_REPRESENTATION_BINARY;
		}
		else
		{
			throw string("The extension of the input data must be either 'txt' or 'dat'.");
		}

	}

	if (     "txt" == ext)
	{
		repres = DATA_REPRESENTATION_ASCII;
	}
	else if ("dat" == ext)
	{
		repres = DATA_REPRESENTATION_BINARY;
	}
	else
	{
		throw string("The extension of the input data must be either 'txt' or 'dat'.");
	}
	load_data_info(path_data_info, repres);

	allocate_storage();
	load_data(path_data, repres);

	redutilcu::create_aliases(comp_dev, sim_data);
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
	n_tpb           = 1;

	t               = 0.0;
	dt              = 0.0;

	sim_data        = 0x0;
	event_counter   = 0;
	d_event_counter = 0x0;
	events          = 0x0;
	d_events        = 0x0;

	a_gd            = 0x0;
	f_gd            = 0x0;

	memset(n_hit_centrum, 0, sizeof(n_hit_centrum));
	memset(n_ejection,    0, sizeof(n_ejection));
	memset(n_collision,   0, sizeof(n_collision));
	memset(n_event,       0, sizeof(n_event));

	memset(threshold, 0, THRESHOLD_N*sizeof(var_t));
}

void pp_disk::allocate_storage()
{
	sim_data = new sim_data_t;

	// These will be only aliases to the actual storage space either in the HOST or DEVICE memory
	sim_data->y.resize(2);
	sim_data->yout.resize(2);

	uint32_t n_total = n_bodies->get_n_total_playing();
	allocate_host_storage(sim_data, n_total);
	ALLOCATE_HOST_VECTOR((void **)&events, n_total*sizeof(event_data_t));
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		allocate_device_storage(sim_data, n_total);
		ALLOCATE_DEVICE_VECTOR((void **)&d_events,        n_total*sizeof(event_data_t));
		ALLOCATE_DEVICE_VECTOR((void **)&d_event_counter,       1*sizeof(uint32_t));
	}
}

void pp_disk::set_computing_device(computing_device_t device)
{
	// If the execution is already on the requested device than nothing to do
	if (this->comp_dev == device)
	{
		return;
	}

	uint32_t n_total = n_bodies->get_n_total_playing();

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
		ALLOCATE_DEVICE_VECTOR((void **)&d_event_counter,       1*sizeof(uint32_t));
		copy_to_device();

		copy_vector_to_device((void *)d_event_counter, (void *)&event_counter, 1*sizeof(uint32_t));
		break;
	default:
		throw string("Parameter 'device' is out of range.");
	}
	redutilcu::create_aliases(device, sim_data);

	this->comp_dev = device;
}

void pp_disk::copy_to_device()
{
	uint32_t n_total = n_bodies->get_n_total_playing();

	for (uint32_t i = 0; i < 2; i++)
	{
		copy_vector_to_device((void *)sim_data->d_y[i],	    (void *)sim_data->h_y[i],    n_total*sizeof(var4_t));
	}
	copy_vector_to_device((void *)sim_data->d_p,		(void *)sim_data->h_p,		     n_total*sizeof(param_t));
	copy_vector_to_device((void *)sim_data->d_body_md,	(void *)sim_data->h_body_md,     n_total*sizeof(body_metadata_t));
	copy_vector_to_device((void *)sim_data->d_epoch,	(void *)sim_data->h_epoch,	     n_total*sizeof(ttt_t));

	copy_vector_to_device((void *)d_event_counter, (void *)&event_counter, 1*sizeof(uint32_t));

	copy_constant_to_device(dc_threshold, threshold, THRESHOLD_N*sizeof(var_t));

	copy_disk_params_to_device();
}

void pp_disk::copy_to_host()
{
	uint32_t n_total = n_bodies->get_n_total_playing();

	for (uint32_t i = 0; i < 2; i++)
	{
		copy_vector_to_host((void *)sim_data->h_y[i],    (void *)sim_data->d_y[i],    n_total*sizeof(var4_t));
	}
	copy_vector_to_host((void *)sim_data->h_p,			(void *)sim_data->d_p,		  n_total*sizeof(param_t));
	copy_vector_to_host((void *)sim_data->h_body_md,	(void *)sim_data->d_body_md,  n_total*sizeof(body_metadata_t));
	copy_vector_to_host((void *)sim_data->h_epoch,		(void *)sim_data->d_epoch,	  n_total*sizeof(ttt_t));

	copy_vector_to_host((void *)&event_counter, (void *)d_event_counter, 1*sizeof(uint32_t));
}

void pp_disk::copy_disk_params_to_device()
{
	switch (g_disk_model)
	{
	case GAS_DISK_MODEL_NONE:
		break;
	case GAS_DISK_MODEL_ANALYTIC:
		copy_constant_to_device((void*)&dc_anal_gd_params,  (void*)&(a_gd->params), sizeof(analytic_gas_disk_params_t));
		break;
	case GAS_DISK_MODEL_FARGO:
		copy_constant_to_device((void*)&dc_fargo_gd_params, (void*)&(f_gd->params), sizeof(fargo_gas_disk_params_t));
		break;
	default:
		throw string("Parameter 'g_disk_model' is out of range.");
	}
}

void pp_disk::copy_event_data_to_host()
{
	copy_vector_to_host((void *)events, (void *)d_events, event_counter*sizeof(event_data_t));
}

void pp_disk::clear_event_counter()
{
	event_counter = 0;
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		copy_vector_to_device((void *)d_event_counter, (void *)&event_counter, 1*sizeof(uint32_t));
	}
}

uint32_t pp_disk::get_n_total_event()
{
	return (n_collision[EVENT_COUNTER_NAME_TOTAL] + n_ejection[EVENT_COUNTER_NAME_TOTAL] + n_hit_centrum[EVENT_COUNTER_NAME_TOTAL]);
}

var_t pp_disk::get_mass_of_star()
{
	const uint32_t n_massive = n_bodies->get_n_massive();

	body_metadata_t* bmd = sim_data->h_body_md;
	for (uint32_t j = 0; j < n_massive; j++ )
	{
		if (bmd[j].body_type == BODY_TYPE_STAR)
		{
			return sim_data->h_p[j].mass;
		}
	}
	throw string("No star is included.");
}

void pp_disk::load_data_info(string& path, data_rep_t repres)
{
	ifstream input;

	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		input.open(path.c_str(), ios::in);
		if (input) 
		{
			file::load_data_info_record_ascii(input, this->t, this->dt, this->dt_CPU, &(this->n_bodies));
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
			file::load_data_info_record_binary(input, this->t, this->dt, this->dt_CPU, &(this->n_bodies));
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	}
	input.close();
}

void pp_disk::load_data(string& path_data, data_rep_t repres)
{
	var4_t* r = sim_data->h_y[0];
	var4_t* v = sim_data->h_y[1];
	param_t* p = sim_data->h_p;
	body_metadata_t* bmd = sim_data->h_body_md;

	ifstream input;
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		input.open(path_data.c_str(), ios::in);
		break;
	case DATA_REPRESENTATION_BINARY:
		input.open(path_data.c_str(), ios::in | ios::binary);
		break;
	default:
		throw string("Parameter 'repres' is out of range.");
	}

	if (input) 
	{
		uint32_t pcd = 1;
		const uint32_t n_total = n_bodies->get_n_total_playing();

		cout << "Loading " << path_data << " ";
		for (uint32_t i = 0; i < n_total; i++)
		{
			string name;
			if (DATA_REPRESENTATION_ASCII == repres)
			{
				file::load_data_record_ascii(input, name, &p[i], &bmd[i], &r[i], &v[i]);
			}
			else
			{
				file::load_data_record_binary(input, name, &p[i], &bmd[i], &r[i], &v[i]);
			}
			body_names.push_back(name);
			if (pcd <= (uint32_t)((((var_t)(i+1)/(var_t)n_total))*100.0))
			{
				pcd++;
				cout << ".";	flush(cout);
			}
		}
		input.close();
		cout << " done" << endl;
	}
	else 
	{
		throw string("Cannot open " + path_data + ".");
	}
}

void pp_disk::print_data(string& path, data_rep_t repres)
{
	ofstream sout;

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		copy_to_host();
	}

	var4_t* r = sim_data->h_y[0];
	var4_t* v = sim_data->h_y[1];
	param_t* p = sim_data->h_p;
	body_metadata_t* bmd = sim_data->h_body_md;

	const uint32_t n = n_bodies->get_n_total_playing();
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		sout.open(path.c_str(), ios::out);
		for (uint32_t i = 0; i < n; i++)
		{
			// Skip inactive bodies
			if (bmd[i].id < 0)
			{
				continue;
			}
			file::print_body_record_ascii_RED(sout, body_names[i], &p[i], &bmd[i], &r[i], &v[i]);
		}
		break;
	case DATA_REPRESENTATION_BINARY:
		sout.open(path.c_str(), ios::out | ios::binary);
		for (uint32_t i = 0; i < n; i++)
		{
			// Skip inactive bodies
			if (bmd[i].id < 0)
			{
				continue;
			}
			file::print_body_record_binary_RED(sout, body_names[i], &p[i], &bmd[i], &r[i], &v[i]);
		}
		break;
	}
	sout.close();
}

void pp_disk::print_data_info(string& path, data_rep_t repres)
{
	ofstream sout;

	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		sout.open(path.c_str(), ios::out);
		if (sout) 
		{
			file::print_data_info_record_ascii_RED(sout, this->t, this->dt, this->dt_CPU, this->n_bodies);
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	case DATA_REPRESENTATION_BINARY:
		sout.open(path.c_str(), ios::out | ios::binary);
		if (sout) 
		{
			file::print_data_info_record_binary_RED(sout, this->t, this->dt, this->dt_CPU, this->n_bodies);
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	}
	sout.close();
}

void pp_disk::print_event_data(string& path, ofstream& log_f)
{
	static uint32_t int_t_w =  8;
	static uint32_t var_t_w = 25;
	string e_names[] = {"NONE", "HIT_CENTRUM", "EJECTION", "COLLISION"};

	ofstream sout;
	sout.open(path.c_str(), ios::out | ios::app);
	if (sout) 
	{
		sout.precision(16);
		sout.setf(ios::right);
		sout.setf(ios::scientific);

		for (uint32_t i = 0; i < sp_events.size(); i++)
		{
			sout << setw(16)      << e_names[sp_events[i].event_name] << SEP       /* type of the event                              []                  */
				 << setw(var_t_w) << sp_events[i].t / constants::Gauss << SEP      /* time of the event                              [day]               */
				 << setw(var_t_w) << sp_events[i].d << SEP                         /* distance of the two bodies                     [AU]                */
				 << setw(int_t_w) << sp_events[i].id1 << SEP		               /* id of the survivor                             []                  */
				 << setw(int_t_w) << sp_events[i].id2 << SEP		               /* id of the merger                               []                  */

																				   /* BEFORE THE EVENT                                                   */ 
				 << setw(var_t_w) << sp_events[i].p1.mass << SEP                   /* survivor's mass                                [solar mass]        */
				 << setw(var_t_w) << sp_events[i].p1.density << SEP                /*            density                             [solar mass / AU^3] */
				 << setw(var_t_w) << sp_events[i].p1.radius << SEP                 /*            radius                              [AU]                */
				 << setw(var_t_w) << sp_events[i].r1.x << SEP                      /*            x-coordiante in barycentric system  [AU]                */
				 << setw(var_t_w) << sp_events[i].r1.y << SEP                      /*            y-coordiante                        [AU]                */
				 << setw(var_t_w) << sp_events[i].r1.z << SEP                      /*            z-coordiante                        [AU]                */
				 << setw(var_t_w) << sp_events[i].v1.x * constants::Gauss << SEP   /*            x-velocity                          [AU / day]          */
				 << setw(var_t_w) << sp_events[i].v1.y * constants::Gauss << SEP   /*            y-velocity                          [AU / day]          */
				 << setw(var_t_w) << sp_events[i].v1.z * constants::Gauss << SEP   /*            z-velocity                          [AU / day]          */

				 << setw(var_t_w) << sp_events[i].p2.mass << SEP                   /* merger's mass                                  [solar mass]        */
				 << setw(var_t_w) << sp_events[i].p2.density << SEP                /*          density                               [solar mass / AU^3] */
				 << setw(var_t_w) << sp_events[i].p2.radius << SEP                 /*          radius                                [AU]                */
				 << setw(var_t_w) << sp_events[i].r2.x << SEP                      /*          x-coordiante in barycentric system    [AU]                */
				 << setw(var_t_w) << sp_events[i].r2.y << SEP                      /*          y-coordiante                          [AU]                */
				 << setw(var_t_w) << sp_events[i].r2.z << SEP                      /*          z-coordiante                          [AU]                */
				 << setw(var_t_w) << sp_events[i].v2.x * constants::Gauss << SEP   /*          x-velocity                            [AU / day]          */
				 << setw(var_t_w) << sp_events[i].v2.y * constants::Gauss << SEP   /*          y-velocity                            [AU / day]          */
				 << setw(var_t_w) << sp_events[i].v2.z * constants::Gauss << SEP   /*          z-velocity                            [AU / day]          */

																				   /* AFTER THE EVENT                                                    */ 
				 << setw(var_t_w) << sp_events[i].ps.mass << SEP                   /* survivor's mass                                [solar mass]        */
				 << setw(var_t_w) << sp_events[i].ps.density << SEP                /*            density                             [solar mass / AU^3] */
				 << setw(var_t_w) << sp_events[i].ps.radius << SEP                 /*            radius                              [AU]                */
				 << setw(var_t_w) << sp_events[i].rs.x << SEP                      /*            x-coordiante in barycentric system  [AU]                */
				 << setw(var_t_w) << sp_events[i].rs.y << SEP                      /*            y-coordiante                        [AU]                */
				 << setw(var_t_w) << sp_events[i].rs.z << SEP                      /*            z-coordiante                        [AU]                */
				 << setw(var_t_w) << sp_events[i].vs.x * constants::Gauss << SEP   /*            x-velocity                          [AU / day]          */
				 << setw(var_t_w) << sp_events[i].vs.y * constants::Gauss << SEP   /*            y-velocity                          [AU / day]          */
				 << setw(var_t_w) << sp_events[i].vs.z * constants::Gauss << SEP   /*            z-velocity                          [AU / day]          */
				 << endl;

			if (log_f)
			{
				log_f.precision(16);
				log_f.setf(ios::right);
				log_f.setf(ios::scientific);

				log_f << tools::get_time_stamp(false) << SEP 
					  << e_names[sp_events[i].event_name] << SEP
					  << setw(int_t_w) << sp_events[i].id1 << SEP
					  << setw(int_t_w) << sp_events[i].id2 << SEP
					  << endl;
			}
		}
		sout.close();
		log_f.flush();
	}
	else 
	{
		throw string("Cannot open " + path + ".");
	}
}

void pp_disk::print_integral_data(string& path, integral_t& I)
{
	static uint32_t var_t_w  = 25;

	ofstream sout;
	sout.open(path.c_str(), ios::out | ios::app);
	if (sout) 
	{
		sout.precision(16);
		sout.setf(ios::right);
		sout.setf(ios::scientific);

		sout << setw(var_t_w) << t / constants::Gauss << SEP       /* time of the record                      [day]                        */
 			 << setw(var_t_w) << I.R.x << SEP                      /* x-coordinate of the system's barycenter [AU]                         */
			 << setw(var_t_w) << I.R.y << SEP                      /* y-coordinate of the system's barycenter [AU]                         */
			 << setw(var_t_w) << I.R.z << SEP                      /* z-coordinate of the system's barycenter [AU]                         */
			 << setw(var_t_w) << I.V.x * constants::Gauss << SEP   /* x velocity of the system's barycenter   [AU/day]                     */
			 << setw(var_t_w) << I.V.y * constants::Gauss << SEP   /* y velocity of the system's barycenter   [AU/day]                     */
			 << setw(var_t_w) << I.V.z * constants::Gauss << SEP   /* z velocity of the system's barycenter   [AU/day]                     */
			 << setw(var_t_w) << I.C.x * constants::Gauss << SEP   /* x component of the system's angular momentum [AU^2 * solar mass/day] */
			 << setw(var_t_w) << I.C.y * constants::Gauss << SEP   /* y component of the system's angular momentum [AU^2 * solar mass/day] */
			 << setw(var_t_w) << I.C.z * constants::Gauss << SEP   /* z component of the system's angular momentum [AU^2 * solar mass/day] */
			 << setw(var_t_w) << I.E * constants::Gauss2 << SEP    /* total energy of the system [solar mass * AU^2/day^2]                 */
			 << endl;

		sout.close();
	}
	else 
	{
		throw string("Cannot open " + path + ".");
	}
}

#undef GAS_REDUCTION_THRESHOLD
#undef GAS_INNER_EDGE
