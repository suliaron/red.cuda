#include "thrust\device_ptr.h"
#include "thrust\fill.h"
#include "thrust\extrema.h"

#include "integrator.h"
#include "ode.h"

#include "redutilcu.h"

using namespace std;
using namespace redutilcu;

integrator::integrator(ode& f, ttt_t dt, bool adaptive, var_t tolerance, uint16_t n_stage, computing_device_t comp_dev) : 
	f(f),
	dt_try(dt),
	adaptive(adaptive),
	tolerance(tolerance),
	n_stage(n_stage),
	comp_dev(comp_dev)
{
	initialize();

	n_var  = f.n_obj * f.n_vpo;
	allocate_storage(n_var);
}

integrator::~integrator()
{}

void integrator::initialize()
{
	n_tried_step  = 0;
	n_passed_step = 0;
	n_failed_step = 0;

	t             = f.t;
	dt_did        = 0.0;

	h_ytemp       = 0x0;
	d_ytemp       = 0x0;

	h_err         = 0x0;
	d_err         = 0x0;

	max_iter      = 100;
	dt_min        = 1.0e10;
}

void integrator::allocate_storage(uint32_t n_var)
{
	allocate_host_storage(n_var);
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		allocate_device_storage(n_var);
	}
}

void integrator::allocate_host_storage(uint32_t n_var)
{
	h_k.resize(n_stage);
	for (uint16_t i = 0; i < n_stage; i++)
	{
		ALLOCATE_HOST_VECTOR((void**)&(h_k[i]), n_var*sizeof(var_t));
	}
	ALLOCATE_HOST_VECTOR((void**)&(h_ytemp), n_var*sizeof(var_t));
	if (adaptive)
	{
		ALLOCATE_HOST_VECTOR((void**)&(h_err), n_var*sizeof(var_t));
	}
}

void integrator::allocate_device_storage(uint32_t n_var)
{
	d_k.resize(n_stage);
	for (uint16_t i = 0; i < n_stage; i++)
	{
		ALLOCATE_DEVICE_VECTOR((void**)&(d_k[i]), n_var*sizeof(var_t));
	}
	ALLOCATE_DEVICE_VECTOR((void**)&(d_ytemp), n_var*sizeof(var_t));
	if (adaptive)
	{
		ALLOCATE_DEVICE_VECTOR((void**)&(d_err), n_var*sizeof(var_t));
	}
}

void integrator::deallocate_storage()
{
	deallocate_host_storage();
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		deallocate_device_storage();
	}
}

void integrator::deallocate_host_storage()
{
	for (uint16_t i = 0; i < n_stage; i++)
	{
		FREE_HOST_VECTOR((void **)&(h_k[i]));
	}
	FREE_HOST_VECTOR((void **)&(h_ytemp));
	if (adaptive)
	{
		FREE_HOST_VECTOR((void **)&(h_err));
	}
}

void integrator::deallocate_device_storage()
{
	for (uint16_t i = 0; i < n_stage; i++)
	{
		FREE_HOST_VECTOR((void **)&(d_k[i]));
	}
	FREE_HOST_VECTOR((void **)&(d_ytemp));
	if (adaptive)
	{
		FREE_HOST_VECTOR((void **)&(d_err));
	}
}

var_t integrator::get_max_error(uint32_t n_var)
{
	var_t max_err = 0.0;

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		// Wrap raw pointer with a device_ptr
		thrust::device_ptr<var_t> d_ptr(d_err);
		// Use thrust to find the maximum element
		thrust::device_ptr<var_t> d_ptr_max = thrust::max_element(d_ptr, d_ptr + n_var);
		// Copy the max element from device memory to host memory
		cudaMemcpy((void*)&max_err, (void*)d_ptr_max.get(), sizeof(var_t), cudaMemcpyDeviceToHost);
	}
	else
	{
		for (uint32_t i = 0; i < n_var; i++)
		{
			if (max_err < fabs(h_err[i]))
			{
				max_err = fabs(h_err[i]);
			}
		}		
	}

	//return (dt_try * lambda * max_err);
	return (max_err);
}

void integrator::update_counters(uint16_t iter)
{
	n_tried_step  += iter;
	n_failed_step += (iter - 1);
	n_passed_step++;
}
