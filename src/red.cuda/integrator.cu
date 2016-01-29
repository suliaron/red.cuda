// includes system
#include <string>
#include <vector>

// includes CUDA

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
#include "integrator.h"
#include "red_constants.h"
#include "red_type.h"
#include "redutilcu.h"

using namespace std;
using namespace redutilcu;

integrator::integrator(pp_disk *ppd, ttt_t dt, bool adaptive, var_t tolerance, int n_stage, computing_device_t comp_dev) : 
	ppd(ppd),
	dt_try(dt),
	adaptive(adaptive),
	tolerance(tolerance),
	n_stage(n_stage),
	comp_dev(comp_dev)
{
	initialize();
	allocate_storage();
	create_aliases();
}

integrator::~integrator()
{
	deallocate_device_storage();
	deallocate_host_storage();
}

void integrator::initialize()
{
	error_check_for_tp = false;

	dt_did        = 0.0;
	dt_next       = 0.0;

	n_failed_step = 0;
	n_passed_step = 0;
	n_tried_step  = 0;

	h_dydx.resize(2);
	d_dydx.resize(2);
	dydx.resize(2);

	h_err.resize(2);
	d_err.resize(2);
	err.resize(2);

	h_ytemp.resize(2);
	d_ytemp.resize(2);
	ytemp.resize(2);
}

void integrator::allocate_storage()
{	
	int n_body = ppd->n_bodies->get_n_total_playing();

	allocate_host_storage(n_body);
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		allocate_device_storage(n_body);
	}

	// Resize the alias vector
	for (int i = 0; i < 2; i++)
	{
		dydx[i].resize(n_stage);
	}
}

void integrator::allocate_host_storage(int n_body)
{
	for (int i = 0; i < 2; i++)
	{
		ALLOCATE_HOST_VECTOR((void**)&(h_ytemp[i]), n_body*sizeof(var4_t));

		h_dydx[i].resize(n_stage);
		for (int r = 0; r < n_stage; r++) 
		{
			ALLOCATE_HOST_VECTOR((void**)&(h_dydx[i][r]), n_body*sizeof(var4_t));
		}
		if (adaptive)
		{
			int n_var = NDIM * n_body;
			ALLOCATE_HOST_VECTOR((void**)&(h_err[i]), n_var * sizeof(var_t));
		}
	}
}

void integrator::allocate_device_storage(int n_body)
{
	for (int i = 0; i < 2; i++)
	{
		ALLOCATE_DEVICE_VECTOR((void**)&(d_ytemp[i]), n_body*sizeof(var4_t));

		d_dydx[i].resize(n_stage);
		for (int r = 0; r < n_stage; r++) 
		{
			ALLOCATE_DEVICE_VECTOR((void**)&(d_dydx[i][r]), n_body*sizeof(var4_t));
		}
		if (adaptive)
		{
			int n_var = NDIM * n_body;
			ALLOCATE_DEVICE_VECTOR((void**)&(d_err[i]), n_var * sizeof(var_t));
		}
	}
}

void integrator::deallocate_host_storage()
{
	for (int i = 0; i < 2; i++)
	{
		FREE_HOST_VECTOR(  (void **)&(h_ytemp[i]));
		for (int r = 0; r < n_stage; r++) 
		{
			FREE_HOST_VECTOR(  (void**)&(h_dydx[i][r]));
		}
		if (adaptive)
		{
			FREE_HOST_VECTOR(  (void**)&(h_err[i]));
		}
	}
}

void integrator::deallocate_device_storage()
{
	for (int i = 0; i < 2; i++)
	{
		FREE_DEVICE_VECTOR((void **)&(d_ytemp[i]));
		for (int r = 0; r < n_stage; r++) 
		{
			FREE_DEVICE_VECTOR((void**)&(d_dydx[i][r]));
		}
		if (adaptive)
		{
			FREE_DEVICE_VECTOR((void**)&(d_err[i]));
		}
	}
}

void integrator::create_aliases()
{
	switch (comp_dev)
	{
	case COMPUTING_DEVICE_CPU:
		for (int i = 0; i < 2; i++)
		{
			ytemp[i] = h_ytemp[i];
			for (int r = 0; r < n_stage; r++) 
			{
				dydx[i][r] = h_dydx[i][r];
			}
			if (adaptive)
			{
				err[i] = h_err[i];
			}
		}
		break;
	case COMPUTING_DEVICE_GPU:
		for (int i = 0; i < 2; i++)
		{
			ytemp[i] = d_ytemp[i];
			for (int r = 0; r < n_stage; r++) 
			{
				dydx[i][r] = d_dydx[i][r];
			}
			if (adaptive)
			{
				err[i] = d_err[i];
			}
		}
		break;
	default:
		throw string("Parameter 'comp_dev' is out of range.");
	}
}

void integrator::set_computing_device(computing_device_t device)
{
	// If the execution is already on the requested device than nothing to do
	if (this->comp_dev == device)
	{
		return;
	}

	int n_body = ppd->n_bodies->get_n_total_playing();

	switch (device)
	{
	case COMPUTING_DEVICE_CPU:
		deallocate_device_storage();
		break;
	case COMPUTING_DEVICE_GPU:
		allocate_device_storage(n_body);
		break;
	default:
		throw string("Parameter 'device' is out of range.");
	}

	this->comp_dev = device;
	create_aliases();
	ppd->set_computing_device(device);
}

var_t integrator::get_max_error(uint32_t n_var)
{
	var_t max_err_r = 0.0;
	var_t max_err_v = 0.0;

	//int64_t idx_max_err_r = -1;
	//int64_t idx_max_err_v = -1;

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		// Wrap raw pointer with a device_ptr
		thrust::device_ptr<var_t> d_ptr_r(err[0]);
		thrust::device_ptr<var_t> d_ptr_v(err[1]);

		// Use thrust to find the maximum element
		thrust::device_ptr<var_t> d_ptr_max_r = thrust::max_element(d_ptr_r, d_ptr_r + n_var);
		thrust::device_ptr<var_t> d_ptr_max_v = thrust::max_element(d_ptr_v, d_ptr_v + n_var);

		// Get the index of the maximum element
		//idx_max_err_r = d_ptr_max_r.get() - d_ptr_r.get();
		//idx_max_err_v = d_ptr_max_v.get() - d_ptr_v.get();

		// Copy the max element from device memory to host memory
		cudaMemcpy((void*)&max_err_r, (void*)d_ptr_max_r.get(), sizeof(var_t), cudaMemcpyDeviceToHost);
		cudaMemcpy((void*)&max_err_v, (void*)d_ptr_max_v.get(), sizeof(var_t), cudaMemcpyDeviceToHost);
	}
	else
	{
		// TODO: The cpu based integrator::get_max_error() function is not yet tested
		for (uint32_t i = 0; i < n_var; i++)
		{
			if (max_err_r < err[0][i])
			{
				max_err_r = err[0][i];
				//idx_max_err_r = i;
			}
			if (max_err_v < err[1][i])
			{
				max_err_v = err[1][i];
				//idx_max_err_v = i;
			}
		}		
	}

	//return fabs(dt_try * lambda * std::max(max_err_r, max_err_v));
	return (std::max(max_err_r, max_err_v));
}

void integrator::update_counters(int iter)
{
	n_tried_step  += iter;
	n_failed_step += (iter - 1);
	n_passed_step++;
}

uint64_t integrator::get_n_failed_step()
{
	return n_failed_step;
}

uint64_t integrator::get_n_passed_step()
{
	return n_passed_step;
}

uint64_t integrator::get_n_tried_step()
{
	return n_tried_step;
}

void integrator::calc_grid(int nData, int n_tpb)
{
	int	nThread = std::min(n_tpb, nData);
	int	nBlock = (nData + nThread - 1)/nThread;
	grid.x  = nBlock;
	block.x = nThread;
}
