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

integrator::integrator(pp_disk *ppd, ttt_t dt, bool adaptive, var_t tolerance, int r_max, computing_device_t comp_dev) : 
	ppd(ppd),
	dt_try(dt * constants::Gauss), // Transform time unit
	adaptive(adaptive),
	tolerance(tolerance),
	r_max(r_max),
	comp_dev(comp_dev),
	error_check_for_tp(false),
	dt_did(0.0),
	dt_next(0.0),
	n_failed_step(0),
	n_passed_step(0),
	n_tried_step(0),
	h_dydx(2),
	d_dydx(2),
	dydx(2),
	h_err(2),
	d_err(2),
	err(2),
	h_ytemp(2),
	d_ytemp(2),
	ytemp(2)
{
	allocate_storage();
	create_aliases();
}

integrator::~integrator()
{
	deallocate_storage();
}

void integrator::allocate_storage()
{	
	int n_body = ppd->get_ups() ? ppd->n_bodies->get_n_prime_total() : ppd->n_bodies->get_n_total();

	allocate_host_storage(n_body);
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		allocate_device_storage(n_body);
	}

	// Resize the alias vector
	for (int i = 0; i < 2; i++)
	{
		dydx[i].resize(r_max);
	}
}

void integrator::allocate_host_storage(int n_body)
{
	for (int i = 0; i < 2; i++)
	{
		ALLOCATE_HOST_VECTOR((void**)&(h_ytemp[i]), n_body*sizeof(vec_t));

		h_dydx[i].resize(r_max);
		for (int r = 0; r < r_max; r++) 
		{
			ALLOCATE_HOST_VECTOR((void**)&(h_dydx[i][r]), n_body*sizeof(vec_t));
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
		ALLOCATE_DEVICE_VECTOR((void**)&(d_ytemp[i]), n_body*sizeof(vec_t));

		d_dydx[i].resize(r_max);
		for (int r = 0; r < r_max; r++) 
		{
			ALLOCATE_DEVICE_VECTOR((void**)&(d_dydx[i][r]), n_body*sizeof(vec_t));
		}
		if (adaptive)
		{
			int n_var = NDIM * n_body;
			ALLOCATE_DEVICE_VECTOR((void**)&(d_err[i]), n_var * sizeof(var_t));
		}
	}
}

void integrator::deallocate_storage()
{
	for (int i = 0; i < 2; i++)
	{
		FREE_HOST_VECTOR(  (void **)&(h_ytemp[i]));
		FREE_DEVICE_VECTOR((void **)&(d_ytemp[i]));
		for (int r = 0; r < r_max; r++) 
		{
			FREE_HOST_VECTOR(  (void**)&(h_dydx[i][r]));
			FREE_DEVICE_VECTOR((void**)&(d_dydx[i][r]));
		}
		if (adaptive)
		{
			FREE_HOST_VECTOR(  (void**)&(h_err[i]));
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
			for (int r = 0; r < r_max; r++) 
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
			for (int r = 0; r < r_max; r++) 
			{
				dydx[i][r] = d_dydx[i][r];
			}
			if (adaptive)
			{
				err[i] = d_err[i];
			}
		}
		break;
	}
}

void integrator::set_computing_device(computing_device_t device)
{
	switch (device)
	{
	case COMPUTING_DEVICE_CPU:
		if (COMPUTING_DEVICE_CPU != this->comp_dev)
		{
			this->comp_dev = device;
			create_aliases();
			// Memory allocation was already done [all variables are redy to use on the host]
		}
		break;
	case COMPUTING_DEVICE_GPU:
		if (COMPUTING_DEVICE_GPU != this->comp_dev)
		{
			deallocate_storage();
			this->comp_dev = device;
			allocate_storage();
			create_aliases();
		}
		break;
	default:
		throw string ("Invalid parameter: computing device was out of range.");
	}
}

var_t integrator::get_max_error(int n_var, var_t lambda)
{
	var_t max_err_r = 0.0;
	var_t max_err_v = 0.0;

	int64_t idx_max_err_r = -1;
	int64_t idx_max_err_v = -1;

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		// Wrap raw pointer with a device_ptr
		thrust::device_ptr<var_t> d_ptr_r(err[0]);
		thrust::device_ptr<var_t> d_ptr_v(err[1]);

		// Use thrust to find the maximum element
		thrust::device_ptr<var_t> d_ptr_max_r = thrust::max_element(d_ptr_r, d_ptr_r + n_var);
		thrust::device_ptr<var_t> d_ptr_max_v = thrust::max_element(d_ptr_v, d_ptr_v + n_var);

		// Get the index of the maximum element
		idx_max_err_r = d_ptr_max_r.get() - d_ptr_r.get();
		idx_max_err_v = d_ptr_max_v.get() - d_ptr_v.get();

		// Copy the max element from device memory to host memory
		cudaMemcpy((void*)&max_err_r, (void*)d_ptr_max_r.get(), sizeof(var_t), cudaMemcpyDeviceToHost);
		cudaMemcpy((void*)&max_err_v, (void*)d_ptr_max_v.get(), sizeof(var_t), cudaMemcpyDeviceToHost);
	}
	else
	{
		// TODO: The cpu based integrator::get_max_error() function is not yet tested
		for (int i = 0; i < n_var; i++)
		{
			if (max_err_r < err[0][i])
			{
				max_err_r = err[0][i];
				idx_max_err_r = i;
			}
			if (max_err_v < err[1][i])
			{
				max_err_v = err[1][i];
				idx_max_err_v = i;
			}
		}		
	}

	return fabs(dt_try * lambda * std::max(max_err_r, max_err_v));
}

void integrator::update_counters(int iter)
{
	n_tried_step  += iter;
	n_failed_step += (iter - 1);
	n_passed_step++;
}

int integrator::get_n_failed_step()
{
	return n_failed_step;
}

int integrator::get_n_passed_step()
{
	return n_passed_step;
}

int integrator::get_n_tried_step()
{
	return n_tried_step;
}

void integrator::calc_grid(int nData, int threads_per_block)
{
	int	nThread = std::min(threads_per_block, nData);
	int	nBlock = (nData + nThread - 1)/nThread;
	grid.x  = nBlock;
	block.x = nThread;
}
