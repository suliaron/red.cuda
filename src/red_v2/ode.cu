#include "ode.h"
#include "redutilcu.h"

using namespace redutilcu;

ode::ode(uint16_t n_dim, uint16_t n_vpo, uint16_t n_ppo, uint32_t n_obj, computing_device_t comp_dev) :
	n_dim(n_dim),
	n_vpo(n_vpo),
	n_ppo(n_ppo),
	n_obj(n_obj),
	comp_dev(comp_dev)
{
	initialize();

	n_var  = n_obj * n_vpo;
	n_par  = n_obj * n_ppo;
	allocate_storage(n_var, n_par);
}

ode::~ode()
{
	deallocate_storage();
}

void ode::initialize()
{
	t      = 0.0;
	tout   = 0.0;

	h_y    = 0x0;
	h_yout = 0x0;

	d_y	   = 0x0;
	d_yout = 0x0;

	y	   = 0x0;
	yout   = 0x0;
	
	h_p    = 0x0;
	d_p	   = 0x0;
	p	   = 0x0;

	n_tpb  = 0;
}

void ode::allocate_storage(uint32_t n_var, uint32_t n_par)
{
	allocate_host_storage(n_var, n_par);
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		allocate_device_storage(n_var, n_par);
	}
}

void ode::allocate_host_storage(uint32_t n_var, uint32_t n_par)
{
	ALLOCATE_HOST_VECTOR((void**)&(h_y),    n_var * sizeof(var_t));
	ALLOCATE_HOST_VECTOR((void**)&(h_yout), n_var * sizeof(var_t));
	ALLOCATE_HOST_VECTOR((void**)&(h_p),    n_par * sizeof(var_t));
}

void ode::allocate_device_storage(uint32_t n_var, uint32_t n_par)
{
	ALLOCATE_DEVICE_VECTOR((void**)&(d_y),    n_var * sizeof(var_t));
	ALLOCATE_DEVICE_VECTOR((void**)&(d_yout), n_var * sizeof(var_t));
	ALLOCATE_DEVICE_VECTOR((void**)&(d_p),    n_par * sizeof(var_t));
}

void ode::deallocate_storage()
{
	deallocate_host_storage();
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		deallocate_device_storage();
	}
}

void ode::deallocate_host_storage()
{
	FREE_HOST_VECTOR((void **)&(h_y));
	FREE_HOST_VECTOR((void **)&(h_yout));
	FREE_HOST_VECTOR((void **)&(h_p));
}

void ode::deallocate_device_storage()
{
	FREE_DEVICE_VECTOR((void **)&(d_y));
	FREE_DEVICE_VECTOR((void **)&(d_yout));
	FREE_DEVICE_VECTOR((void **)&(d_p));
}

void ode::swap()
{
	std::swap(t, tout);

	std::swap(h_y, h_yout);
	std::swap(d_y, d_yout);
}
