#pragma once

// includes system
#include <string>

// includes project
#include "red_type.h"

using namespace std;

class gas_disk
{
public:

	gas_disk(string& dir, string& filename, bool verbose);
	~gas_disk();

	__host__ __device__ var_t virtual get_density(var2_t sch, var2_t rho, const vec_t* rVec) = 0;
	__host__ __device__ vec_t virtual get_velocity(var_t mu, var2_t eta, const vec_t* rVec) = 0;

	void virtual calc(var_t m_star) = 0;
	//! Copies parameters and variables from the host to the cuda device
	virtual void copy_to_device() = 0;
	//! Copies parameters and variables from the cuda device to the host
	virtual void copy_to_host() = 0;

	//__host__ __device__ vec_t circular_velocity(var_t mu, const vec_t* rVec);

	bool verbose;
	string dir;
	string filename;
	string name;
	string desc;
};
