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

	//! Calculate the mean free path length and temperature profile
	/*
		\param m_star The mass of the star (time dependent)
	*/
	void calc(var_t	m_star);
	//! Copies parameters and variables from the host to the cuda device
	void copy_to_device();
	//! Copies parameters and variables from the cuda device to the host
	void copy_to_host();

	//! The decrease type for the gas density
	string	name;
	string	desc;


	// Input/Output streams
	friend ostream& operator<<(ostream& stream, const gas_disk* g_disk);

	var_t virtual get_density(vec_t r) = 0;
	vec_t virtual get_velocity(vec_t r) = 0;

private:
	void parse();
	void set_param(string& key, string& value);

	//! holds a copy of the file containing the parameters of the simulation
	string	data;
	bool verbose;  //!< print the key - value information to the screen
};
