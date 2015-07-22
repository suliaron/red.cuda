#pragma once
// includes system
#include <string>
#include <vector>

// includes project
#include "red_type.h"

using namespace std;

class analytic_gas_disk
{
public:

	analytic_gas_disk(string& dir, string& filename, bool verbose);
	~analytic_gas_disk();

	//! Calculate the mean free path length and temperature profile
	/*
		\param m_star The mass of the star (time dependent)
	*/
	void calc(var_t	m_star);

	string name;
	string desc;
	analytic_gas_disk_params_t params;
	
	// Input/Output streams
	friend ostream& operator<<(ostream& stream, const analytic_gas_disk* g_disk);

private:
	void initialize();
	void parse();
	void set_param(string& key, string& value);
	void transform_data();
	void transform_time();
	void transform_velocity();

	bool verbose;
	string dir;
	string filename;
	string data;  //!< holds a copy of the file containing the parameters of the simulation
};
