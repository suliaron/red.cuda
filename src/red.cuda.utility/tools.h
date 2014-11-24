#pragma once

//includes system
#include <string>

// includes project
#include "red_type.h"

using namespace std;

namespace redutilcu
{
	namespace tools
	{
		bool is_number(const string& str);
		void trim_right(string& str);
		void trim_right(string& str, char c);
		void trim_left(string& str);
		void trim(string& str);
		string get_time_stamp();
		string convert_time_t(time_t t);

		var_t get_total_mass(int n, sim_data_t *sim_data);
		var_t get_total_mass(int n, body_type_t type, sim_data_t *sim_data);
		void compute_bc(int n, sim_data_t *sim_data, vec_t* R0, vec_t* V0);
		void transform_to_bc(int n, sim_data_t *sim_data);

		var_t calculate_radius(var_t m, var_t density);
		var_t calculate_density(var_t m, var_t R);
		var_t caclulate_mass(var_t R, var_t density);

		int	kepler_equation_solver(var_t ecc, var_t mean, var_t eps, var_t* E);
		int calculate_phase(var_t mu, const orbelem_t* oe, vec_t* rVec, vec_t* vVec);

	} /* tools */
} /* redutilcu_tools */
