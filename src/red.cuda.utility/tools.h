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
		/// Default white-space characters
		static const char* ws = " \t\n\r\f\v";
		/// Default comment character
		static const char* comment = "#";

		bool is_number(const string& str);

		/// Removes all leading white-space characters from the current string object.
		/// The default white spaces are: " \t\n\r\f\v"
		string& ltrim(string& s);
		string& ltrim(string& s, const char* t);

		string& rtrim(string& s);
		string& rtrim(string& s, const char* t);

		string& trim(string& s);
		string& trim(string& s, const char* t);

		string& trim_comment(string& s);
		string& trim_comment(string& s, const char* t);

		string get_time_stamp(bool use_comma);
		string convert_time_t(time_t t);

		void populate_data(unsigned int* n_bodies, sim_data_t *sim_data);

		//! Computes the total mass of the system
		var_t get_total_mass(unsigned int n, const sim_data_t *sim_data);
		//! Computes the total mass of the bodies with type in the system
		var_t get_total_mass(unsigned int n, body_type_t type, const sim_data_t *sim_data);
		void calc_bc(unsigned int n, bool verbose, const sim_data_t *sim_data, var_t M, vec_t* R0, vec_t* V0);
		void transform_to_bc(unsigned int n, bool verbose, const sim_data_t *sim_data);

		var_t calc_radius(var_t m, var_t density);
		var_t calc_density(var_t m, var_t R);
		var_t calc_mass(var_t R, var_t density);

		void calc_position_after_collision(var_t m1, var_t m2, const vec_t* r1, const vec_t* r2, vec_t& r);
		void calc_velocity_after_collision(var_t m1, var_t m2, const vec_t* v1, const vec_t* v2, vec_t& v);
		void calc_physical_properties(const param_t &p1, const param_t &p2, param_t &p);

		var_t norm(const vec_t* r);
		var_t calc_dot_product(const vec_t& u, const vec_t& v);
		vec_t calc_cross_product(const vec_t& u, const vec_t& v);
		var_t calc_kinetic_energy(const vec_t* v);
		var_t calc_pot_energy(var_t mu, const vec_t* r);

        var_t calc_total_energy(        unsigned int n, const sim_data_t *sim_data);
        var_t calc_total_energy_CMU(    unsigned int n, const sim_data_t *sim_data);
        vec_t calc_angular_momentum(    unsigned int n, const sim_data_t *sim_data);
        vec_t calc_angular_momentum_CMU(unsigned int n, const sim_data_t *sim_data);
       
		void kepler_equation_solver(var_t ecc, var_t mean, var_t eps, var_t* E);
		void calc_phase(var_t mu, const orbelem_t* oe, vec_t* rVec, vec_t* vVec);
		void calc_oe(   var_t mu, const vec_t* rVec, const vec_t* vVec, orbelem_t* oe);

		void print_vector(const vec_t *v);
		void print_parameter(const param_t *p);
		void print_body_metadata(const body_metadata_t *b);
		void print_body_metadata(const body_metadata_new_t *b);
	} /* tools */
} /* redutilcu_tools */
