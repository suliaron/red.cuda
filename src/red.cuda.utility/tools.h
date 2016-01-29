#pragma once

//includes system
#include <string>

// includes project
#include "red_type.h"

namespace redutilcu
{
	namespace tools
	{
		/// Default white-space characters
		static const char* ws = " \t\n\r\f\v";
		/// Default comment character
		static const char* comment = "#";

		bool is_number(const std::string& str);

		/// Removes all leading white-space characters from the current string object.
		/// The default white spaces are: " \t\n\r\f\v"
		std::string& ltrim(std::string& s);
		std::string& ltrim(std::string& s, const char* t);

		std::string& rtrim(std::string& s);
		std::string& rtrim(std::string& s, const char* t);

		std::string& trim(std::string& s);
		std::string& trim(std::string& s, const char* t);

		std::string& trim_comment(std::string& s);
		std::string& trim_comment(std::string& s, const char* t);

		std::string get_time_stamp(bool use_comma);
		std::string convert_time_t(time_t t);

		void populate_data(uint32_t* n_bodies, pp_disk_t::sim_data_t *sim_data);

		//! Computes the total mass of the system
		var_t get_total_mass(uint32_t n, const pp_disk_t::sim_data_t *sim_data);
		//! Computes the total mass of the bodies with type in the system
		var_t get_total_mass(uint32_t n, body_type_t type, const pp_disk_t::sim_data_t *sim_data);
		void calc_bc(uint32_t n, bool verbose, const pp_disk_t::sim_data_t *sim_data, var_t M, var4_t* R0, var4_t* V0);
		void transform_to_bc(uint32_t n, bool verbose, const pp_disk_t::sim_data_t *sim_data);

		var_t calc_radius(var_t m, var_t density);
		var_t calc_density(var_t m, var_t R);
		var_t calc_mass(var_t R, var_t density);

		void calc_position_after_collision(var_t m1, var_t m2, const var4_t* r1, const var4_t* r2, var4_t& r);
		void calc_velocity_after_collision(var_t m1, var_t m2, const var4_t* v1, const var4_t* v2, var4_t& v);
		void calc_physical_properties(const pp_disk_t::param_t &p1, const pp_disk_t::param_t &p2, pp_disk_t::param_t &p);

		var_t norm(const var4_t* r);
		var_t calc_dot_product(const var4_t& u, const var4_t& v);
		var4_t calc_cross_product(const var4_t& u, const var4_t& v);
		var_t calc_kinetic_energy(const var4_t* v);
		var_t calc_pot_energy(var_t mu, const var4_t* r);

        var_t calc_total_energy(        uint32_t n, const pp_disk_t::sim_data_t *sim_data);
        var_t calc_total_energy_CMU(    uint32_t n, const pp_disk_t::sim_data_t *sim_data);
        var4_t calc_angular_momentum(    uint32_t n, const pp_disk_t::sim_data_t *sim_data);
        var4_t calc_angular_momentum_CMU(uint32_t n, const pp_disk_t::sim_data_t *sim_data);
       
		void kepler_equation_solver(var_t ecc, var_t mean, var_t eps, var_t* E);
		void calc_phase(var_t mu, const orbelem_t* oe, var4_t* rVec, var4_t* vVec);
		void calc_oe(   var_t mu, const var4_t* rVec, const var4_t* vVec, orbelem_t* oe);
		ttt_t calc_orbital_period(var_t mu, var_t a);

		void print_vector(const var4_t *v);
		void print_parameter(const pp_disk_t::param_t *p);
		void print_body_metadata(const pp_disk_t::body_metadata_t *b);
		void print_body_metadata(const pp_disk_t::body_metadata_new_t *b);
	} /* tools */
} /* redutilcu_tools */
