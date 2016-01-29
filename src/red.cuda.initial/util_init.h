#pragma once
// includes, system 
#include <string>

// includes, project
#include "red.cuda.initial.type.h"

string create_name(int i, int type);
void initialize(body_disk &disk);
uint32_t calc_number_of_bodies(body_disk &bd);
uint32_t calc_number_of_bodies(body_disk &bd, body_type_t bt);
//void generate_oe(orbelem_name_t name, oe_dist_t *oe_d, var_t* oe);
void generate_oe(oe_dist_t *oe_d, orbelem_t& oe);
//void generate_pp(phys_prop_name_t name, phys_prop_dist_t* pp_d, var_t* value);
void generate_pp(phys_prop_dist_t *pp_d, pp_disk_t::param_t& param);

//! Extract the orbital elements (a, e, i, w, O, M) from the data
/*
	\param data contains information got from the HORIZONS Web-Interface (http://ssd.jpl.nasa.gov/horizons.cgi#top)
	\param oe the orbital element structure will hold the data extracetd from data
	\return the epoch for which the orbital elements are valid
*/
ttt_t extract_from_horizon_output(string &data, orbelem_t& oe);

void print_uint32_t(string &path, uint32_t n);
template <typename T>
void print_number(string& path, T number);

void print_data(string &path, body_disk_t& disk, pp_disk_t::sim_data_t* sd, input_format_name_t format);
void print_data_info(string &path, ttt_t t, ttt_t dt, body_disk_t& disk, input_format_name_t format);
void print_oe(string &path, uint32_t n, ttt_t t, pp_disk_t::sim_data_t *sd);
