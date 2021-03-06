#pragma once

#include <string>

#include "red.cuda.initial.type.h"

std::string create_name(int i, int type);

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
ttt_t extract_from_horizon_output(std::string &data, orbelem_t& oe);

template <typename T>
void print_number(std::string& path, T number);

void print_data(std::string &path, body_disk_t& disk, pp_disk_t::sim_data_t* sd, input_format_name_t format);
void print_data_info(std::string &path, ttt_t t, ttt_t dt, uint32_t dt_CPU, body_disk_t& disk, input_format_name_t format);
void print_oe(std::string &path, uint32_t n, ttt_t t, pp_disk_t::sim_data_t *sd);
