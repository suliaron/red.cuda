#pragma once

// includes system
#include <string>

// includes project
#include "red_type.h"

using namespace std;

class parameter
{
public:

	parameter(string& dir, string& filename, bool verbose);
	~parameter();

	//! holds the path of the file containing the parameters of the simulation
	string filename;

	//! name of the simulation
	string simulation_name;				
	//! description of the simulation
	string simulation_desc;
	//! the center of the reference frame (bary or astro centric frame)
	frame_center_t fr_cntr;
	//! type of the integrator
	integrator_type_t int_type;
	//! tolerance/eps/accuracy of the simulation
	var_t tolerance;
	//! Check the error also for the test particle
	bool_t error_check_for_tp;
	//! Adaptive step size
	bool_t adaptive;
	//! start time of the simulation [day]
	ttt_t start_time;
	//! length of the simulation [day]
	ttt_t simulation_length;
	//! stop time of the simulation [day] (= start_time + sim_length)
	ttt_t stop_time;
	//! interval between two succesive output epoch [day]
	ttt_t output_interval;
	//! the hit centrum distance: inside this limit the body is considered to have hitted the central body and removed from the simulation [AU]
	//! the ejection distance: beyond this limit the body is removed from the simulation [AU]
	//! two bodies collide when their mutual distance is smaller than the sum of their radii multiplied by this number. Real physical collision corresponds to the value of 1.0.
	//! Contains the threshold values: hit_centrum_dst, ejection_dst, collision_factor
	var_t thrshld[THRESHOLD_N];

	// Input/Output streams
	friend ostream& operator<<(ostream& stream, const parameter* param);

private:
	void parse();
	void set_param(string& key, string& value);
	void transform_time();

	//! holds a copy of the file containing the parameters of the simulation
	string	data;
	bool verbose;  //!< print the key - value information to the screen
};
