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

	string filename;              //!< holds the path of the file containing the parameters of the simulation

	string simulation_name;       //! name of the simulation
	string simulation_desc;       //! description of the simulation
	integrator_type_t int_type;	  //! type of the integrator
	var_t tolerance;              //! tolerance/eps/accuracy of the simulation
	bool_t error_check_for_tp;    //! Check the error also for the test particle
	bool_t adaptive;              //! Adaptive step size
	ttt_t start_time;             //! start time of the simulation [day]
	ttt_t simulation_length;      //! length of the simulation [day]
	ttt_t stop_time;              //! stop time of the simulation [day] (= start_time + sim_length)
	ttt_t output_interval;        //! interval between two succesive output epoch [day]

	//! the hit centrum distance: inside this limit the body is considered to have hitted the central body and removed from the simulation [AU]
	//! the ejection distance: beyond this limit the body is removed from the simulation [AU]
	//! two bodies collide when their mutual distance is smaller than the sum of their radii multiplied by this number. Real physical collision corresponds to the value of 1.0.
	//! Contains the threshold values: hit_centrum_dst, ejection_dst, collision_factor
	var_t thrshld[THRESHOLD_N];

	// Input/Output streams
	friend ostream& operator<<(ostream& stream, const parameter* param);

private:
	void create_default();
	void parse();
	void set_param(string& key, string& value);
	void transform_time();

	string	data;   //!< holds a copy of the file containing the parameters of the simulation
	bool verbose;   //!< print the key - value information to the screen
};
