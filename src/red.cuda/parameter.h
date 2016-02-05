#pragma once

#include <string>

#include "red_type.h"

class parameter
{
public:
	parameter(std::string& dir, std::string& filename, bool verbose);
	~parameter();

	std::string get_data();

	std::string filename;              //!< holds the path of the file containing the parameters of the simulation
	std::string simulation_name;       //! name of the simulation
	std::string simulation_desc;       //! description of the simulation
								      
	data_rep_t output_data_rep;        //! Defines the representation (text or binary) of the output. 

	integrator_type_t int_type;	       //! type of the integrator
	var_t tolerance;                   //! tolerance/eps/accuracy of the simulation
	bool_t error_check_for_tp;         //! Check the error also for the test particle
	bool_t adaptive;                   //! Adaptive step size
								       
	ttt_t simulation_length;           //! length of the simulation [day]
	ttt_t output_interval;             //! interval between two succesive output epoch [day]

	ttt_t info_dt;                     //!< The time interval in seconds between two subsequent information print to the screen (default value is 5 sec)
	ttt_t dump_dt;                     //!< The time interval in seconds between two subsequent data dump to the hdd (default value is 3600 sec)

	var_t threshold[THRESHOLD_N];	   //! Contains the threshold values: hit_centrum_dst, ejection_dst, collision_factor

	// Input/Output streams
	friend ostream& operator<<(ostream& stream, const parameter* param);

private:
	void create_default();
	void parse();
	void set_param(std::string& key, std::string& value);
	void transform_time();

	std::string	data;   //!< holds a copy of the file containing the parameters of the simulation
	bool verbose;       //!< print the key - value information to the screen
};
