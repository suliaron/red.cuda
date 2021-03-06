#pragma once

#include <string>

#include "parameter.h"

#include "red_type.h"

class integrator;
class ode;

class options
{
public:
	options(int argc, const char** argv);
	~options();

	ode* create_tbp1D();
	ode* create_rtbp1D();
	ode* create_tbp3D();
	ode* create_rtbp3D();

	integrator* create_integrator(ode& f, ttt_t dt);
	
	dyn_model_t dyn_model;

	bool continue_simulation;       //!< Continues a simulation from its last saved output
	bool benchmark;                 //!< Run benchmark test to tune the optimal number of threds per block
	bool test;                      //!< Run tests for functions
	bool verbose;                   //!< Print every event to the log file
	bool print_to_screen;           //!< Print every event to the standard output stream (cout) 
	bool ef;                        //!< Extend the file names with command line information. Only for developer and debugger purposes.

	ttt_t info_dt;                  //!< The time interval in seconds between two subsequent information print to the screen (default value is 5 sec)
	ttt_t dump_dt;                  //!< The time interval in seconds between two subsequent data dump to the hdd (default value is 3600 sec)

	uint32_t id_dev;                //!< The id of the device which will execute the code
	uint32_t n_change_to_cpu;       //!< The threshold value for the total number of SI bodies to change to the CPU

	computing_device_t comp_dev;    //!< The computing device to carry out the calculations (cpu or gpu)
	gas_disk_model_t g_disk_model;

	parameter* param;

	std::string out_fn[OUTPUT_NAME_N];   //!< Array for the output filenames
	std::string in_fn[INPUT_NAME_N];     //!< Array for the input filenames
	std::string dir[DIRECTORY_NAME_N];   //!< Array for the input and output directories

private:
	void create_default();
	void parse(int argc, const char** argv);

	void print_usage();
};
