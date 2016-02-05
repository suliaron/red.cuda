#pragma once

#include <string>

#include "integrator.h"
#include "parameter.h"
#include "pp_disk.h"

#include "red_type.h"

class options
{
public:
	options(int argc, const char** argv);
	~options();

	pp_disk* create_pp_disk();
	integrator* create_integrator(pp_disk* ppd, ttt_t dt);

	bool benchmark;                 //!< Run benchmark test to tune the optimal number of threds per block
	bool test;                      //!< Run tests for functions
	bool verbose;                   //!< Print every event to the log file
	bool print_to_screen;           //!< Print every event to the standard output stream (cout) 
	bool ef;                        //!< Extend the file names with command line information. Only for developer and debugger purposes.

	uint32_t id_dev;                //!< The id of the device which will execute the code
	uint32_t n_change_to_cpu;       //!< The threshold value for the total number of SI bodies to change to the CPU

	computing_device_t comp_dev;    //!< The computing device to carry out the calculations (cpu or gpu)
	gas_disk_model_t g_disk_model;

	n_objects_t *n_bodies;          //!< Contains the number of bodies used for benchmark 
	parameter* param;

	std::string out_fn[OUTPUT_NAME_N];   //!< Array for the output filenames
	std::string in_fn[INPUT_NAME_N];     //!< Array for the input filenames
	std::string dir[DIRECTORY_NAME_N];   //!< Array for the input and output directories

private:
	void create_default();
	void parse(int argc, const char** argv);

	void print_usage();
};
