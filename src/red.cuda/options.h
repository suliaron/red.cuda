#pragma once

// includes system
#include <cstdlib>
#include <string>

// includes project
#include "integrator.h"
#include "number_of_bodies.h"
#include "parameter.h"
#include "pp_disk.h"
#include "red_type.h"

using namespace std;

class options
{
public:
	options(int argc, const char** argv);
	~options();

	pp_disk* create_pp_disk();
	integrator* create_integrator(pp_disk* ppd, ttt_t dt);

	bool continue_simulation;       //!< Continues a simulation from its last saved output
	bool benchmark;                 //!< Run benchmark test to tune the optimal number of threds per block
	bool test;                      //!< Run tests for functions
	bool verbose;                   //!< Print every event to the log file
	bool print_to_screen;           //!< Print every event to the standard output stream (cout) 
	bool ups;                       //!< Use padded storage to store data (default is false)
	bool ef;                        //!< Extend the file names with command line information. Only for developer and debugger purposes.

	ttt_t info_dt;                  //!< The time interval in seconds between two subsequent information print to the screen (default value is 5 sec)
	ttt_t dump_dt;                  //!< The time interval in seconds between two subsequent data dump to the hdd (default value is 3600 sec)

	unsigned int id_dev;            //!< The id of the device which will execute the code
	unsigned int n_change_to_cpu;   //!< The threshold value for the total number of SI bodies to change to the CPU

	computing_device_t comp_dev;    //!< The computing device to carry out the calculations (cpu or gpu)
	gas_disk_model_t g_disk_model;

	number_of_bodies *n_bodies;     //!< Contains the number of bodies used for benchmark 
	parameter* param;

	string out_fn[OUTPUT_NAME_N];   //!< Array for the output filenames
	string in_fn[INPUT_NAME_N];     //!< Array for the input filenames
	string dir[DIRECTORY_NAME_N];   //!< Array for the input and output directories

private:
	void create_default();
	void parse(int argc, const char** argv);

	void print_usage();
};
