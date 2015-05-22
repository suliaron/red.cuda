#pragma once

// includes system
#include <cstdlib>
#include <string>

// includes project
#include "integrator.h"
#include "analytic_gas_disk.h"
#include "fargo_gas_disk.h"
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

	string	printout_dir;           //!< Printout directory
	string	input_dir;              //!< Input directory
	bool	ef;                     //!< Extend the file names with command line information. Only for developer and debugger purposes.
	bool    benchmark;              //!< Run benchmark test to tune the optimal number of threds per block
	bool    test;                   //!< Run tests for functions
	bool    continue_simulation;    //!< Continues a simulation from its last saved output
	bool	verbose;                //!< Print every event to the log file
	bool	print_to_screen;        //!< Print every event to the standard output stream (cout) 
	bool	ups;                    //!< Use padded storage to store data (default is false)

	unsigned int n_tpb0;            //!< Number of initial thread per block to use in kernel lunches (default is 64)
	unsigned int n_change_to_cpu;   //!< The threshold value for the total number of SI bodies to change to the CPU
	unsigned int id_a_dev;          //!< The id of the device which will execute the code

	ttt_t info_dt;                  //!< The time interval in seconds between two subsequent information print to the screen (default value is 5 sec)
	ttt_t dump_dt;                  //!< The time interval in seconds between two subsequent data dump to the hdd (default value is 3600 sec)

	computing_device_t comp_dev;    //!< The computing device to carry out the calculations (cpu or gpu)
	gas_disk_model_t g_disk_model;

	number_of_bodies *n_bodies;     //!< Contains the number of bodies used for benchmark 
	parameter* param;
	analytic_gas_disk* a_gd;
	fargo_gas_disk* f_gd;

	string info_filename;
	string event_filename;
	string log_filename;
	string result_filename;

private:
	string parameters_filename;     //!< Path of the file containing the parameters of the simulation
	string bodylist_filename;       //!< Path of the file containing the data of the bodies
	string gasdisk_filename;        //!< Path of the file containing the parameters of the nebula

	void initialize();
	void create_default();
	void parse(int argc, const char** argv);

	void print_usage();
};