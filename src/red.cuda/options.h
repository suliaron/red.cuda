#pragma once

// includes system
#include <cstdlib>
#include <string>

// includes project
#include "integrator.h"
#include "gas_disk.h"
#include "analytic_gas_disk.h"
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
	gas_disk* create_gas_disk();

	string	printout_dir;			//!< Printout directory
	string	input_dir;				//!< Input directory
	bool	verbose;				//!< Print more information to the screen or log file
	bool	use_padded_storage;		//!< Use padded storage to store data (default is false)
	int		n_tpb;					//!< Number of thread per block to use in kernel lunches (default is 64)
	int     n_change_to_cpu;        //!< The threshold value for the total number of SI bodies to change to the CPU
	int		id_a_dev;				//!< The id of the device which will execute the code
	bool	ef;						//!< Extend the file names with command line information. Only for developer and debugger purposes.

	computing_device_t comp_dev;    //!< The computing device to carry out the calculations (cpu or gpu)
	gas_disk_model_t g_disk_model;

	parameter* param;
	gas_disk*  g_disk;	

	string info_filename;
	string event_filename;
	string log_filename;
	string result_filename;

private:
	string parameters_filename;     //!< Path of the file containing the parameters of the simulation
	string gasdisk_filename;        //!< Path of the file containing the parameters of the nebula
	string bodylist_filename;       //!< Path of the file containing the data of the bodies

	bool has_gas;

	void print_usage();
	void create_default_options();
	void parse_options(int argc, const char** argv);
};