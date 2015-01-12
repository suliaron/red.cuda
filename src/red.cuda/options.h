#pragma once

// includes system
#include <cstdlib>
#include <string>

// includes project
#include "integrator.h"
#include "gas_disk.h"
#include "parameter.h"
#include "pp_disk.h"

using namespace std;

class options
{
public:
	options(int argc, const char** argv);
	~options();

	pp_disk* create_pp_disk();
	integrator* create_integrator(pp_disk* ppd, ttt_t dt);

	bool	verbose;				//!< Print more information to the screen or log file
	string	printout_dir;			//!< Printout directory
	string	input_dir;				//!< Input directory
	bool	use_padded_storage;		//!< Use padded storage to store data (default is false)
	int		n_tpb;					//!< Number of thread per block to use in kernel lunches (default is 64)
	int		id_a_dev;				//!< The id of the device which will execute the code

	parameter* param;
	gas_disk*  g_disk;	

private:
	//! holds the path of the file containing the parameters of the simulation
	string parameters_filename;
	//! holds the path of the file containing the parameters of the nebula
	string gasdisk_filename;
	//! holds the path of the file containing the data of the bodies
	string bodylist_filename;

	bool has_gas;

	void print_usage();
	void create_default_options();
	void parse_options(int argc, const char** argv);
};