#pragma once

// includes system
#include <cstdlib>
#include <string>

// includes project
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

	bool	verbose;				//!< print more information to the screen or log file
	string	printout_dir;			//!<  Printout directory
	string	input_dir;				//!<  Input directory

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