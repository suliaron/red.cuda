#pragma once

// includes system
#include <cstdlib>
#include <string>

// includes project
#include "gas_disk.h"
#include "parameter.h"

using namespace std;

class options
{
public:
	options(int argc, const char** argv);
	~options();

	bool	verbose;				//!< print more information to the screen or log file
	string	printoutDir;			//!<  Printout directory
	string	inputDir;				//!<  Input directory

	parameter* param;
	gas_disk*  g_disk;	

private:
	//! holds the path of the file containing the parameters of the simulation
	string parameters_filename;
	//! holds the path of the file containing the parameters of the nebula
	string gasdisk_filename;
	//! holds the path of the file containing the data of the bodies
	string bodylist_filename;

	void print_usage();
	void create_default_options();
	void parse_options(int argc, const char** argv);
};