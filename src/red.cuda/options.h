#pragma once

#include <cstdlib>
#include <string>

using namespace std;

class options
{

public:
	options(int argc, const char** argv);
	~options();

	bool	verbose;				//!< print more information to the screen or log file
	string	printoutDir;			//!<  Printout directory
	string	inputDir;				//!<  Input directory

private:
	//! holds the path of the file containing the parameters of the simulation
	string parameters_filename;
	string parameters_path;
	//! holds the path of the file containing the parameters of the nebula
	string gasdisk_filename;
	string gasDisk_path;
	//! holds the path of the file containing the data of the bodies
	string bodylist_filename;
	string bodylist_path;

	void print_usage();
	void create_default_options();
	void parse_options(int argc, const char** argv);
};