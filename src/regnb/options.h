#pragma once

// includes system
#include <cstdlib>
#include <string>

// includes project
#include "parameter.h"
#include "red_type.h"

using namespace std;

class options
{
public:
	options(int argc, const char** argv);
	~options();

	bool verbose;                   //!< Print every event to the log file
	bool print_to_screen;           //!< Print every event to the standard output stream (cout) 

	parameter* param;

	string out_fn[OUTPUT_NAME_N];   //!< Array for the output filenames
	string in_fn[INPUT_NAME_N];     //!< Array for the input filenames
	string dir[DIRECTORY_NAME_N];   //!< Array for the input and output directories

private:
	void create_default();
	void parse(int argc, const char** argv);

	void print_usage();
};
