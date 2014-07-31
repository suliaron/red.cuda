// includes system
#include <iostream>

// includes project
#include "nbody_exception.h"
#include "options.h"

options::options(int argc, const char** argv)
{
	create_default_options();
	parse_options(argc, argv);
	if (parameters_filename.length() > 0)
	{
		param = new parameter(inputDir, parameters_filename, verbose);
	}
	if (gasdisk_filename.length() > 0)
	{
		g_disk = new gas_disk(inputDir, gasdisk_filename, verbose);
	}
	
}

options::~options() 
{
}

void options::print_usage()
{
	cout << "Usage: red.cuda <parameterlis>" << endl;
	cout << "Parameters:" << endl;
	cout << "     -iDir | --inputDir <directory>          : the directory containig the input files"  << endl;
	cout << "     -p | --parameter <filename>             : the file containig the parameters of the simulation"  << endl;
	cout << "     -gd | --gas_disk <filename>             : the file containig the parameters of the gas disk"  << endl;
	cout << "     -ic | --initial_conditions <filename>   : the file containig the initial conditions"  << endl;
	cout << "     -v | --verbose                          : verbose mode" << endl;
	cout << "     -h | --help                             : print this help" << endl;
}

void options::create_default_options()
{
}

void options::parse_options(int argc, const char** argv)
{
	int i = 1;

	while (i < argc) {
		string p = argv[i];

		// Print-out location
		if (     p == "--inputDir" || p == "-iDir")	{
			i++;
			inputDir = argv[i];
			printoutDir = inputDir;
		}
		else if (p =="--parameters" || p == "-p") {
			i++;
			parameters_filename = argv[i];
		}
		else if (p == "--gas_disk" || p == "-gd") {
			i++;
			gasdisk_filename = argv[i];
		}
		else if (p == "--initial_conditions" || p == "-ic") {
			i++;
			bodylist_filename = argv[i];
		}
		else if (p == "--verbose" || p == "-v") {
			verbose = true;
		}
		else if (p == "--help" || p == "-h") {
			print_usage();
			exit(EXIT_SUCCESS);
		}
		else {
			throw nbody_exception("Invalid switch on command-line.");
		}
		i++;
	}
}