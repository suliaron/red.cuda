// includes system
#include <iostream>

// includes project
#include "redutilcu.h"
#include "nbody_exception.h"
#include "number_of_bodies.h"
#include "options.h"
#include "pp_disk.h"

using namespace redutilcu;

options::options(int argc, const char** argv) :
	has_gas(false),
	verbose(false),
	param(0),
	g_disk(0)
{
	create_default_options();
	parse_options(argc, argv);
	if (parameters_filename.length() == 0)
	{
		throw string("Missing filename for -p  | --parameter!");
	}
	if (bodylist_filename.length() == 0)
	{
		throw string("Missing filename for -ic | --initial_conditions!");
	}

	param = new parameter(input_dir, parameters_filename, verbose);
	if (gasdisk_filename.length() > 0)
	{
		g_disk = new gas_disk(input_dir, gasdisk_filename, verbose);
		has_gas = true;
	}
}

options::~options() 
{
}

void options::print_usage()
{
	cout << "Usage: red.cuda <parameterlis>" << endl;
	cout << "Parameters:" << endl;
	cout << "     -iDir | --inputDir <directory>         : the directory containig the input files"  << endl;
	cout << "     -p    | --parameter <filename>         : the file containig the parameters of the simulation"  << endl;
	cout << "     -gd   | --gas_disk <filename>          : the file containig the parameters of the gas disk"  << endl;
	cout << "     -ic   | --initial_condition <filename> : the file containig the initial conditions"  << endl;
	cout << "     -v    | --verbose                      : verbose mode" << endl;
	cout << "     -h    | --help                         : print this help" << endl;
}

// TODO: implement
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
			input_dir = argv[i];
			printout_dir = input_dir;
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

pp_disk* options::create_pp_disk()
{
	string path = file::combine_path(input_dir, bodylist_filename);
	pp_disk* ppd = new pp_disk(path, g_disk);
	if (ppd->g_disk != 0)
	{
		ppd->g_disk->calculate(ppd->get_mass_of_star());
	}
	if (param->fr_cntr == FRAME_CENTER_BARY)
	{
		ppd->transform_to_bc();
	}

	ppd->t = param->start_time;

	return ppd;
}
