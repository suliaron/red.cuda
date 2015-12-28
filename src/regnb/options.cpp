// includes system
#include <iostream>

// includes project
#include "options.h"
#include "parameter.h"

options::options(int argc, const char** argv) :
	param(0x0)
{
	create_default();
	parse(argc, argv);

	param = new parameter(dir[DIRECTORY_NAME_IN], in_fn[INPUT_NAME_PARAMETER], print_to_screen);
}

options::~options() 
{
}

void options::create_default()
{
	verbose             = false;
	print_to_screen     = false;
}

void options::parse(int argc, const char** argv)
{
	int i = 1;

	while (i < argc)
	{
		string p = argv[i];

		if (     p == "--continue" || p == "-c")
		{
		}
		else if (p == "--benchmark" || p == "-b")
		{
		}
		else if (p == "--test" || p == "-t")
		{
		}
		else
		{
			throw string("Invalid switch on command-line: " + p + ".");
		}
		i++;
	}

	if (0 == dir[DIRECTORY_NAME_OUT].length())
	{
		dir[DIRECTORY_NAME_OUT] = dir[DIRECTORY_NAME_IN];
	}
}

void options::print_usage()
{
	cout << "Usage: reg <parameterlist>" << endl;
	cout << "Parameters:" << endl;
	cout << "     -c      | --continue                     : continue a simulation from a previous run's saved state" << endl;
	cout << "     -b      | --benchmark                    : run benchmark to find out the optimal number of threads per block" << endl;
	cout << "     -t      | --test                         : run tests" << endl;
	cout << "     -v      | --verbose                      : verbose mode (log all event during the execution fo the code to the log file)" << endl;
	cout << "     -pts    | --print_to_screen              : verbose mode and print everything to the standard output stream too" << endl;
	cout << "     -ef     |                                : use extended file names (use only for debuging purposes)" << endl;

	cout << "     -i_dt   | --info-dt <number>             : the time interval in seconds between two subsequent information print to the screen (default value is 5 sec)" << endl;
	cout << "     -d_dt   | --dump-dt <number>             : the time interval in seconds between two subsequent data dump to the hard disk (default value is 3600 sec)" << endl;

	cout << "     -id_dev | --id_active_device <number>    : the id of the device which will execute the code (default value is 0)" << endl;
	cout << "     -n_chg  | --n_change_to_cpu <number>     : the threshold value for the total number of SI bodies to change to the CPU (default value is 100)" << endl;

	cout << "     -gpu    | --gpu                          : execute the code on the graphics processing unit (GPU) (default value is true)" << endl;
	cout << "     -cpu    | --cpu                          : execute the code on the cpu if required by the user or if no GPU is installed (default value is false)" << endl;

	cout << "     -ic     | --initial_condition <filename> : the file containing the initial conditions"  << endl;
	cout << "     -info   | --info-filename <filename>     : the name of the file where the runtime output of the code will be stored (default value is info.txt)" << endl;
	cout << "     -event  | --event-filename <filename>    : the name of the file where the details of each event will be stored (default value is event.txt)" << endl;
	cout << "     -log    | --log-filename <filename>      : the name of the file where the details of the execution of the code will be stored (default value is log.txt)" << endl;
	cout << "     -result | --result-filename <filename>   : the name of the file where the simlation data for a time instance will be stored (default value is result.txt)" << endl;

	cout << "     -iDir   | --inputDir <directory>         : the directory containing the input files"  << endl;
	cout << "     -pDir   | --printDir <directory>         : the directory where the output files will be stored (if omitted the input directory will be used)" << endl;
	cout << "     -p      | --parameter <filename>         : the file containing the parameters of the simulation"  << endl;
	cout << "     -ga     | --analytic_gas_disk <filename> : the file containing the parameters of an analyticaly prescribed gas disk"  << endl;
	cout << "     -gf     | --fargo_gas_disk <filename>    : the file containing the details of the gas disk resulted from FARGO simulations"  << endl;
	cout << "     -nb     | --number-of-bodies             : set the number of bodies for benchmarking (pattern: n_st n_gp n_rp n_pp n_spl n_pl n_tp)" << endl;

	cout << "     -h      | --help                         : print this help" << endl;
}
