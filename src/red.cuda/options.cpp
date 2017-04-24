#include <iostream>
#include <fstream>

#include "options.h"
#include "analytic_gas_disk.h"
#include "fargo_gas_disk.h"
#include "int_euler.h"
#include "int_rungekutta2.h"
#include "int_rungekutta4.h"
#include "int_rungekutta8.h"

#include "redutilcu.h"
#include "red_constants.h"
#include "red_macro.h"

#ifdef __GNUC__
#include <stdlib.h>
#endif

using namespace std;
using namespace redutilcu;

options::options(int argc, const char** argv) :
	n_bodies(0x0),
	param(0x0)
{
	create_default();
	parse(argc, argv);

	if (!test)
	{
		param = new parameter(dir[DIRECTORY_NAME_IN], in_fn[INPUT_NAME_PARAMETER], print_to_screen);
	}
}

options::~options() 
{ }

void options::create_default()
{
	test                = false;
	verbose             = false;
	print_to_screen     = false;
    print_dbe           = false;
	ef                  = false;

	id_dev              = 0;
	n_change_to_cpu     = 100;

	comp_dev            = COMPUTING_DEVICE_CPU;
	g_disk_model        = GAS_DISK_MODEL_NONE;

	out_fn[OUTPUT_NAME_LOG]            = "log";
	out_fn[OUTPUT_NAME_INFO]           = "info";
	out_fn[OUTPUT_NAME_EVENT]          = "event";
	out_fn[OUTPUT_NAME_DATA]           = "data";
	out_fn[OUTPUT_NAME_DATA_INFO]      = "data.info";
	out_fn[OUTPUT_NAME_INTEGRAL]       = "integral";
	out_fn[OUTPUT_NAME_INTEGRAL_EVENT] = "integral.event";
}

void options::parse(int argc, const char** argv)
{
	int i = 1;

	while (i < argc)
	{
		string p = argv[i];

		if (p == "-t")
		{
			test = true;
		}
		else if (p == "-v")
		{
			verbose = true;
		}
		else if (p == "-pts")
		{
			verbose = true;
			print_to_screen = true;
		}
		else if (p == "-pe")
		{
            print_dbe = true;
		}
		else if (p == "-ef")
		{
			ef = true;
		}

		else if (p == "-id_dev")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			id_dev = atoi(argv[i]);
		}
		else if (p == "-n_chg")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			n_change_to_cpu = atoi(argv[i]);
		}

		else if (p == "-cpu")
		{
			comp_dev = COMPUTING_DEVICE_CPU;
		}
		else if (p == "-gpu")
		{
			comp_dev = COMPUTING_DEVICE_GPU;
		}


		else if (p == "-idir")
		{
			i++;
			dir[DIRECTORY_NAME_IN] = argv[i];
		}
		else if (p == "-odir")
		{
			i++;
			dir[DIRECTORY_NAME_OUT] = argv[i];
		}


		else if (p == "-i")
		{
			i++;
			in_fn[INPUT_NAME_START_FILES] = argv[i];
		}
		else if (p == "-p")
		{
			i++;
			in_fn[INPUT_NAME_PARAMETER] = argv[i];
		}
		else if (p == "-ga")
		{
			g_disk_model = GAS_DISK_MODEL_ANALYTIC;
			i++;
			in_fn[INPUT_NAME_GAS_DISK_MODEL] = argv[i];
		}
		else if (p == "-gf")
		{
			g_disk_model = GAS_DISK_MODEL_FARGO;
			i++;
			in_fn[INPUT_NAME_GAS_DISK_MODEL] = argv[i];
		}


		else if (p == "-h")
		{
			print_usage();
			exit(EXIT_SUCCESS);
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

pp_disk* options::create_pp_disk()
{
	pp_disk* ppd = 0x0;

	if (0) //(benchmark)
	{
		ppd = new pp_disk(n_bodies, g_disk_model, id_dev, comp_dev);
	}
	else
	{
		string path_data;
		string path_data_info;

		if (0 < in_fn[INPUT_NAME_START_FILES].length())
		{
			string path = file::combine_path(dir[DIRECTORY_NAME_IN], in_fn[INPUT_NAME_START_FILES]);
			std::ifstream file(path.c_str(), ifstream::in);
			if (file)
			{
				uint32_t n = 0;
				string str;
				while (getline(file, str))
				{
					if (0 == n)
					{
						in_fn[INPUT_NAME_DATA_INFO] = str;
					}
					if (1 == n)
					{
						in_fn[INPUT_NAME_DATA] = str;
					}
					n++;
				} 	
				file.close();
			}
			else
			{
				throw string("The file '" + path + "' could not opened.");
			}
		}
		else
		{
			throw string("The -i <file> option was not provided.");
		}
		path_data      = file::combine_path(dir[DIRECTORY_NAME_IN], in_fn[INPUT_NAME_DATA]);
		path_data_info = file::combine_path(dir[DIRECTORY_NAME_IN], in_fn[INPUT_NAME_DATA_INFO]);
		ppd = new pp_disk(path_data, path_data_info, g_disk_model, id_dev, comp_dev, param->threshold);
	}

	switch (g_disk_model)
	{
	case GAS_DISK_MODEL_NONE:
		break;
	case GAS_DISK_MODEL_ANALYTIC:
		ppd->a_gd = new analytic_gas_disk(dir[DIRECTORY_NAME_IN], in_fn[INPUT_NAME_GAS_DISK_MODEL], print_to_screen);
		ppd->a_gd->calc(ppd->get_mass_of_star());
		break;
	case GAS_DISK_MODEL_FARGO:
		ppd->f_gd = new fargo_gas_disk(in_fn[INPUT_NAME_DATA], in_fn[INPUT_NAME_GAS_DISK_MODEL], comp_dev, print_to_screen);
		break;
	default:
		throw string("Parameter 'g_disk_model' is out of range");
	}

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		ppd->copy_to_device();
	}

    // If backward integration than and the stepsize is positive than make it negative
    if (0.0 > param->simulation_length && 0.0 < ppd->dt)
    {
        ppd->dt *= -1.0;
    }

	return ppd;
}

integrator* options::create_integrator(pp_disk* ppd, ttt_t dt)
{
	integrator* intgr = 0x0;

	switch (param->int_type)
	{
	case INTEGRATOR_EULER:
		intgr = new euler(ppd, dt, comp_dev);
		break;
	case INTEGRATOR_RUNGEKUTTA2:
		intgr = new rungekutta2(ppd, dt, comp_dev);
		break;
	case INTEGRATOR_RUNGEKUTTA4:
		intgr = new rungekutta4(ppd, dt, param->adaptive, param->tolerance, comp_dev);
		break;
	case INTEGRATOR_RUNGEKUTTAFEHLBERG78:
		intgr = new rungekutta8(ppd, dt, param->adaptive, param->tolerance, comp_dev);
		//intgr = new c_rungekutta8(ppd, dt, param->adaptive, param->tolerance);
		break;
	default:
		throw string("Requested integrator is not implemented.");
	}

	if (param->error_check_for_tp)
	{
		intgr->error_check_for_tp = true;
	}

	return intgr;
}

void options::print_usage()
{
	cout << "Usage: red [options] -i <file> -p <file>" << endl;
	cout << "Options:" << endl;
	cout << "  -t                Run self-tests" << endl;
	cout << "  -v                Verbose mode (log all event during the execution fo the code to the log file)" << endl;
	cout << "  -pe               Print the data before the event is detetced (the state before the event-detetcion)" << endl;
	cout << "  -pts              Verbose mode and print everything to the stdout too" << endl << endl;
					      
	cout << "  -id_dev <number>  The id of the device which will execute the code (default value is 0)" << endl;
	cout << "  -n_chg  <number>  The threshold value for the total number of SI bodies to change to the CPU (default value is 100)" << endl << endl;
					      
	cout << "  -gpu              Execute the code on the graphics processing unit (GPU)" << endl;
	cout << "  -cpu              Execute the code on the CPU if required by the user or if no GPU is installed (default)" << endl << endl;
					      
	cout << "  -idir <directory> The directory containing the input files"  << endl;
	cout << "  -odir <directory> The directory where the output files will be stored (if omitted the input directory will be used)" << endl << endl;
					      
	cout << "  -i <file>         The input file containing the filenames of the input data" << endl;
	cout << "  -p <file>         The input file containing the parameters of the simulation"  << endl;
	cout << "  -ga <file>        The input file containing the parameters of an analyticaly prescribed gas disk"  << endl;
	cout << "  -gf <file>        The input file containing the details of the gas disk resulted from FARGO simulations"  << endl << endl;
					      
	cout << "  -h                Display this information" << endl << endl;
	cout << "To start a simulation the -i and -p are obligatory command line arguments." << endl;
	cout << "Arguments inside the <> brackets are obligatory." << endl;
	cout << "All other options are facultative." << endl << endl;
}
