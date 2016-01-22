#include <iostream>

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
{
}

void options::create_default()
{
	continue_simulation = false;
	benchmark           = false;
	test                = false;
	verbose             = false;
	print_to_screen     = false;
	ef                  = false;

	info_dt             = 5.0;     // [sec]
	dump_dt             = 3600.0;  // [sec]

	id_dev              = 0;
	n_change_to_cpu     = 100;

	comp_dev            = COMPUTING_DEVICE_GPU;
	g_disk_model        = GAS_DISK_MODEL_NONE;

	out_fn[OUTPUT_NAME_DUMP]           = "dump";
	out_fn[OUTPUT_NAME_DUMP_AUX]       = "dump.aux";
	out_fn[OUTPUT_NAME_EVENT]          = "event";
	out_fn[OUTPUT_NAME_INFO]           = "info";
	out_fn[OUTPUT_NAME_LOG]            = "log";
	out_fn[OUTPUT_NAME_RESULT]         = "result";
	out_fn[OUTPUT_NAME_INTEGRAL]       = "integral";
	out_fn[OUTPUT_NAME_INTEGRAL_EVENT] = "integral_event";
}

void options::parse(int argc, const char** argv)
{
	int i = 1;

	while (i < argc)
	{
		string p = argv[i];

		if (     p == "--continue" || p == "-c")
		{
			continue_simulation = true;
		}
		else if (p == "--benchmark" || p == "-b")
		{
			benchmark = true;
		}
		else if (p == "--test" || p == "-t")
		{
			test = true;
		}
		else if (p == "--verbose" || p == "-v")
		{
			verbose = true;
		}
		else if (p == "--print_to_screen" || p == "-pts")
		{
			verbose = true;
			print_to_screen = true;
		}
		else if (p == "-ef")
		{
			ef = true;
		}

		else if (p == "--info-dt" || p == "-i_dt")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			info_dt = atof(argv[i]);
		}
		else if (p == "--dump-dt" || p == "-d_dt")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			dump_dt = atof(argv[i]);
		}

		else if (p == "--id_active_device" || p == "-id_dev")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			id_dev = atoi(argv[i]);
		}
		else if (p == "--n_change_to_cpu" || p == "-n_chg")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			n_change_to_cpu = atoi(argv[i]);
		}

		else if (p == "--cpu" || p == "-cpu")
		{
			comp_dev = COMPUTING_DEVICE_CPU;
		}
		else if (p == "--gpu" || p == "-gpu")
		{
			comp_dev = COMPUTING_DEVICE_GPU;
		}
		else if (p == "--analytic_gas_disk" || p == "-ga")
		{
			i++;
			in_fn[INPUT_NAME_GAS_DISK_MODEL] = argv[i];
			g_disk_model = GAS_DISK_MODEL_ANALYTIC;
		}
		else if (p == "--fargo_gas_disk" || p == "-gf")
		{
			i++;
			in_fn[INPUT_NAME_GAS_DISK_MODEL] = argv[i];
			g_disk_model = GAS_DISK_MODEL_FARGO;
		}

		else if (p == "--info-filename" || p == "-info")
		{
			i++;
			out_fn[OUTPUT_NAME_INFO] = argv[i];
		}
		else if (p == "--event-filename" || p == "-event")
		{
			i++;
			out_fn[OUTPUT_NAME_EVENT] = argv[i];
		}
		else if (p == "--log-filename" || p == "-log")
		{
			i++;
			out_fn[OUTPUT_NAME_LOG] = argv[i];
		}
		else if (p == "--result-filename" || p == "-result")
		{
			i++;
			out_fn[OUTPUT_NAME_RESULT] = argv[i];
		}



		else if (p == "--initial_conditions" || p == "-ic")
		{
			i++;
			in_fn[INPUT_NAME_BODYLIST] = argv[i];
		}

		else if (p =="--parameters" || p == "-p")
		{
			i++;
			in_fn[INPUT_NAME_PARAMETER] = argv[i];
		}

		else if (p == "--inputDir" || p == "-iDir")
		{
			i++;
			dir[DIRECTORY_NAME_IN] = argv[i];
		}
		else if (p == "--outputDir" || p == "-oDir")
		{
			i++;
			dir[DIRECTORY_NAME_OUT] = argv[i];
		}

		else if (p == "--number-of-bodies"   || p == "-nb")
		{
			int n_st  = 0;
			int n_gp  = 0;
			int n_rp  = 0;
			int n_pp  = 0;
			int n_spl = 0;
			int n_pl  = 0;
			int n_tp  = 0;
			i++;    n_st  = atoi(argv[i]);
			i++;    n_gp  = atoi(argv[i]);
			i++;    n_rp  = atoi(argv[i]);
			i++;    n_pp  = atoi(argv[i]);
			i++;    n_spl = atoi(argv[i]);
			i++;    n_pl  = atoi(argv[i]);
			i++;    n_tp  = atoi(argv[i]);
			n_bodies = new number_of_bodies(n_st, n_gp, n_rp, n_pp, n_spl, n_pl, n_tp);
		}
		else if (p == "--help" || p == "-h")
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

	if (benchmark)
	{
		ppd = new pp_disk(n_bodies, g_disk_model, param->cdm, id_dev, comp_dev);
	}
	else
	{
		string path = file::combine_path(dir[DIRECTORY_NAME_IN], in_fn[INPUT_NAME_BODYLIST]);
		ppd = new pp_disk(path, continue_simulation, g_disk_model, param->cdm, id_dev, comp_dev, param->threshold, print_to_screen);
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
		ppd->f_gd = new fargo_gas_disk(in_fn[INPUT_NAME_BODYLIST], in_fn[INPUT_NAME_GAS_DISK_MODEL], comp_dev, print_to_screen);
		break;
	default:
		throw string("Parameter 'g_disk_model' is out of range");
	}

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		ppd->copy_to_device();
	}
	ppd->t = (continue_simulation ? ppd->sim_data->h_epoch[0] : param->start_time);

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
	//case INTEGRATOR_RUNGEKUTTANYSTROM:
	//	throw string("Requested integrator is not implemented.");
	//	//intgr = new rungekuttanystrom<9>(*f, dt, adaptive, tolerance, comp_dev);
	//	break;
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
	cout << "Usage: red.cuda <parameterlist>" << endl;
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
	cout << "     -oDir   | --outputDir <directory>        : the directory where the output files will be stored (if omitted the input directory will be used)" << endl;
	cout << "     -p      | --parameter <filename>         : the file containing the parameters of the simulation"  << endl;
	cout << "     -ga     | --analytic_gas_disk <filename> : the file containing the parameters of an analyticaly prescribed gas disk"  << endl;
	cout << "     -gf     | --fargo_gas_disk <filename>    : the file containing the details of the gas disk resulted from FARGO simulations"  << endl;
	cout << "     -nb     | --number-of-bodies             : set the number of bodies for benchmarking (pattern: n_st n_gp n_rp n_pp n_spl n_pl n_tp)" << endl;

	cout << "     -h      | --help                         : print this help" << endl;
}
