// includes system
#include <iostream>

// includes project
#include "redutilcu.h"
#include "options.h"
#include "gas_disk.h"
#include "int_euler.h"
#include "int_rungekutta2.h"
#include "int_rungekutta4.h"
#include "int_rungekutta8.h"
#include "util.h"
#include "red_constants.h"
#include "red_macro.h"

using namespace redutilcu;

options::options(int argc, const char** argv)
{
	initialize();
	create_default();
	parse(argc, argv);

	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		set_device(id_a_dev, verbose);
	}
	if (COMPUTING_DEVICE_CPU == comp_dev)
	{
		ups = false;
	}

	param = new parameter(input_dir, parameters_filename, verbose);

	switch (g_disk_model)
	{
	case GAS_DISK_MODEL_NONE:
		break;
	case GAS_DISK_MODEL_ANALYTIC:
		a_gd = new analytic_gas_disk(input_dir, gasdisk_filename, verbose);
		break;
	case GAS_DISK_MODEL_FARGO:
		f_gd = new fargo_gas_disk(input_dir, gasdisk_filename, comp_dev, verbose);
		break;
	default:
		throw string("Parameter 'g_disk_model' is out of range");
	}
}

options::~options() 
{
}

void options::print_usage()
{
	cout << "Usage: red.cuda <parameterlist>" << endl;
	cout << "Parameters:" << endl;
	cout << "     -id_dev | --id_active_device             : the id of the device which will execute the code" << endl;
	cout << "     -ups    | --use-padded-storage           : use padded storage to store data (default is false)" << endl; 
	cout << "     -n_tpb  | --n_thread-per-block           : the number of thread per block to use in kernel lunches (default is 64)" << endl;
	cout << "     -n_chg  | --n_change_to_cpu              : the threshold value for the total number of SI bodies to change to the CPU (default is 100)" << endl;
	cout << "     -iDir   | --inputDir <directory>         : the directory containing the input files"  << endl;
	cout << "     -p      | --parameter <filename>         : the file containing the parameters of the simulation"  << endl;
	cout << "     -ga     | --analytic_gas_disk <filename> : the file containing the parameters of an analyticaly prescribed gas disk"  << endl;
	cout << "     -gf     | --fargo_gas_disk <filename>    : the file containing the details of the gas disk resulted from FARGO simulations"  << endl;
	cout << "     -ic     | --initial_condition <filename> : the file containing the initial conditions"  << endl;
	cout << "     -info   | --info-filename                : the name of the file where the runtime output of the code will be stored (default is info.txt)" << endl;
	cout << "     -event  | --event-filename               : the name of the file where the details of each event will be stored (default is event.txt)" << endl;
	cout << "     -log    | --log-filename                 : the name of the file where the details of the execution of the code will be stored (default is log.txt)" << endl;
	cout << "     -result | --result-filename              : the name of the file where the simlation data for a time instance will be stored (default is result.txt" << endl;
	cout << "                                                where [...] contains data describing the integrator)" << endl;
	cout << "     -b      | --benchmark                    : run benchmark to find out the optimal number of threads per block" << endl;
	cout << "     -t      | --test                         : run tests" << endl;
	cout << "     -nb     | --number-of-bodies             : set the number of bodies for benchmarking (pattern: n_st n_gp n_rp n_pp n_spl n_pl n_tp)" << endl;
	cout << "     -v      | --verbose                      : verbose mode" << endl;
	cout << "     -gpu    | --gpu                          : Execute the code on the graphics processing unit (GPU) (default is true)" << endl;
	cout << "     -cpu    | --cpu                          : Execute the code on the cpu if required by the user or if no GPU is installed (default is false)" << endl;
	cout << "     -h      | --help                         : print this help" << endl;
}

void options::initialize()
{
	n_bodies = 0x0;
	param    = 0x0;
	a_gd     = 0x0;
	f_gd     = 0x0;
}

void options::create_default()
{
	benchmark       = false;
	test            = false;
	ef              = false;
	verbose         = false;
	ups             = false;

	id_a_dev        = 0;
	n_tpb0          = 64;
	n_change_to_cpu = 100;

	comp_dev        = COMPUTING_DEVICE_GPU;
	g_disk_model    = GAS_DISK_MODEL_NONE;

	info_filename   = "info";
	event_filename  = "event";
	log_filename    = "log";
	result_filename = "result";
}

void options::parse(int argc, const char** argv)
{
	int n_st  = 0;
	int n_gp  = 0;
	int n_rp  = 0;
	int n_pp  = 0;
	int n_spl = 0;
	int n_pl  = 0;
	int n_tp  = 0;

	int i = 1;

	while (i < argc)
	{
		string p = argv[i];

		if (     p == "--id_active_device" || p == "-id_dev")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			id_a_dev = atoi(argv[i]);
		}
		else if (p == "--use-padded-storage" || p == "-ups")
		{
			ups = true;
		}
		else if (p == "--n_thread-per-block" || p == "-n_tpb")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			n_tpb0 = atoi(argv[i]);
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
		else if (p == "--inputDir" || p == "-iDir")
		{
			i++;
			input_dir = argv[i];
			printout_dir = input_dir;
		}
		else if (p =="--parameters" || p == "-p")
		{
			i++;
			parameters_filename = argv[i];
		}
		else if (p == "--analytic_gas_disk" || p == "-ga")
		{
			i++;
			gasdisk_filename = argv[i];
			g_disk_model = GAS_DISK_MODEL_ANALYTIC;
		}
		else if (p == "--fargo_gas_disk" || p == "-gf")
		{
			i++;
			gasdisk_filename = argv[i];
			g_disk_model = GAS_DISK_MODEL_FARGO;
		}
		else if (p == "--initial_conditions" || p == "-ic")
		{
			i++;
			bodylist_filename = argv[i];
		}
		else if (p == "--info-filename" || p == "-info")
		{
			i++;
			info_filename = argv[i];
		}
		else if (p == "--event-filename" || p == "-event")
		{
			i++;
			event_filename = argv[i];
		}
		else if (p == "--log-filename" || p == "-log")
		{
			i++;
			log_filename = argv[i];
		}
		else if (p == "--result-filename" || p == "-result")
		{
			i++;
			result_filename = argv[i];
		}
		else if (p == "--verbose" || p == "-v")
		{
			verbose = true;
		}
		else if (p == "--benchmark" || p == "-b")
		{
			benchmark = true;
		}
		else if (p == "--test" || p == "-t")
		{
			test = true;
		}
		else if (p == "-ef")
		{
			ef = true;
		}
		else if (p == "--cpu" || p == "-cpu")
		{
			comp_dev = COMPUTING_DEVICE_CPU;
		}
		else if (p == "--gpu" || p == "-gpu")
		{
			comp_dev = COMPUTING_DEVICE_GPU;
		}
		else if (p == "--number-of-bodies"   || p == "-nb")
		{
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
}

pp_disk* options::create_pp_disk()
{
	pp_disk* ppd = 0x0;

	if (benchmark)
	{
		ppd = new pp_disk(n_bodies, n_tpb0, ups, g_disk_model, comp_dev);
	}
	else
	{
		string path = file::combine_path(input_dir, bodylist_filename);
		ppd = new pp_disk(path, n_tpb0, ups, g_disk_model, comp_dev);
	}
	switch (g_disk_model)
	{
	case GAS_DISK_MODEL_NONE:
		break;
	case GAS_DISK_MODEL_ANALYTIC:
		ppd->a_gd = a_gd;
		ppd->a_gd->calc(ppd->get_mass_of_star());
		//ppd->print_result_ascii(cout);
		break;
	case GAS_DISK_MODEL_FARGO:
		ppd->f_gd = f_gd;
		break;
	}

	ppd->transform_to_bc(verbose);
	ppd->transform_time(verbose);
	ppd->transform_velocity(verbose);
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		ppd->copy_to_device();
		ppd->copy_disk_params_to_device();
	}
	ppd->copy_threshold(param->thrshld);
	ppd->t = param->start_time;

	return ppd;
}

integrator* options::create_integrator(pp_disk* ppd, ttt_t dt)
{
	integrator* intgr;

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
	case INTEGRATOR_RUNGEKUTTANYSTROM:
		throw string("Requested integrator is not implemented.");
		//intgr = new rungekuttanystrom<9>(*f, dt, adaptive, tolerance, comp_dev);
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
