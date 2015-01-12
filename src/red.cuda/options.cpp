// includes system
#include <iostream>

// includes project
#include "redutilcu.h"
#include "number_of_bodies.h"
#include "options.h"
#include "pp_disk.h"
#include "int_euler.h"
#include "int_rungekutta2.h"
#include "int_rungekutta4.h"
#include "int_rungekutta5.h"
#include "int_rungekutta8.h"
#include "util.h"
#include "red_constants.h"
#include "red_macro.h"

using namespace redutilcu;

options::options(int argc, const char** argv) :
	has_gas(false),
	verbose(false),
	use_padded_storage(false),
	n_tpb(64),
	param(0),
	g_disk(0)
{
	create_default_options();
	parse_options(argc, argv);

	cudaError_t cudaStatus = cudaSuccess;

	int n_device = 0;
	cudaGetDeviceCount(&n_device);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus)
	{
		throw string("cudaGetDeviceCount() failed");
	}
	if (verbose)
	{
		printf("The number of CUDA device(s) : %2d\n", n_device);
	}
	if (1 > n_device)
	{
		printf("No CUDA device was found. The code will execute on the CPU. Implementation is missing.\n");
		// TODO : implement the CPU code
		exit(0);
	}

	// Set the desired id of the device
	if (n_device > id_a_dev)
	{
		id_active_device = id_a_dev;
        cudaSetDevice(id_active_device);
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus)
		{
			throw string("cudaSetDevice() failed");
		}
	}

	if (parameters_filename.length() == 0)
	{
		throw string("Missing filename for -p | --parameter!");
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
	cout << "Usage: red.cuda <parameterlist>" << endl;
	cout << "Parameters:" << endl;
	cout << "     -ups    | --use-padded-storage           : use padded storage to store data (default is false)" << endl; 
	cout << "     -n_tpb  | --n_thread-per-block           : the number of thread per block to use in kernel lunches (default is 64)" << endl;
	cout << "     -iDir   | --inputDir <directory>         : the directory containing the input files"  << endl;
	cout << "     -p      | --parameter <filename>         : the file containing the parameters of the simulation"  << endl;
	cout << "     -gd     | --gas_disk <filename>          : the file containing the parameters of the gas disk"  << endl;
	cout << "     -ic     | --initial_condition <filename> : the file containing the initial conditions"  << endl;
	cout << "     -id_dev | --id_active_device             : the id of the device which will execute the code" << endl;
	cout << "     -v      | --verbose                      : verbose mode" << endl;
	cout << "     -h      | --help                         : print this help" << endl;
}

// TODO: implement
void options::create_default_options()
{
	use_padded_storage = false;
	n_tpb = 64;
}

void options::parse_options(int argc, const char** argv)
{
	int i = 1;

	while (i < argc) {
		string p = argv[i];

		if      (p == "--use-padded-storage" || p == "-ups")
		{
			use_padded_storage = true;
		}
		else if (p == "--id_active_device" || p == "-id_dev")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			id_a_dev = atoi(argv[i]);
		}
		else if (p == "--n_thread-per-block" || p == "-n_tpb")
		{
			i++;
			if (!tools::is_number(argv[i])) 
			{
				throw string("Invalid number at: " + p);
			}
			n_tpb = atoi(argv[i]);
		}
		else if (p == "--inputDir" || p == "-iDir")	{
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
			throw string("Invalid switch on command-line: " + p + ".");
		}
		i++;
	}
}

pp_disk* options::create_pp_disk()
{
	string path = file::combine_path(input_dir, bodylist_filename);
	pp_disk* ppd = new pp_disk(path, g_disk, n_tpb, use_padded_storage);
	if (ppd->g_disk != 0)
	{
		ppd->g_disk->calc(ppd->get_mass_of_star());
		//ppd->print_result_ascii(cout);
	}
	if (param->fr_cntr == FRAME_CENTER_BARY)
	{
		ppd->transform_to_bc();
		//cout << "Body data after transformation:" << endl;
		//ppd->print_result_ascii(cout);
	}
	ppd->transform_time();
	ppd->copy_to_device();
	ppd->copy_threshold_to_device(param->thrshld);

	//ppd->test_call_kernel_print_sim_data();

	ppd->t = param->start_time;

	return ppd;
}

integrator* options::create_integrator(pp_disk* ppd, ttt_t dt)
{
	integrator* intgr;

	switch (param->int_type)
	{
	case INTEGRATOR_EULER:
		intgr = new euler(ppd, dt);
		break;
	case INTEGRATOR_RUNGEKUTTA2:
		intgr = new rungekutta2(ppd, dt);
		break;
	case INTEGRATOR_RUNGEKUTTA4:
		intgr = new rungekutta4(ppd, dt, param->adaptive, param->tolerance);
		break;
	case INTEGRATOR_RUNGEKUTTA5:
		intgr = new rungekutta5(ppd, dt, param->adaptive, param->tolerance);
		break;
	case INTEGRATOR_RUNGEKUTTAFEHLBERG78:
		intgr = new rungekutta8(ppd, dt, param->adaptive, param->tolerance);
		//intgr = new c_rungekutta8(ppd, dt, param->adaptive, param->tolerance);
		break;
	case INTEGRATOR_RUNGEKUTTANYSTROM:
		throw string("Requested integrator is not implemented.");
		//intgr = new rungekuttanystrom<9>(*f, dt, adaptive, tolerance);
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
