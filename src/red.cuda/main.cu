// includes system
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>

// includes CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes Thrust
#ifdef __GNUC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#else
#include "thrust\device_ptr.h"
#include "thrust\fill.h"
#include "thrust\extrema.h"
#endif

// includes project
#include "int_euler.h"
#include "int_rungekutta2.h"
#include "int_rungekutta4.h"
#include "parameter.h"
#include "redutilcu.h"
#include "nbody_exception.h"
#include "options.h"
#include "red_type.h"

using namespace std;
using namespace redutilcu;

void create_result_filename(options &opt, integrator *intgr, string &path)
{
	string adapt = (opt.param->adaptive == true ? "_a_" : "_");
	string ups = (opt.use_padded_storage == true ? "ups_" : "");
	string result_filename = "result_4" + adapt + ups + intgr->name + ".txt";
	path = file::combine_path(opt.printout_dir, result_filename);
}

void create_event_filename(options &opt, integrator *intgr, string &path)
{
	string ups = (opt.use_padded_storage == true ? "ups_" : "");
	path = file::combine_path(opt.printout_dir, ups + "event_4.txt");
}

void create_log_filename(options &opt, integrator *intgr, string &path)
{
	string ups = (opt.use_padded_storage == true ? "ups_" : "");
	path = file::combine_path(opt.printout_dir, ups + "log_4.txt");
}

void create_info_filename(options &opt, integrator *intgr, string &path)
{
	string ups = (opt.use_padded_storage == true ? "ups_" : "");
	path = file::combine_path(opt.printout_dir, ups + "info_4.txt");
}

ttt_t step(integrator *intgr, clock_t* sum_time_of_steps, clock_t* time_of_one_step)
{
	clock_t start_of_step = clock();
	ttt_t dt = intgr->step();
	clock_t end_of_step = clock();

	*time_of_one_step = (end_of_step - start_of_step);
	*sum_time_of_steps += *time_of_one_step;

	return dt;
}

void print_info(ostream& sout, const pp_disk* ppd, integrator *intgr, ttt_t dt, clock_t* sum_time_of_steps, clock_t* time_of_one_step, time_t* time_info_start)
{
	cout.setf(ios::right);
	cout.setf(ios::scientific);

	sout.setf(ios::right);
	sout.setf(ios::scientific);

	number_of_bodies* nb = ppd->n_bodies; 

	*time_info_start = clock();
	cout << tools::get_time_stamp() << " t: " << setprecision(6) << setw(12) << ppd->t 
		 << ", dt: " << setprecision(6) << setw(12)  << dt;
	cout << ", dT_cpu: " << setprecision(3) << setw(10) << *time_of_one_step / (double)CLOCKS_PER_SEC << " s";
	cout << ", dT_avg: " << setprecision(3) << setw(10) << (*sum_time_of_steps / (double)CLOCKS_PER_SEC) / intgr->get_n_passed_step() << " s";
	cout << ", Nc: " << setw(5) << ppd->n_collision[  EVENT_COUNTER_NAME_TOTAL]
	     << ", Ne: " << setw(5) << ppd->n_ejection[   EVENT_COUNTER_NAME_TOTAL]
		 << ", Nh: " << setw(5) << ppd->n_hit_centrum[EVENT_COUNTER_NAME_TOTAL]
		 << ", N : " << setw(6) << nb->get_n_total() << "(" << setw(6) << nb->get_n_total_inactive() << ")" << endl;

	sout << tools::get_time_stamp() << " t: " << setprecision(6) << setw(12) << ppd->t 
		 << ", dt: " << setprecision(6) << setw(12)  << dt;
	sout << ", dT_cpu: " << setprecision(3) << setw(10) << *time_of_one_step / (double)CLOCKS_PER_SEC << " s";
	sout << ", dT_avg: " << setprecision(3) << setw(10) << (*sum_time_of_steps / (double)CLOCKS_PER_SEC) / intgr->get_n_passed_step() << " s";
	sout << ", Nc: " << setw(5) << ppd->n_collision[  EVENT_COUNTER_NAME_TOTAL]
	     << ", Ne: " << setw(5) << ppd->n_ejection[   EVENT_COUNTER_NAME_TOTAL]
		 << ", Nh: " << setw(5) << ppd->n_hit_centrum[EVENT_COUNTER_NAME_TOTAL]
		 << ", N : " << setw(6) << nb->get_n_total() << "(" << setw(6) << nb->get_n_total_inactive() << ")"
	     << ", N_st: " << setw(5) << nb->n_s   << "(" << setw(5) << nb->n_i_s << ")"
		 << ", N_gp: " << setw(5) << nb->n_gp  << "(" << setw(5) << nb->n_i_gp << ")"
		 << ", N_rp: " << setw(5) << nb->n_rp  << "(" << setw(5) << nb->n_i_rp << ")"
		 << ", N_pp: " << setw(5) << nb->n_pp  << "(" << setw(5) << nb->n_i_pp << ")"
		 << ", N_sp: " << setw(5) << nb->n_spl << "(" << setw(5) << nb->n_i_spl << ")"
		 << ", N_pl: " << setw(5) << nb->n_pl  << "(" << setw(5) << nb->n_i_pl << ")"
		 << ", N_tp: " << setw(5) << nb->n_tp  << "(" << setw(5) << nb->n_i_tp << ")" << endl;	
}

//http://stackoverflow.com/questions/11666049/cuda-kernel-results-different-in-release-mode
//http://developer.download.nvidia.com/assets/cuda/files/NVIDIA-CUDA-Floating-Point.pdf

//--verbose -iDir C:\Work\Projects\red.cuda\TestRun\InputTest\Debug\TwoBody -p parameters.txt -ic TwoBody.txt
//--verbose -iDir C:\Work\Projects\red.cuda\TestRun\InputTest\Release\TwoBody -p parameters.txt -ic TwoBody.txt

//--verbose -iDir C:\Work\Projects\red.cuda\TestRun\DvorakDisk\Run01 -p parameters.txt -ic run01.txt
//-v -n_tpb 64 -iDir C:\Work\Projects\red.cuda.TestRun\Emese_Dvorak\cf_5 -p parameters.txt -ic suli-data-collision-N10001-heliocentric-vecelem-binary.txt
//-v C:\Work\Projects\red.cuda\TestRun\DvorakDisk\Run_cf_5 -p parameters.txt -ic run01.txt -ic Run_cf_5.txt
int main(int argc, const char** argv, const char** env)
{
	time_t start = time(NULL);

	ostream* log_f = 0x0;
	try
	{
		options opt = options(argc, argv);
		pp_disk *ppd = opt.create_pp_disk();
		integrator *intgr = opt.create_integrator(ppd, 0.001);

		if (opt.verbose)
		{
			file::log_start_cmd(cout, argc, argv, env);
			device_query(cout);
		}

		string path;
		create_result_filename(opt, intgr, path);
		ostream* result_f = new ofstream(path.c_str(), ios::out);

		create_event_filename(opt, intgr, path);
		ostream* event_f = new ofstream(path.c_str(), ios::out);

		create_log_filename(opt, intgr, path);
		log_f = new ofstream(path.c_str(), ios::out);

		create_info_filename(opt, intgr, path);
		ostream* info_f = new ofstream(path.c_str(), ios::out);

		file::log_start_cmd(*log_f, argc, argv, env);
		device_query(*log_f);

		ttt_t ps = 0;
		ttt_t dt = 0;
		clock_t sum_time_of_steps = 0.0;
		clock_t time_of_one_step  = 0.0;

		time_t time_info_start = clock();

		ppd->print_result_ascii(*result_f);
		while (ppd->t <= opt.param->stop_time)
		{
			if (fabs(ps) >= opt.param->output_interval)
			{
				ps = 0.0;
				ppd->copy_to_host();
				ppd->print_result_ascii(*result_f);
			}

			if (ppd->check_for_ejection_hit_centrum())
			{
				ppd->print_event_data(*event_f, *log_f);
				ppd->clear_event_counter();
			}

			dt = step(intgr, &sum_time_of_steps, &time_of_one_step);
			ps += fabs(dt);

			if (opt.param->thrshld[THRESHOLD_COLLISION_FACTOR] > 0.0 && ppd->check_for_collision())
			{
				ppd->print_event_data(*event_f, *log_f);
				ppd->clear_event_counter();
			}

			if (ppd->check_for_rebuild_vectors(8))
			{
				file::log_rebuild_vectors(*log_f, ppd->t);
			}

			if ((clock() - time_info_start) / (double)CLOCKS_PER_SEC > 5.0) 
			{
				print_info(*info_f, ppd, intgr, dt, &sum_time_of_steps, &time_of_one_step, &time_info_start);
			}
		} /* while */
		print_info(*info_f, ppd, intgr, dt, &sum_time_of_steps, &time_of_one_step, &time_info_start);

		// To avoid duplicate save at the end of the simulation
		if (ps > 0.0)
		{
			ppd->copy_to_host();
			ppd->print_result_ascii(*result_f);
		}

	} /* try */
	catch (const nbody_exception& ex)
	{
		if (0x0 != log_f)
		{
			file::log_message(*log_f, "Error: " + string(ex.what()));
		}
		cerr << "Error: " << ex.what() << endl;
	}
	catch (const string& msg)
	{
		if (0x0 != log_f)
		{
			file::log_message(*log_f, "Error: " + msg);
		}
		cerr << "Error: " << msg << endl;
	}
	file::log_message(*log_f, " Total time: " + tools::convert_time_t(time(NULL) - start) + " s");

	cout << " Total time: " << time(NULL) - start << " s" << endl;

	// Needed by nvprof.exe
	cudaDeviceReset();

	return (EXIT_SUCCESS);
}
