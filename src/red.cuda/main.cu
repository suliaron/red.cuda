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

// includes project
#include "int_euler.h"
#include "int_rungekutta2.h"
#include "int_rungekutta4.h"
#include "parameter.h"
#include "redutilcu.h"
#include "nbody_exception.h"
#include "options.h"
#include "red_type.h"
#include "red_constants.h"

using namespace std;
using namespace redutilcu;

void open_streams(const options& opt, const integrator* intgr, ostream** result_f, ostream** info_f, ostream** event_f, ostream** log_f)
{
	string path;
	string prefix;
	string ext = "txt";

	if (opt.ef)
	{
		char sep = '_';
		string config;
#ifdef _DEBUG
		config = "D";
#else
		config = "R";
#endif
		string dev = (opt.comp_dev == COMPUTING_DEVICE_CPU ? "cpu" : "gpu");
		// as: adaptive step-size, fs: fix step-size
		string adapt = opt.param->adaptive == true ? "as" : "fs";
		// ps: padded storage, ns: normal storage
		string strorage = opt.use_padded_storage == true ? "ps" : "ns";
		string int_name = intgr->short_name;
		prefix = config + sep + dev + sep + strorage + sep + adapt + sep + int_name + sep;
	}

	path = file::combine_path(opt.printout_dir, prefix + opt.result_filename) + "." + ext;
	*result_f = new ofstream(path.c_str(), ios::out);

	path = file::combine_path(opt.printout_dir, prefix + opt.info_filename) + "." + ext;
	*info_f = new ofstream(path.c_str(), ios::out);

	path = file::combine_path(opt.printout_dir, prefix + opt.event_filename) + "." + ext;
	*event_f = new ofstream(path.c_str(), ios::out);

	path = file::combine_path(opt.printout_dir, prefix + opt.log_filename) + "." + ext;
	*log_f = new ofstream(path.c_str(), ios::out);
}

void print_info(ostream& sout, const pp_disk* ppd, integrator *intgr, ttt_t dt, clock_t* sum_time_of_steps, clock_t* time_of_one_step, time_t* time_info_start)
{
	cout.setf(ios::right);
	cout.setf(ios::scientific);

	sout.setf(ios::right);
	sout.setf(ios::scientific);

	number_of_bodies* nb = ppd->n_bodies; 

	*time_info_start = clock();
	cout << tools::get_time_stamp() 
		 << " t: " << setprecision(6) << setw(12) << ppd->t / constants::Gauss
		 << ", dt: " << setprecision(6) << setw(12)  << dt / constants::Gauss;
	cout << ", dT_cpu: " << setprecision(3) << setw(10) << *time_of_one_step / (double)CLOCKS_PER_SEC << " s";
	cout << ", dT_avg: " << setprecision(3) << setw(10) << (*sum_time_of_steps / (double)CLOCKS_PER_SEC) / intgr->get_n_passed_step() << " s";
	cout << ", Nc: " << setw(5) << ppd->n_collision[  EVENT_COUNTER_NAME_TOTAL]
	     << ", Ne: " << setw(5) << ppd->n_ejection[   EVENT_COUNTER_NAME_TOTAL]
		 << ", Nh: " << setw(5) << ppd->n_hit_centrum[EVENT_COUNTER_NAME_TOTAL]
		 << ", N : " << setw(6) << nb->get_n_total() << "(" << setw(6) << nb->get_n_total_inactive() << ")" << endl;

	sout << tools::get_time_stamp()
		 << " t: " << setprecision(6) << setw(12) << ppd->t  / constants::Gauss
		 << ", dt: " << setprecision(6) << setw(12)  << dt / constants::Gauss;
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

ttt_t step(integrator *intgr, clock_t* sum_time_of_steps, clock_t* t_step)
{
	clock_t t_start = clock();
	ttt_t dt = intgr->step();
	clock_t t_stop = clock();

	*t_step = (t_stop - t_start);
	*sum_time_of_steps += *t_step;

	return dt;
}

//http://stackoverflow.com/questions/11666049/cuda-kernel-results-different-in-release-mode
//http://developer.download.nvidia.com/assets/cuda/files/NVIDIA-CUDA-Floating-Point.pdf

int main(int argc, const char** argv, const char** env)
{
	time_t start = time(NULL);

	ostream* result_f = 0x0;
	ostream* info_f   = 0x0;
	ostream* event_f  = 0x0;
	ostream* log_f    = 0x0;
	try
	{
		options opt = options(argc, argv);
		pp_disk *ppd = opt.create_pp_disk();
		integrator *intgr = opt.create_integrator(ppd, 0.001);
		open_streams(opt, intgr, &result_f, &info_f, &event_f, &log_f);

		file::log_start_cmd(*log_f, argc, argv, env);
		if (opt.verbose)
		{
			file::log_start_cmd(cout, argc, argv, env);
			if (COMPUTING_DEVICE_GPU == opt.comp_dev)
			{
				device_query(cout, opt.id_a_dev);
			}
		}
		if (COMPUTING_DEVICE_GPU == opt.comp_dev)
		{
			device_query(*log_f, opt.id_a_dev);
		}

		ttt_t ps = 0.0;
		ttt_t dt = 0.0;
		clock_t sum_time_of_steps = 0;
		clock_t time_of_one_step  = 0;

		time_t time_info_start = clock();

		ppd->print_result_ascii(*result_f);

		int dummy_k = 0;
		while (ppd->t <= opt.param->stop_time)
		{
			//if (10 == dummy_k)
			//{
			//	redutilcu::set_device(0, opt.verbose);
			//	intgr->set_computing_device(COMPUTING_DEVICE_GPU);
			//}

			if (opt.param->output_interval <= fabs(ps))
			{
				ps = 0.0;
				if (COMPUTING_DEVICE_GPU == opt.comp_dev)
				{
					ppd->copy_to_host();
				}
				ppd->print_result_ascii(*result_f);
			}

			if (ppd->check_for_ejection_hit_centrum())
			{
				ppd->print_event_data(*event_f, *log_f);
				ppd->clear_event_counter();
			}

			dt = step(intgr, &sum_time_of_steps, &time_of_one_step);
			ps += fabs(dt);

			if (0.0 < opt.param->thrshld[THRESHOLD_RADII_ENHANCE_FACTOR] && ppd->check_for_collision())
			{
				ppd->print_event_data(*event_f, *log_f);
				ppd->clear_event_counter();
			}

			if (ppd->check_for_rebuild_vectors(8))
			{
				file::log_rebuild_vectors(*log_f, ppd->t);
			}

			if (5.0 < (clock() - time_info_start) / (double)CLOCKS_PER_SEC) 
			{
				print_info(*info_f, ppd, intgr, dt, &sum_time_of_steps, &time_of_one_step, &time_info_start);
			}

			dummy_k++;
		} /* while */
		print_info(*info_f, ppd, intgr, dt, &sum_time_of_steps, &time_of_one_step, &time_info_start);

		// To avoid duplicate save at the end of the simulation
		if (0.0 < ps)
		{
			if (COMPUTING_DEVICE_GPU == opt.comp_dev)
			{
				ppd->copy_to_host();
			}
			ppd->print_result_ascii(*result_f);
		}
		// Needed by nvprof.exe
		if (COMPUTING_DEVICE_GPU == opt.comp_dev)
		{
			cudaDeviceReset();
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
	if (0x0 != log_f)
	{
		file::log_message(*log_f, "Total time: " + tools::convert_time_t(time(NULL) - start) + " s");
	}
	cout << "Total time: " << time(NULL) - start << " s" << endl;

	return (EXIT_SUCCESS);
}
