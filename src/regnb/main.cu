#define CODE_START      "2015.12.20"
#define CODE_AUTOHOR    "Süli Áron"
#define CODE_NAME       "regnb"
#define CODE_VERSION    "0.1"

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
#include "integrator.h"
#include "int_rungekutta8.h"
#include "nbody.h"
#include "options.h"

#include "red_type.h"
#include "red_constants.h"
#include "redutilcu.h"

using namespace std;
using namespace redutilcu;

string create_prefix(const options& opt)
{
	static const char* integrator_type_short_name[] = 
	{
				"E",
				"RK2",
				"RK4",
				"RK5",
				"RKF8",
				"RKN"
	};

	string prefix;

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
		string adapt = (opt.param->adaptive == true ? "as" : "fs");
		// collision detection model
		string cdm;
		switch (opt.param->cdm)
		{
		case COLLISION_DETECTION_MODEL_STEP:
			// bs: between step
			cdm = "bs";
			break;
		case COLLISION_DETECTION_MODEL_SUB_STEP:
			// bs: sub-step
			cdm = "ss";
			break;
		case COLLISION_DETECTION_MODEL_INTERPOLATION:
			throw string("COLLISION_DETECTION_MODEL_INTERPOLATION is not implemented.");
		default:
			throw string("Parameter 'cdm' is out of range.");
		}

		string int_name(integrator_type_short_name[opt.param->int_type]);
		prefix += config + sep + dev + sep + cdm + sep + adapt + sep + int_name + sep;
	}

	return prefix;
}

void open_streams(const options& opt, ofstream** output)
{
	string path;
	string prefix = create_prefix(opt);
	string ext = "txt";

	path = file::combine_path(opt.dir[DIRECTORY_NAME_OUT], prefix + opt.out_fn[OUTPUT_NAME_RESULT]) + "." + ext;
	output[OUTPUT_NAME_RESULT] = new ofstream(path.c_str(), ios::out);
	if (!*output[OUTPUT_NAME_RESULT]) 
	{
		throw string("Cannot open " + path + ".");
	}

	path = file::combine_path(opt.dir[DIRECTORY_NAME_OUT], prefix + opt.out_fn[OUTPUT_NAME_INFO]) + "." + ext;
	output[OUTPUT_NAME_INFO] = new ofstream(path.c_str(), ios::out);
	if (!*output[OUTPUT_NAME_INFO]) 
	{
		throw string("Cannot open " + path + ".");
	}

	path = file::combine_path(opt.dir[DIRECTORY_NAME_OUT], prefix + opt.out_fn[OUTPUT_NAME_EVENT]) + "." + ext;
	output[OUTPUT_NAME_EVENT] = new ofstream(path.c_str(), ios::out);
	if (!*output[OUTPUT_NAME_EVENT]) 
	{
		throw string("Cannot open " + path + ".");
	}

	path = file::combine_path(opt.dir[DIRECTORY_NAME_OUT], prefix + opt.out_fn[OUTPUT_NAME_LOG]) + "." + ext;
	output[OUTPUT_NAME_LOG] = new ofstream(path.c_str(), ios::out);
	if (!*output[OUTPUT_NAME_LOG]) 
	{
		throw string("Cannot open " + path + ".");
	}

	path = file::combine_path(opt.dir[DIRECTORY_NAME_OUT], prefix + opt.out_fn[OUTPUT_NAME_INTEGRAL]) + "." + ext;
	output[OUTPUT_NAME_INTEGRAL] = new ofstream(path.c_str(), ios::out);
	if (!*output[OUTPUT_NAME_INTEGRAL]) 
	{
		throw string("Cannot open " + path + ".");
	}

	path = file::combine_path(opt.dir[DIRECTORY_NAME_OUT], prefix + opt.out_fn[OUTPUT_NAME_INTEGRAL_EVENT]) + "." + ext;
	output[OUTPUT_NAME_INTEGRAL_EVENT] = new ofstream(path.c_str(), ios::out);
	if (!*output[OUTPUT_NAME_INTEGRAL_EVENT]) 
	{
		throw string("Cannot open " + path + ".");
	}
}

void print_info(ofstream& sout, const nbody* nbd, integrator *intgr, ttt_t dt, clock_t* T_CPU, clock_t* dT_CPU)
{
	static const string header_str = "dev,date      ,time    ,t [d]      ,dt [d]     ,dt_avg [d] ,dT [s]     ,dT_avg [s] ,Nc   ,Ne   ,Nh   ,nb_p ,nb_i,nb_r ,ns_t    ,ns_p    ,ns_f    ,ns_a,ns_r  ,ngp_a,ngp_r,nrp_a,nrp_r,npp_a,npp_r,nspl_a,nspl_r,npl_a,npl_r,ntp_a,ntp_r";
	static bool first_call = true;
	static string cvs = ",";
	
	cout.setf(ios::right);
	cout.setf(ios::scientific);

	sout.setf(ios::right);
	sout.setf(ios::scientific);

	number_of_bodies* nb = nbd->n_bodies; 
	string dev = (intgr->get_computing_device() == COMPUTING_DEVICE_CPU ? "CPU" : "GPU");

	cout << "[" << dev << "] " 
		 << tools::get_time_stamp(false) 
		 << " t: " << setprecision(4) << setw(10) << nbd->t / constants::Gauss 
		 << ", dt: " << setprecision(4) << setw(10) << dt / constants::Gauss
		 << " (" << setprecision(4) << setw(10) << (nbd->t / constants::Gauss)/intgr->get_n_passed_step() << ") [d]";
	cout << ", dT: " << setprecision(4) << setw(10) << *dT_CPU / (double)CLOCKS_PER_SEC;
	cout << " (" << setprecision(4) << setw(10) << (*T_CPU / (double)CLOCKS_PER_SEC) / intgr->get_n_passed_step() << ") [s]";
	cout << ", Nc: " << setw(5) << nbd->n_collision[  EVENT_COUNTER_NAME_TOTAL]
	     << ", Ne: " << setw(5) << nbd->n_ejection[   EVENT_COUNTER_NAME_TOTAL]
		 << ", Nh: " << setw(5) << nbd->n_hit_centrum[EVENT_COUNTER_NAME_TOTAL]
		 << ", N : " << setw(6) << nb->get_n_total_playing() << "(" << setw(3) << nb->get_n_total_inactive() << ", " << setw(5) << nb->get_n_total_removed() << ")"
	     << ", nt: " << setw(11) << intgr->get_n_tried_step()
	     << ", np: " << setw(11) << intgr->get_n_passed_step()
	     << ", nf: " << setw(11) << intgr->get_n_failed_step()
		 << endl;

	if (first_call)
	{
		first_call = false;
		sout << header_str << endl;
	}

	sout << dev << cvs
		 << tools::get_time_stamp(true) << cvs
		 << setprecision(4) << nbd->t / constants::Gauss << cvs
		 << setprecision(4) << dt / constants::Gauss << cvs
		 << setprecision(4) << (nbd->t / constants::Gauss)/intgr->get_n_passed_step() << cvs;
	sout << setprecision(4) << (*dT_CPU  / (double)CLOCKS_PER_SEC) << cvs;
	sout << setprecision(4) << (*T_CPU / (double)CLOCKS_PER_SEC) / intgr->get_n_passed_step() << cvs;
	sout << nbd->n_collision[  EVENT_COUNTER_NAME_TOTAL] << cvs
	     << nbd->n_ejection[   EVENT_COUNTER_NAME_TOTAL] << cvs
		 << nbd->n_hit_centrum[EVENT_COUNTER_NAME_TOTAL] << cvs
		 << nb->get_n_total_playing()  << cvs
		 << nb->get_n_total_inactive() << cvs
		 << nb->get_n_total_removed()  << cvs
	     << intgr->get_n_tried_step()  << cvs
	     << intgr->get_n_passed_step() << cvs
	     << intgr->get_n_failed_step() << cvs;
	for (int i = 0; i < BODY_TYPE_N; i++)
	{
		sout << nb->playing[i] - nb->inactive[i] << cvs
			 << nb->removed[i] << (i < BODY_TYPE_TESTPARTICLE ? cvs : "");
	}
	sout << endl;
}

void run_simulation(const options& opt, nbody* nbd, integrator* intgr, ofstream** output)
{
	ttt_t ps = 0.0;
	ttt_t dt = 0.0;

	clock_t T_CPU = 0;
	clock_t dT_CPU = 0;

	time_t time_last_info = clock();
	time_t time_last_dump = clock();

	integral_t integrals[2];

	uint32_t n_removed = 0;
	uint32_t n_dump = 1;

	nbd->print_result(*output[OUTPUT_NAME_RESULT], DATA_REPRESENTATION_ASCII);

/* main cycle */
#if 1
	while (nbd->t <= opt.param->stop_time && 1 < nbd->n_bodies->get_n_total_active())
	{
		// make the integration step, and measure the time it takes
		clock_t T0_CPU = clock();
		dt = intgr->step();
		dT_CPU = (clock() - T0_CPU);
		T_CPU += dT_CPU;
		ps += fabs(dt);

		if (0.0 < opt.param->threshold[THRESHOLD_RADII_ENHANCE_FACTOR])
		{
			integral_t I;

			bool collision = nbd->check_for_collision();
			if (collision)
			{
				integral_t I;

				nbd->calc_integral(false, I);
				nbd->print_integral_data(I, *output[OUTPUT_NAME_INTEGRAL_EVENT]);

				{
					// Restore the state before the collision
					nbd->swap();
					nbd->t -= dt;
					if (COMPUTING_DEVICE_GPU == opt.comp_dev)
					{
						nbd->copy_event_data_to_host();
					}
					nbd->populate_sp_events();
					for (uint32_t i = 0; i < nbd->sp_events.size(); i++)
					{
						// Create the subsystem containing the colliding bodies
						nbody* reg_tbp = nbd->create_reg_tbp(nbd->sp_events[i].idx1, nbd->sp_events[i].idx2);
						integrator* reg_int = new rungekutta8(reg_tbp, dt, true, 1.0e-15, opt.comp_dev);
						cout << "Subsystem with " << nbd->sp_events[i].id1 << " and " << nbd->sp_events[i].id2 << " was created." << endl;

						//uint32_t ictv_idx = nbd->sp_events[i] .idx2;

						//handle_collision_pair(i, &sp_events[i]);
						//increment_event_counter(n_collision);
						//// Make the merged body inactive 
						//sim_data->h_body_md[ictv_idx].id *= -1;
						//// Copy it up to GPU 
						//if (COMPUTING_DEVICE_GPU == comp_dev)
						//{
						//	copy_vector_to_device((void **)&sim_data->d_body_md[ictv_idx], (void **)&sim_data->h_body_md[ictv_idx], sizeof(body_metadata_t));
						//}
						//// Update number of inactive bodies
						//n_bodies->inactive[sim_data->h_body_md[ictv_idx].body_type]++;
					}


				}

				//nbd->handle_collision();
				//nbd->calc_integral(false, I);
				//nbd->print_integral_data(I, *output[OUTPUT_NAME_INTEGRAL_EVENT]);
				//nbd->print_event_data(*output[OUTPUT_NAME_EVENT], *output[OUTPUT_NAME_LOG]);
				nbd->clear_event_counter();
			}
		}

		if (opt.param->output_interval <= fabs(ps))
		{
			integral_t I;

			ps = 0.0;
			nbd->print_result(*output[OUTPUT_NAME_RESULT], DATA_REPRESENTATION_ASCII);
			nbd->calc_integral(false, I);
			nbd->print_integral_data(I, *output[OUTPUT_NAME_INTEGRAL]);
		}

		if (opt.info_dt < (clock() - time_last_info) / (double)CLOCKS_PER_SEC) 
		{
			time_last_info = clock();
			print_info(*output[OUTPUT_NAME_INFO], nbd, intgr, dt, &T_CPU, &dT_CPU);
		}
	} /* while */
#endif

	print_info(*output[OUTPUT_NAME_INFO], nbd, intgr, dt, &T_CPU, &dT_CPU);
	// To avoid duplicate save at the end of the simulation
	if (0.0 < ps)
	{
		ps = 0.0;
		nbd->print_result(*output[OUTPUT_NAME_RESULT], DATA_REPRESENTATION_ASCII);
	}
}


int main(int argc, const char** argv, const char** env)
{
	time_t start = time(NULL);

	ofstream* output[OUTPUT_NAME_N];
	memset(output, 0x0, sizeof(output));

	try
	{
		options opt = options(argc, argv);
		open_streams(opt, output);
		file::log_start(*output[OUTPUT_NAME_LOG], argc, argv, env, opt.param->cdm, opt.print_to_screen);
		nbody *nbd = opt.create_nbody();

		ttt_t dt = 0.1; // [day]
		integrator *intgr = opt.create_integrator(nbd, dt);

		run_simulation(opt, nbd, intgr, output);
		// Needed by nvprof.exe
		if (COMPUTING_DEVICE_GPU == nbd->get_computing_device())
		{
			cudaDeviceReset();
		}

	} /* try */
	catch (const string& msg)
	{
		if (0x0 != output[OUTPUT_NAME_LOG])
		{
			file::log_message(*output[OUTPUT_NAME_LOG], "Error: " + msg, false);
		}
		cerr << "Error: " << msg << endl;
	}

	if (0x0 != output[OUTPUT_NAME_LOG])
	{
		file::log_message(*output[OUTPUT_NAME_LOG], "Total time: " + tools::convert_time_t(time(NULL) - start) + " s", false);
	}
	cout << "Total time: " << time(NULL) - start << " s" << endl;

	return (EXIT_SUCCESS);
}
