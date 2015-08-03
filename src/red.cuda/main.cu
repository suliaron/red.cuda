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
#include "nbody_exception.h"
#include "number_of_bodies.h"
#include "integrator.h"
#include "options.h"
#include "parameter.h"
#include "test.h"

#include "red_type.h"
#include "red_constants.h"
#include "redutilcu.h"

using namespace std;
using namespace redutilcu;

string create_prefix(const options& opt)
{
	string prefix;
	if (opt.benchmark)
	{
		prefix = "b_";
	}

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
		// ps: padded storage, ns: normal storage
		string strorage = (opt.ups == true ? "ps" : "ns");
		// 
		string cdm;
		switch (opt.param->cdm)
		{
		case COLLISION_DETECTION_MODEL_STEP:
			cdm = "bs";
			break;
		case COLLISION_DETECTION_MODEL_SUB_STEP:
			cdm = "ss";
			break;
		case COLLISION_DETECTION_MODEL_INTERPOLATION:
			throw string("COLLISION_DETECTION_MODEL_INTERPOLATION is not implemented.");
		default:
			throw string("Parameter 'cdm' is out of range.");
		}

		string int_name(integrator_type_short_name[opt.param->int_type]);
		prefix += config + sep + dev + sep + cdm + sep + strorage + sep + adapt + sep + int_name + sep;
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
}

void print_info(ofstream& sout, const pp_disk* ppd, integrator *intgr, ttt_t dt, clock_t* sum_time_of_steps, clock_t* time_of_one_step)
{
	static const string header_str = "dev,date      ,time    ,t [d]      ,dt [d]     ,dt_avg [d] ,dT [s]     ,dT_avg [s] ,Nc   ,Ne   ,Nh   ,nb_p ,nb_i,nb_r ,ns_t    ,ns_f    ,ns_p    ,ns_a,ns_r  ,ngp_a,ngp_r,nrp_a,nrp_r,npp_a,npp_r,nspl_a,nspl_r,npl_a,npl_r,ntp_a,ntp_r";
	static bool first_call = true;
	
	cout.setf(ios::right);
	cout.setf(ios::scientific);

	sout.setf(ios::right);
	sout.setf(ios::scientific);

	number_of_bodies* nb = ppd->n_bodies; 
	string dev = (intgr->get_computing_device() == COMPUTING_DEVICE_CPU ? "CPU" : "GPU");

	cout << "[" << dev << "] " 
		 << tools::get_time_stamp(false) 
		 << " t: " << setprecision(4) << setw(10) << ppd->t / constants::Gauss 
		 << ", dt: " << setprecision(4) << setw(10) << dt / constants::Gauss
		 << " (" << setprecision(4) << setw(10) << (ppd->t / constants::Gauss)/intgr->get_n_passed_step() << ") [d]";
	cout << ", dT: " << setprecision(4) << setw(10) << *time_of_one_step / (double)CLOCKS_PER_SEC;
	cout << " (" << setprecision(4) << setw(10) << (*sum_time_of_steps / (double)CLOCKS_PER_SEC) / intgr->get_n_passed_step() << ") [s]";
	cout << ", Nc: " << setw(5) << ppd->n_collision[  EVENT_COUNTER_NAME_TOTAL]
	     << ", Ne: " << setw(5) << ppd->n_ejection[   EVENT_COUNTER_NAME_TOTAL]
		 << ", Nh: " << setw(5) << ppd->n_hit_centrum[EVENT_COUNTER_NAME_TOTAL]
		 << ", N : " << setw(6) << nb->get_n_total_playing() << "(" << setw(3) << nb->get_n_total_inactive() << ", " << setw(5) << nb->get_n_total_removed() << ")"
	     << ", nt: " << setw(8) << intgr->get_n_tried_step()
	     << ", nf: " << setw(8) << intgr->get_n_failed_step()
	     << ", np: " << setw(8) << intgr->get_n_passed_step()
		 << endl;

	if (first_call)
	{
		first_call = false;
		sout << header_str << endl;
	}

	sout << dev << ","
		 << tools::get_time_stamp(true) << ","
		 << setprecision(4) << setw(10) << ppd->t / constants::Gauss << ","
		 << setprecision(4) << setw(10) << dt / constants::Gauss << ","
		 << setprecision(4) << setw(10) << (ppd->t / constants::Gauss)/intgr->get_n_passed_step() << ",";
	sout << setprecision(4) << setw(10) << (*time_of_one_step  / (double)CLOCKS_PER_SEC) << ",";
	sout << setprecision(4) << setw(10) << (*sum_time_of_steps / (double)CLOCKS_PER_SEC) / intgr->get_n_passed_step() << ",";
	sout << setw(5) << ppd->n_collision[  EVENT_COUNTER_NAME_TOTAL] << ","
	     << setw(5) << ppd->n_ejection[   EVENT_COUNTER_NAME_TOTAL] << ","
		 << setw(5) << ppd->n_hit_centrum[EVENT_COUNTER_NAME_TOTAL] << ","
		 << setw(6) << nb->get_n_total_playing()  << ","
		 << setw(3) << nb->get_n_total_inactive() << ","
		 << setw(5) << nb->get_n_total_removed()  << ","
	     << setw(8) << intgr->get_n_tried_step()  << ","
	     << setw(8) << intgr->get_n_failed_step() << ","
	     << setw(8) << intgr->get_n_passed_step() << ",";
	for (int i = 0; i < BODY_TYPE_N; i++)
	{
		if (BODY_TYPE_PADDINGPARTICLE == i)
		{
			continue;
		}
		sout << setw(5) << nb->playing[i] - nb->inactive[i] << ","
			<< setw(5) << nb->removed[i] << (i < BODY_TYPE_TESTPARTICLE ? "," : "");
	}
	sout << endl;
}

void print_dump_aux_data(ofstream& sout, dump_aux_data_t* dump_aux)
{
	sout.write((char*)(dump_aux), sizeof(dump_aux_data_t));
}

void print_dump(ofstream **output, const options& opt, pp_disk* ppd, ttt_t dt, int n_dump, data_representation_t repres)
{
	string prefix = create_prefix(opt);
	string ext = (repres == DATA_REPRESENTATION_ASCII ? "txt" : "bin");
	string path = file::combine_path(opt.dir[DIRECTORY_NAME_OUT], prefix + opt.out_fn[OUTPUT_NAME_DUMP] + redutilcu::number_to_string(n_dump) + "_" + ppd->n_bodies->get_n_playing() + "." + ext);
	output[OUTPUT_NAME_DUMP] = new ofstream(path.c_str(), ios::out | ios::binary);
	if(!output[OUTPUT_NAME_DUMP])
	{
		throw string("Cannot open " + path + ".");
	}
	ppd->print_dump(*output[OUTPUT_NAME_DUMP], repres);
	//output[OUTPUT_NAME_DUMP]->~ofstream();
	delete output[OUTPUT_NAME_DUMP];

	dump_aux_data_t dump_aux;
	dump_aux.dt = dt;

	path = file::combine_path(opt.dir[DIRECTORY_NAME_OUT], prefix + opt.out_fn[OUTPUT_NAME_DUMP_AUX] + redutilcu::number_to_string(n_dump) + "_" + ppd->n_bodies->get_n_playing() + ".dat");
	output[OUTPUT_NAME_DUMP_AUX] = new ofstream(path.c_str(), ios::out | ios::binary);
	if(!output[OUTPUT_NAME_DUMP_AUX])
	{
		throw string("Cannot open " + path + ".");
	}
	print_dump_aux_data(*output[OUTPUT_NAME_DUMP_AUX], &dump_aux);
	delete output[OUTPUT_NAME_DUMP_AUX];
}

void read_dump_aux_data(ifstream& input, dump_aux_data_t* dump_aux)
{
	input.read((char*)(dump_aux), sizeof(dump_aux_data_t));
}

dump_aux_data_t load_dump_aux_data(const options& opt)
{
	dump_aux_data_t dump_aux;

	string path = file::combine_path(opt.dir[DIRECTORY_NAME_IN], file::get_filename_without_ext(opt.in_fn[INPUT_NAME_BODYLIST]) + ".aux.dat");
	ifstream input(path.c_str(), ios::in | ios::binary);
	if (input) 
	{
		read_dump_aux_data(input, &dump_aux);
	}
	else
	{
		throw string("Cannot open " + path + ".");
	}
	input.close();

	return dump_aux;
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

void run_benchmark(const options& opt, pp_disk* ppd, integrator* intgr, ofstream& sout)
{
	cout << "See the log file for the result." << endl;

	sout.setf(ios::right);
	sout.setf(ios::scientific);

	sout.setf(ios::right);
	sout.setf(ios::scientific);

	size_t size = ppd->n_bodies->get_n_total_playing() * sizeof(vec_t);

	// Create aliases
	vec_t* r   = ppd->sim_data->y[0];
	vec_t* v   = ppd->sim_data->y[1];
	param_t* p = ppd->sim_data->p;
	body_metadata_t* bmd = ppd->sim_data->body_md;

	sout << endl;
	ttt_t curr_t = 0.0;
	if (COMPUTING_DEVICE_GPU == opt.comp_dev)
	{
		sout << "----------------------------------------------" << endl;
		sout << "GPU:" << endl;
		sout << "----------------------------------------------" << endl << endl;

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, opt.id_dev);

		int half_warp_size = deviceProp.warpSize/2;
		vector<float2> execution_time;

		vec_t* d_dy = 0x0;
		ALLOCATE_DEVICE_VECTOR((void**)&d_dy, size);

		unsigned int n_sink = ppd->n_bodies->get_n_SI();
		unsigned int n_pass = 0;
		if (0 < n_sink)
		{
			sout << "SI:" << endl;
			sout << "----------------------------------------------" << endl;

			for (int n_tpb = half_warp_size; n_tpb <= deviceProp.maxThreadsPerBlock; n_tpb += half_warp_size)
			{
				sout << "n_tpb: " << setw(6) << n_tpb;
				ppd->set_n_tpb(n_tpb);
				interaction_bound int_bound = ppd->n_bodies->get_bound_SI(false, n_tpb);

				clock_t t_start = clock();
				float cu_elt = ppd->benchmark_calc_grav_accel(curr_t, n_sink, int_bound, bmd, p, r, v, d_dy);
				ttt_t elapsed_time = ((double)(clock() - t_start)/(double)CLOCKS_PER_SEC) * 1000.0; // [ms]

				cudaError_t cuda_status = cudaGetLastError();
				if (cudaSuccess != cuda_status)
				{
					break;
				}

				float2 exec_t = {(float)elapsed_time, cu_elt};
				execution_time.push_back(exec_t);

				sout << " dt: " << setprecision(6) << setw(6) << elapsed_time << " (" << setw(6) << cu_elt << ") [ms]" << endl;
				n_pass++;
			}

			float min_y = 1.0e10;
			int min_idx = 0;
			for (unsigned int i = 0; i < n_pass; i++)
			{
				if (min_y > execution_time[i].y)
				{
					min_y = execution_time[i].y;
					min_idx = i;
				}
			}
			sout << "Minimum at n_tpb = " << ((min_idx + 1) * half_warp_size) << ", where execution time is: " << execution_time[min_idx].y << " [ms]" << endl;
		}

		FREE_DEVICE_VECTOR((void**)&d_dy);
		// Needed by nvprof.exe
		cudaDeviceReset();
	} /* if */
	else
	{
		sout << "----------------------------------------------" << endl;
		sout << "CPU:" << endl;
		sout << "----------------------------------------------" << endl << endl;

		clock_t t_start;
		ttt_t elapsed_time;

		vec_t* h_dy = 0x0;
		ALLOCATE_HOST_VECTOR((void**)&h_dy, size);

		int n_sink = ppd->n_bodies->get_n_SI();
		if (0 < n_sink)
		{
			interaction_bound int_bound = ppd->n_bodies->get_bound_SI(false, 1);

			t_start = clock();
			ppd->cpu_calc_grav_accel_SI(curr_t, int_bound, bmd, p, r, v, h_dy, 0x0, 0x0);
			elapsed_time = ((double)(clock() - t_start)/(double)CLOCKS_PER_SEC) * 1000.0; // [ms]

			sout << "SI:" << endl;
			sout << "----------------------------------------------" << endl;
			sout << "dt: " << setprecision(10) << setw(16) << elapsed_time << " [ms]" << endl;
		}

		n_sink = ppd->n_bodies->get_n_NSI();
		if (0 < n_sink)
		{
			interaction_bound int_bound = ppd->n_bodies->get_bound_NSI(false, 1);

			t_start = clock();
			ppd->cpu_calc_grav_accel_NSI(curr_t, int_bound, bmd, p, r, v, h_dy, 0x0, 0x0);
			elapsed_time = ((double)(clock() - t_start)/(double)CLOCKS_PER_SEC) * 1000.0; // [ms]

			sout << "NSI:" << endl;
			sout << "----------------------------------------------" << endl;
			sout << "dt: " << setprecision(10) << setw(16) << elapsed_time << " [ms]" << endl;
		}

		n_sink = ppd->n_bodies->get_n_NI();
		if (0 < n_sink)
		{
			interaction_bound int_bound = ppd->n_bodies->get_bound_NI(false, 1);

			t_start = clock();			
			ppd->cpu_calc_grav_accel_NI(curr_t, int_bound, bmd, p, r, v, h_dy, 0x0, 0x0);
			elapsed_time = ((double)(clock() - t_start)/(double)CLOCKS_PER_SEC) * 1000.0; // [ms]

			sout << "NI:" << endl;
			sout << "----------------------------------------------" << endl;
			sout << "dt: " << setprecision(10) << setw(16) << elapsed_time << " [ms]" << endl;
		}
	}
}

void run_simulation(const options& opt, pp_disk* ppd, integrator* intgr, ofstream** output)
{
	ttt_t ps = 0.0;
	ttt_t dt = 0.0;
	clock_t sum_time_of_steps = 0;
	clock_t time_of_one_step  = 0;

	time_t time_info_start = clock();
	time_t time_of_last_dump = clock();

	unsigned int n_removed = 0;
	unsigned int n_dump = 0;

	if (COMPUTING_DEVICE_GPU == ppd->get_computing_device())
	{
		unsigned int n_tpb = ppd->benchmark();
		ppd->set_n_tpb(n_tpb);
		if (opt.verbose)
		{
			string msg = "Number of thread per block was set to " + redutilcu::number_to_string(ppd->get_n_tpb());
			file::log_message(*output[OUTPUT_NAME_LOG], msg, opt.print_to_screen);
		}
	}

	//ppd->print_result_ascii(*output[OUTPUT_NAME_RESULT]);


	while (ppd->t <= opt.param->stop_time && 1 < ppd->n_bodies->get_n_total_active())
	{
		if (COMPUTING_DEVICE_GPU == intgr->get_computing_device() && opt.n_change_to_cpu >= ppd->n_bodies->get_n_SI())
		{
			intgr->set_computing_device(COMPUTING_DEVICE_CPU);
			if (opt.verbose)
			{
				string msg = "Number of self-interacting bodies dropped below " + redutilcu::number_to_string(opt.n_change_to_cpu) + ". Execution was transferred to CPU.";
				file::log_message(*output[OUTPUT_NAME_LOG], msg, opt.print_to_screen);
			}
		}

		if ((0.0 < opt.param->threshold[THRESHOLD_EJECTION_DISTANCE] || 0.0 < opt.param->threshold[THRESHOLD_HIT_CENTRUM_DISTANCE]))
		{
			bool eje_hc = ppd->check_for_ejection_hit_centrum();
			if (eje_hc)
			{
				ppd->handle_ejection_hit_centrum();
				ppd->print_event_data(*output[OUTPUT_NAME_EVENT], *output[OUTPUT_NAME_LOG]);
				ppd->clear_event_counter();
			}
		}

		dt = step(intgr, &sum_time_of_steps, &time_of_one_step);
		ps += fabs(dt);

		if (0.0 < opt.param->threshold[THRESHOLD_RADII_ENHANCE_FACTOR])
		{
			bool collision = false;
			switch (opt.param->cdm)
			{
			case COLLISION_DETECTION_MODEL_STEP:
			case COLLISION_DETECTION_MODEL_SUB_STEP:
				collision = ppd->check_for_collision();
				if (collision)
				{
					ppd->handle_collision();
					ppd->print_event_data(*output[OUTPUT_NAME_EVENT], *output[OUTPUT_NAME_LOG]);
					ppd->clear_event_counter();
				}
				break;
			case COLLISION_DETECTION_MODEL_INTERPOLATION:
				throw string("COLLISION_DETECTION_MODEL_INTERPOLATION is not implemented.");
			default:
				throw string("Parameter 'cdm' is out of range.");
			}
		}

		if (opt.param->output_interval <= fabs(ps))
		{
			ps = 0.0;
			if (COMPUTING_DEVICE_GPU == ppd->get_computing_device())
			{
				ppd->copy_to_host();
			}
			ppd->print_result_ascii(*output[OUTPUT_NAME_RESULT]);
		}

		if (4 <= ppd->n_event[EVENT_COUNTER_NAME_LAST_CLEAR])
		{
			ppd->set_event_counter(EVENT_COUNTER_NAME_LAST_CLEAR, 0);
			ppd->rebuild_vectors();
			n_removed += ppd->n_bodies->n_removed;
			string msg = "Rebuild the vectors (removed " + redutilcu::number_to_string(ppd->n_bodies->n_removed) + " inactive bodies at t: " + redutilcu::number_to_string(ppd->t / constants::Gauss) + " [d])";
			file::log_message(*output[OUTPUT_NAME_LOG], msg, opt.print_to_screen);

			if (COMPUTING_DEVICE_GPU == ppd->get_computing_device())
			{
				unsigned int n_tpb = ppd->benchmark();
				ppd->set_n_tpb(n_tpb);
				if (opt.verbose)
				{
					string msg = "Number of thread per block was set to " + redutilcu::number_to_string(ppd->get_n_tpb());
					file::log_message(*output[OUTPUT_NAME_LOG], msg, opt.print_to_screen);
				}
			}
		}

		//if (10 <= n_removed || opt.dump_dt < (clock() - time_of_last_dump) / (double)CLOCKS_PER_SEC)
		//{
		//	n_removed = 0;
		//	time_of_last_dump = clock();
		//	if (COMPUTING_DEVICE_GPU == ppd->get_computing_device())
		//	{
		//		ppd->copy_to_host();
		//	}
		//	print_dump(output, opt, ppd, dt, n_dump, DATA_REPRESENTATION_BINARY);
		//	n_dump++;
		//}

		if (opt.info_dt < (clock() - time_info_start) / (double)CLOCKS_PER_SEC) 
		{
			time_info_start = clock();
			print_info(*output[OUTPUT_NAME_INFO], ppd, intgr, dt, &sum_time_of_steps, &time_of_one_step);
		}
	} /* while */


	print_info(*output[OUTPUT_NAME_INFO], ppd, intgr, dt, &sum_time_of_steps, &time_of_one_step);
	// To avoid duplicate save at the end of the simulation
	if (0.0 < ps)
	{
		if (COMPUTING_DEVICE_GPU == ppd->get_computing_device())
		{
			ppd->copy_to_host();
		}
		ppd->print_result_ascii(*output[OUTPUT_NAME_RESULT]);
	}

	// Needed by nvprof.exe
	if (COMPUTING_DEVICE_GPU == ppd->get_computing_device())
	{
		cudaDeviceReset();
	}
}

void run_test()
{
	test_number_of_bodies();
}

//http://stackoverflow.com/questions/11666049/cuda-kernel-results-different-in-release-mode
//http://developer.download.nvidia.com/assets/cuda/files/NVIDIA-CUDA-Floating-Point.pdf

//-gpu -v -pts -ef -iDir C:\Work\red.cuda.Results\Dvorak\2D\NewRun_2\Run_cf4.0_2 -p parameters.txt -ic run_04.txt
int main(int argc, const char** argv, const char** env)
{
	time_t start = time(NULL);

	ofstream* output[OUTPUT_NAME_N];
	memset(output, 0x0, sizeof(output));

	try
	{
		options opt = options(argc, argv);

		if (opt.test)
		{
			run_test();
			return (EXIT_SUCCESS);
		}
		
		open_streams(opt, output);

		file::log_start(*output[OUTPUT_NAME_LOG], argc, argv, env, opt.param->cdm, opt.print_to_screen);

		if (COMPUTING_DEVICE_GPU == opt.comp_dev)
		{
			set_device(opt.id_dev, std::cout);
			device_query(*output[OUTPUT_NAME_LOG], opt.id_dev, opt.print_to_screen);
		}
		else
		{
			opt.ups = false;
		}

		pp_disk *ppd = opt.create_pp_disk();

		ttt_t dt = 0.1; // [day]
		if (opt.continue_simulation)
		{
			string msg = "Simulation continues from t = " + redutilcu::number_to_string(ppd->t / constants::Gauss) + " [day]";
			file::log_message(*output[OUTPUT_NAME_LOG], msg, opt.print_to_screen);

			dump_aux_data_t dump_aux = load_dump_aux_data(opt);
			dt = dump_aux.dt / constants::Gauss;
		}

		integrator *intgr = opt.create_integrator(ppd, dt);

		if (opt.benchmark)
		{
			run_benchmark(opt, ppd, intgr, *output[OUTPUT_NAME_LOG]);
			return (EXIT_SUCCESS);
		}

		run_simulation(opt, ppd, intgr, output);

	} /* try */
	catch (const nbody_exception& ex)
	{
		if (0x0 != output[OUTPUT_NAME_LOG])
		{
			file::log_message(*output[OUTPUT_NAME_LOG], "Error: " + string(ex.what()), false);
		}
		cerr << "Error: " << ex.what() << endl;
	}
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
