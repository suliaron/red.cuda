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
		string int_name(integrator_type_short_name[opt.param->int_type]);
		prefix += config + sep + dev + sep + strorage + sep + adapt + sep + int_name + sep;
	}

	return prefix;
}

void open_streams(const options& opt, ofstream** output)
{
	string path;
	string prefix = create_prefix(opt);
	string ext = "txt";

	path = file::combine_path(opt.printout_dir, prefix + opt.result_filename) + "." + ext;
	output[OUTPUT_NAME_RESULT] = new ofstream(path.c_str(), ios::out);
	if (!*output[OUTPUT_NAME_RESULT]) 
	{
		throw string("Cannot open " + path + ".");
	}

	path = file::combine_path(opt.printout_dir, prefix + opt.info_filename) + "." + ext;
	output[OUTPUT_NAME_INFO] = new ofstream(path.c_str(), ios::out);
	if (!*output[OUTPUT_NAME_INFO]) 
	{
		throw string("Cannot open " + path + ".");
	}

	path = file::combine_path(opt.printout_dir, prefix + opt.event_filename) + "." + ext;
	output[OUTPUT_NAME_EVENT] = new ofstream(path.c_str(), ios::out);
	if (!*output[OUTPUT_NAME_EVENT]) 
	{
		throw string("Cannot open " + path + ".");
	}

	path = file::combine_path(opt.printout_dir, prefix + opt.log_filename) + "." + ext;
	output[OUTPUT_NAME_LOG] = new ofstream(path.c_str(), ios::out);
	if (!*output[OUTPUT_NAME_LOG]) 
	{
		throw string("Cannot open " + path + ".");
	}
}

void print_info(ofstream& sout, const pp_disk* ppd, integrator *intgr, ttt_t dt, clock_t* sum_time_of_steps, clock_t* time_of_one_step, time_t* time_info_start)
{
	static const char* body_type_name[] = {"N_st", "N_gp", "N_rp", "N_pp", "N_sp", "N_pl", "N_tp"};
	
	cout.setf(ios::right);
	cout.setf(ios::scientific);

	sout.setf(ios::right);
	sout.setf(ios::scientific);

	number_of_bodies* nb = ppd->n_bodies; 
	string dev = (intgr->get_computing_device() == COMPUTING_DEVICE_CPU ? "CPU" : "GPU");

	*time_info_start = clock();
	cout << "[" << dev << "] " 
		 << tools::get_time_stamp() 
		 << " t: " << setprecision(6) << setw(12) << ppd->t / constants::Gauss
		 << ", dt: " << setprecision(6) << setw(12)  << dt / constants::Gauss;
	cout << ", dT: " << setprecision(3) << setw(10) << *time_of_one_step / (double)CLOCKS_PER_SEC << " s";
	cout << " (" << setprecision(3) << setw(10) << (*sum_time_of_steps / (double)CLOCKS_PER_SEC) / intgr->get_n_passed_step() << " s)";
	cout << ", Nc: " << setw(5) << ppd->n_collision[  EVENT_COUNTER_NAME_TOTAL]
	     << ", Ne: " << setw(5) << ppd->n_ejection[   EVENT_COUNTER_NAME_TOTAL]
		 << ", Nh: " << setw(5) << ppd->n_hit_centrum[EVENT_COUNTER_NAME_TOTAL]
		 << ", N : " << setw(6) << nb->get_n_total_playing() << "(" << setw(3) << nb->get_n_total_inactive() << ", " << setw(5) << nb->get_n_total_removed() << ")" << endl;

	sout << "[" << dev << "] " 
		 << tools::get_time_stamp()
		 << " t: " << setprecision(6) << setw(12) << ppd->t / constants::Gauss
		 << ", dt: " << setprecision(6) << setw(12)  << dt / constants::Gauss;
	sout << ", dT: " << setprecision(3) << setw(10) << *time_of_one_step / (double)CLOCKS_PER_SEC << " s";
	sout << " (" << setprecision(3) << setw(10) << (*sum_time_of_steps / (double)CLOCKS_PER_SEC) / intgr->get_n_passed_step() << " s)";
	sout << ", Nc: " << setw(5) << ppd->n_collision[  EVENT_COUNTER_NAME_TOTAL]
	     << ", Ne: " << setw(5) << ppd->n_ejection[   EVENT_COUNTER_NAME_TOTAL]
		 << ", Nh: " << setw(5) << ppd->n_hit_centrum[EVENT_COUNTER_NAME_TOTAL]
		 << ", N : " << setw(6) << nb->get_n_total_playing() << "(" << setw(3) << nb->get_n_total_inactive() << ", " << setw(5) << nb->get_n_total_removed() << ")";
	for (int i = 0; i < BODY_TYPE_N; i++)
	{
		if (BODY_TYPE_PADDINGPARTICLE == i)
		{
			continue;
		}
		sout << ", " << body_type_name[i] << ": " << setw(5) << nb->playing[i] - nb->inactive[i] << "(" << setw(5) << nb->removed[i] << ")";
	}
	sout << endl;
}

void print_dump(ofstream *output, const options& opt, pp_disk* ppd, int n_dump, data_representation_t repres)
{
	string prefix = create_prefix(opt);
	string ext = (repres == DATA_REPRESENTATION_ASCII ? "txt" : "bin");
	string path = file::combine_path(opt.printout_dir, prefix + "dump" + redutilcu::number_to_string(n_dump) + "_" + ppd->n_bodies->get_n_playing() + "." + ext);
	output = new ofstream(path.c_str(), ios::out | ios::binary);
	if(!output)
	{
		throw string("Cannot open " + path + ".");
	}
	ppd->print_dump(*output, repres);
	output->~ofstream();
}

void print_dump_aux_data(ofstream& sout, dump_aux_data_t* dump_aux)
{
	sout.write((char*)(dump_aux), sizeof(dump_aux_data_t));
}

void load_dump_aux_data(ifstream& input, dump_aux_data_t* dump_aux)
{
	input.read((char*)(dump_aux), sizeof(dump_aux_data_t));
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
		//int n_pass = (deviceProp.maxThreadsPerBlock - half_warp_size)/half_warp_size + 1;
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
				clock_t elapsed_time = clock() - t_start;

				cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
				if (cudaSuccess != cudaStatus)
				{
					string msg(cudaGetErrorString( cudaStatus ));
			        sout << " " << msg << endl;
					break;
				}

				float2 exec_t = {(float)elapsed_time, cu_elt};
				execution_time.push_back(exec_t);

				sout << " dt: " << setprecision(6) << setw(6) << elapsed_time << " (" << setw(6) << cu_elt << ") [ms]" << endl;
				n_pass++;
			}

			float min_y = 1.0e10;
			int min_idx = 0;
			for (int i = 0; i < n_pass; i++)
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
		clock_t elapsed_time;

		vec_t* h_dy = 0x0;
		ALLOCATE_HOST_VECTOR((void**)&h_dy, size);

		int n_sink = ppd->n_bodies->get_n_SI();
		if (0 < n_sink)
		{
			interaction_bound int_bound = ppd->n_bodies->get_bound_SI(false, 1);

			t_start = clock();
			ppd->cpu_calc_grav_accel_SI(curr_t, int_bound, bmd, p, r, v, h_dy, 0x0, 0x0);
			elapsed_time = clock() - t_start;
		}
		sout << "SI:" << endl;
		sout << "----------------------------------------------" << endl;
		sout << "dt: " << setprecision(10) << setw(16) << elapsed_time << " ms" << endl;

		n_sink = ppd->n_bodies->get_n_NSI();
		if (0 < n_sink)
		{
			interaction_bound int_bound = ppd->n_bodies->get_bound_NSI(false, 1);

			t_start = clock();
			ppd->cpu_calc_grav_accel_NSI(curr_t, int_bound, bmd, p, r, v, h_dy, 0x0, 0x0);
			elapsed_time = clock() - t_start;
		}
		sout << "NSI:" << endl;
		sout << "----------------------------------------------" << endl;
		sout << "dt: " << setprecision(10) << setw(16) << elapsed_time << " ms" << endl;

		n_sink = ppd->n_bodies->get_n_NI();
		if (0 < n_sink)
		{
			interaction_bound int_bound = ppd->n_bodies->get_bound_NI(false, 1);

			t_start = clock();			
			ppd->cpu_calc_grav_accel_NI(curr_t, int_bound, bmd, p, r, v, h_dy, 0x0, 0x0);
			elapsed_time = clock() - t_start;
		}
		sout << "NI:" << endl;
		sout << "----------------------------------------------" << endl;
		sout << "dt: " << setprecision(10) << setw(16) << elapsed_time << " ms" << endl;
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

	int dummy_k = 0;
	computing_device_t target_device = intgr->get_computing_device();

	unsigned int n_inactive = 10;
	unsigned int n_removed = 0;
	unsigned int n_dump = 0;

	ppd->benchmark();
	if (opt.verbose)
	{
		string msg = "Number of thread per block was set to " + redutilcu::number_to_string(ppd->get_n_tpb());
		file::log_message(*output[OUTPUT_NAME_LOG], msg, opt.print_to_screen);
	}

	while (1 < ppd->n_bodies->get_n_total_active() && ppd->t <= opt.param->stop_time)
	{
#if 0
		//if (0 < dummy_k && 0 == dummy_k % 10)
		if (n_removed >= 10)
		//if (ppd->n_bodies->get_n_total_inactive() >= n_inactive)
		{
			n_removed = 0;
			n_inactive += 10;

			int tmp = intgr->get_computing_device();
			tmp++;
			if (COMPUTING_DEVICE_N == tmp)
			{
				tmp = 0;
			}
			target_device = (computing_device_t)tmp;

			switch (target_device)
			{
			case COMPUTING_DEVICE_CPU:
				{
					intgr->set_computing_device(target_device);
					if (opt.verbose)
					{
						file::log_message(*output[OUTPUT_NAME_LOG], "Execution was transferred to CPU", opt.print_to_screen);
					}
					break;
				}
			case COMPUTING_DEVICE_GPU:
				{
					int id_of_target_GPU = redutilcu::get_id_fastest_cuda_device();
					redutilcu::set_device(id_of_target_GPU, *output[OUTPUT_NAME_LOG], opt.verbose, opt.print_to_screen);

					intgr->set_computing_device(target_device);
					if (opt.verbose)
					{
						file::log_message(*output[OUTPUT_NAME_LOG], "Execution was transferred to GPU " + redutilcu::get_name_cuda_device(id_of_target_GPU), opt.print_to_screen);
					}
					break;
				}
			default:
				{
					throw string ("Invalid parameter: target_device was out of range.");
				}
			}
		}
#endif
		if (COMPUTING_DEVICE_GPU == intgr->get_computing_device() && opt.n_change_to_cpu >= ppd->n_bodies->get_n_SI())
		{
			//ppd->copy_to_host();

			//string prefix = create_prefix(opt);
			//string path = file::combine_path(opt.printout_dir, prefix + "dump_before_change_to_GPU.dat");
			//output[OUTPUT_NAME_DUMP] = new ofstream(path.c_str(), ios::out | ios::binary);
			//ppd->print_dump(*output[OUTPUT_NAME_DUMP], DATA_REPRESENTATION_BINARY);
			//output[OUTPUT_NAME_DUMP]->~ostream();

			//path = file::combine_path(opt.printout_dir, prefix + "dump_before_change_to_GPU.txt");
			//output[OUTPUT_NAME_DUMP] = new ofstream(path.c_str());
			//ppd->print_dump(*output[OUTPUT_NAME_DUMP], DATA_REPRESENTATION_ASCII);
			//output[OUTPUT_NAME_DUMP]->~ostream();

			//dump_aux_data_t dump_aux;
			//dump_aux.dt = dt;

			//path = file::combine_path(opt.printout_dir, prefix + "dump_before_change_to_GPU.aux.dat");
			//output[OUTPUT_NAME_DUMP_AUX] = new ofstream(path.c_str(), ios::out | ios::binary);
			//print_dump_aux_data(*output[OUTPUT_NAME_DUMP_AUX], &dump_aux);
			//output[OUTPUT_NAME_DUMP_AUX]->~ostream();

			intgr->set_computing_device(COMPUTING_DEVICE_CPU);
			if (opt.verbose)
			{
				string msg = "Number of self-interacting bodies dropped below " + redutilcu::number_to_string(opt.n_change_to_cpu) + ". Execution was transferred to CPU.";
				file::log_message(*output[OUTPUT_NAME_LOG], msg, opt.print_to_screen);
			}
		}

		if ((0.0 < opt.param->thrshld[THRESHOLD_EJECTION_DISTANCE] || 
			 0.0 < opt.param->thrshld[THRESHOLD_HIT_CENTRUM_DISTANCE]) &&  ppd->check_for_ejection_hit_centrum())
		{
			ppd->print_event_data(*output[OUTPUT_NAME_EVENT], *output[OUTPUT_NAME_LOG]);
			ppd->clear_event_counter();
		}

		if (0.0 < opt.param->thrshld[THRESHOLD_RADII_ENHANCE_FACTOR] && ppd->check_for_collision())
		{
			ppd->print_event_data(*output[OUTPUT_NAME_EVENT], *output[OUTPUT_NAME_LOG]);
			ppd->clear_event_counter();
		}

		dt = step(intgr, &sum_time_of_steps, &time_of_one_step);
		ps += fabs(dt);

		if (opt.param->output_interval <= fabs(ps))
		{
			ps = 0.0;
			if (COMPUTING_DEVICE_GPU == ppd->get_computing_device())
			{
				ppd->copy_to_host();
			}
			ppd->print_result_ascii(*output[OUTPUT_NAME_RESULT]);
		}

		if (ppd->check_for_rebuild_vectors(4))
		{
			n_inactive = 0;
			n_removed += ppd->n_bodies->n_removed;
			string msg = "Rebuild the vectors (removed " + redutilcu::number_to_string(ppd->n_bodies->n_removed) + " inactive bodies at t: " + redutilcu::number_to_string(ppd->t / constants::Gauss) + ")";
			file::log_message(*output[OUTPUT_NAME_LOG], msg, opt.print_to_screen);

			if (COMPUTING_DEVICE_GPU == ppd->get_computing_device())
			{
				ppd->benchmark();
				if (opt.verbose)
				{
					string msg = "Number of thread per block was set to " + redutilcu::number_to_string(ppd->get_n_tpb());
					file::log_message(*output[OUTPUT_NAME_LOG], msg, opt.print_to_screen);
				}
			}
		}

		if (10 < n_removed || opt.dump_dt < (clock() - time_of_last_dump) / (double)CLOCKS_PER_SEC)
		{
			//n_removed = 0;
			//time_of_last_dump = clock();
			//if (COMPUTING_DEVICE_GPU == ppd->get_computing_device())
			//{
			//	ppd->copy_to_host();
			//}
			//print_dump(output[OUTPUT_NAME_DUMP], opt, ppd, n_dump, DATA_REPRESENTATION_BINARY);
			//n_dump++;
		}

		if (opt.info_dt < (clock() - time_info_start) / (double)CLOCKS_PER_SEC) 
		{
			print_info(*output[OUTPUT_NAME_INFO], ppd, intgr, dt, &sum_time_of_steps, &time_of_one_step, &time_info_start);
		}

		//dummy_k++;
	} /* while */
	print_info(*output[OUTPUT_NAME_INFO], ppd, intgr, dt, &sum_time_of_steps, &time_of_one_step, &time_info_start);
	// To avoid duplicate save at the end of the simulation
	if (0.0 < ps)
	{
		if (COMPUTING_DEVICE_GPU == ppd->get_computing_device())
		{
			ppd->copy_to_host();
		}
		ppd->print_result_ascii(*output[OUTPUT_NAME_RESULT]);
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
			exit(0);
		}

		ttt_t dt = 0.1; // [day]
		pp_disk *ppd = opt.create_pp_disk();
		open_streams(opt, output);

		if (opt.continue_simulation)
		{
			string msg = "Simulation continues from t = " + redutilcu::number_to_string(ppd->t / constants::Gauss) + " [day]";
			file::log_message(*output[OUTPUT_NAME_LOG], msg, opt.print_to_screen);

			string path = file::combine_path(opt.input_dir, file::get_filename_without_ext(opt.bodylist_filename) + ".aux.dat");
			ifstream input(path.c_str(), ios::in | ios::binary);
			if (input) 
			{
				dump_aux_data_t dump_aux;
				load_dump_aux_data(input, &dump_aux);
				dt = dump_aux.dt / constants::Gauss;
			}
			else
			{
				throw string("Cannot open " + path + ".");
			}
			input.close();
		}
		integrator *intgr = opt.create_integrator(ppd, dt);

		file::log_start_cmd(*output[OUTPUT_NAME_LOG], argc, argv, env, opt.print_to_screen);
		if (opt.verbose && COMPUTING_DEVICE_GPU == opt.comp_dev)
		{
			device_query(*output[OUTPUT_NAME_LOG], opt.id_dev, opt.print_to_screen);
		}

		if (opt.benchmark)
		{
			run_benchmark(opt, ppd, intgr, *output[OUTPUT_NAME_LOG]);
		}
		else
		{
			ppd->print_result_ascii(*output[OUTPUT_NAME_RESULT]);
			run_simulation(opt, ppd, intgr, output);
			// Needed by nvprof.exe
			if (COMPUTING_DEVICE_GPU == ppd->get_computing_device())
			{
				cudaDeviceReset();
			}
		} /* else benchmark */
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
