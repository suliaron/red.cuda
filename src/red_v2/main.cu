#include <ctime>
#include <iostream>
#include <fstream>

#include "integrator.h"
#include "ode.h"
#include "options.h"
#include "rtbp1D.h"
#include "tbp1D.h"
#include "rtbp3D.h"
#include "tbp3D.h"

#include "red_type.h"
#include "redutilcu.h"

typedef struct system
			{
				ode* f;
				integrator* intgr;
			} system_t;

using namespace std;
using namespace redutilcu;

string create_prefix(const options* opt)
{
	static const char* integrator_type_short_name[] = 
	{
				"E",
				"RK2",
				"RK4",
				"RKF5",
				"RKF7"
	};

	string prefix;

	if (opt->ef)
	{
		char sep = '_';
		string config;
#ifdef _DEBUG
		config = "D";
#else
		config = "R";
#endif
		string dev = (opt->comp_dev == COMPUTING_DEVICE_CPU ? "cpu" : "gpu");
		// as: adaptive step-size, fs: fix step-size
		string adapt = (opt->param->adaptive == true ? "as" : "fs");
		
		string int_name(integrator_type_short_name[opt->param->int_type]);
		prefix += config + sep + dev + sep + adapt + sep + int_name + sep;
	}

	return prefix;
}

void open_streams(const options* opt, ofstream** output, data_rep_t output_data_rep)
{
	string path;
	string prefix = create_prefix(opt);
	string ext = (DATA_REPRESENTATION_ASCII == output_data_rep ? "txt" : "bin");

	// Result
	path = file::combine_path(opt->dir[DIRECTORY_NAME_OUT], prefix + opt->out_fn[OUTPUT_NAME_DATA]) + "." + ext;
	output[OUTPUT_NAME_DATA] = (DATA_REPRESENTATION_ASCII == output_data_rep ? new ofstream(path.c_str(), ios::out) : new ofstream(path.c_str(), ios::out | ios::binary));
	if (!*output[OUTPUT_NAME_DATA]) 
	{
		throw string("Cannot open " + path + ".");
	}

	// First integrals
	path = file::combine_path(opt->dir[DIRECTORY_NAME_OUT], prefix + opt->out_fn[OUTPUT_NAME_INTEGRAL]) + "." + ext;
	output[OUTPUT_NAME_INTEGRAL] = (DATA_REPRESENTATION_ASCII == output_data_rep ? new ofstream(path.c_str(), ios::out) : new ofstream(path.c_str(), ios::out | ios::binary));
	if (!*output[OUTPUT_NAME_INTEGRAL]) 
	{
		throw string("Cannot open " + path + ".");
	}

	// Log
	path = file::combine_path(opt->dir[DIRECTORY_NAME_OUT], prefix + opt->out_fn[OUTPUT_NAME_LOG]) + ".txt";
	output[OUTPUT_NAME_LOG] = new ofstream(path.c_str(), ios::out);
	if (!*output[OUTPUT_NAME_LOG]) 
	{
		throw string("Cannot open " + path + ".");
	}
}

void run_simulation(const options* opt, ode* f, integrator* intgr, ofstream** output)
{
	ttt_t ps = 0.0;
	ttt_t dt = 0.0;

	clock_t T_CPU = 0;
	clock_t dT_CPU = 0;

	time_t time_last_info = clock();
	time_t time_last_dump = clock();

	/* 
	 * Main cycle
	 */
	//print_result(opt, f, output);
	f->print_result(output, opt->param->output_data_rep);

	while (f->t <= opt->param->stop_time)
	{
		// make the integration step, and measure the time it takes
		clock_t T0_CPU = clock();
		dt = intgr->step();
		dT_CPU = (clock() - T0_CPU);
		T_CPU += dT_CPU;
		ps += fabs(dt);

		if (opt->param->output_interval <= fabs(ps))
		{
			ps = 0.0;
			f->print_result(output, opt->param->output_data_rep);
		}

		if (opt->info_dt < (clock() - time_last_info) / (double)CLOCKS_PER_SEC) 
		{
			time_last_info = clock();
			//print_info(*output[OUTPUT_NAME_INFO], ppd, intgr, dt, &T_CPU, &dT_CPU);
		}
	} /* while : main cycle*/
}

void simulator(options* opt, vector<system_t>& systems, fstream** output)
{
	systems[0].intgr->step();
	
	// Create new system? check distance from central body
	if (true)
	{
		var3_t r = {systems[0].f->h_y[0], systems[0].f->h_y[1], systems[0].f->h_y[2]};
		var3_t v = {systems[0].f->h_y[3], systems[0].f->h_y[4], systems[0].f->h_y[5]};
		var4_t u, uv;
		ttt_t t = 0.0;
		tbp3D* model = dynamic_cast<tbp3D*>(systems[0].f);
		
		rtbp3D::trans_to_parameter(r, v, u, uv);
		ode* f = new rtbp3D(1, t, model->h_md, model->h_p, &r, &v, opt->comp_dev);
		integrator* intgr = opt->create_integrator(*f, 0.01);
	}
}

int main(int argc, const char** argv, const char** env)
{
	time_t start = time(NULL);

	ofstream* output[OUTPUT_NAME_N];
	memset(output, 0x0, sizeof(output));

	ode* f = 0x0;
	options* opt = 0x0;

	try
	{
		opt = new options(argc, argv);

		open_streams(opt, output, (*opt).param->output_data_rep);
		file::log_start(*output[OUTPUT_NAME_LOG], argc, argv, env, opt->param->get_data(), opt->print_to_screen);

		switch (opt->dyn_model)
		{
		case DYN_MODEL_TBP1D:
			f = opt->create_tbp1D();
			break;
		case DYN_MODEL_RTBP1D:
			f = opt->create_rtbp1D();
			break;
		case DYN_MODEL_TBP3D:
			f = opt->create_tbp3D();
			break;
		case DYN_MODEL_RTBP3D:
			f = opt->create_rtbp3D();
			break;
		default:
			throw string("Invalid dynamical model.");
		}

		ttt_t dt = 0.01;
		integrator *intgr = opt->create_integrator(*f, dt);
		// TODO: For every model it should be provieded a method to determine the minimum stepsize
		// OR use the solution provided by the Numerical Recepies
		intgr->set_dt_min(1.0e-6); // day
		intgr->set_max_iter(10);


		system_t sys = {f, intgr};
		vector<system_t> systems;

		systems.push_back(sys);
		simulator(opt, systems, output);

		run_simulation(opt, f, intgr, output);

	} /* try */
	catch (const string& msg)
	{
		f->print_result(output,opt->param->output_data_rep);
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
