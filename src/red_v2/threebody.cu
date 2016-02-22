#include <iostream>
#include <iomanip>
#include <fstream>

#include "threebody.h"

#include "redutilcu.h"
#include "red_constants.h"

using namespace std;
using namespace redutilcu;


threebody::threebody(uint16_t n_ppo, computing_device_t comp_dev) :
	ode(3, 17, n_ppo, 1, comp_dev)
{
	initialize();
	allocate_storage();
}

threebody::~threebody()
{
	deallocate_storage();
}

void threebody::initialize()
{
	h_md    = 0x0;
	h_epoch = 0x0;

	h       = 0.0;
}

void threebody::allocate_storage()
{
	allocate_host_storage();
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		allocate_device_storage();
	}
}

void threebody::allocate_host_storage()
{
	ALLOCATE_HOST_VECTOR((void**)&(h_md),    n_obj * sizeof(threebody_t::metadata_t));
	ALLOCATE_HOST_VECTOR((void**)&(h_epoch), n_obj * sizeof(var_t));
}

void threebody::allocate_device_storage()
{
	ALLOCATE_DEVICE_VECTOR((void**)&(d_md),    n_obj * sizeof(threebody_t::metadata_t));
	ALLOCATE_DEVICE_VECTOR((void**)&(d_epoch), n_obj * sizeof(var_t));
}

void threebody::deallocate_storage()
{
	deallocate_host_storage();
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		deallocate_device_storage();
	}
}

void threebody::deallocate_host_storage()
{
	FREE_HOST_VECTOR((void **)&(h_md));
	FREE_HOST_VECTOR((void **)&(h_epoch));
}

void threebody::deallocate_device_storage()
{
	FREE_DEVICE_VECTOR((void **)&(h_md));
	FREE_DEVICE_VECTOR((void **)&(h_epoch));
}

void threebody::calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy)
{
	if (COMPUTING_DEVICE_CPU == comp_dev)
	{
		cpu_calc_dy(stage, curr_t, y_temp, dy);
	}
	else
	{
		gpu_calc_dy(stage, curr_t, y_temp, dy);
	}
}

void threebody::calc_energy()
{
	const threebody_t::param_t* p = (threebody_t::param_t*)h_p;

	var_t r  = sqrt( SQR(h_y[0]) + SQR(h_y[1]) + SQR(h_y[2]) );
	var_t v2 = SQR(h_y[3]) + SQR(h_y[4]) + SQR(h_y[5]);

	h = 0.5 * v2 - p[0].m / r;
}

void threebody::cpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy)
{
	const threebody_t::param_t* p = (threebody_t::param_t*)h_p;

	/*
	Q1 = y_temp[0], Q2 = y_temp[1], Q3 = y_temp[2], Q4 = y_temp[3],
	Q5 = y_temp[4], Q6 = y_temp[5], Q7 = y_temp[6], Q8 = y_temp[7],
	P1 = y_temp[8], P2 = y_temp[9], P3 = y_temp[10], P4 = y_temp[11],
	P5 = y_temp[12], P6 = y_temp[13], P7 = y_temp[14], P8 = y_temp[15]
	*/

	var_t R1 = SQR(y_temp[0]) + SQR(y_temp[1]) + SQR(y_temp[2]);
	var_t R2 = SQR(y_temp[4]) + SQR(y_temp[5]) + SQR(y_temp[6]);
	var_t R = SQR(y_temp[0] - y_temp[4]) + SQR(y_temp[1] - y_temp[5]) + SQR(y_temp[2] - y_temp[6]);
	var_t mu13 =  (p[0].m * p[2].m) / (p[0].m + p[2].m);
	var_t mu23 =  (p[1].m * p[2].m) / (p[1].m + p[2].m);

	dy[0] = 0.25 / p[2].m * ( y_temp[12] * ( y_temp[0]*y_temp[4] + y_temp[1]*y_temp[5] + y_temp[2]*y_temp[6] ) + y_temp[13] * ( y_temp[1]*y_temp[4] - y_temp[0]*y_temp[5] + y_temp[2]*y_temp[7] ) - y_temp[14] * ( y_temp[0]*y_temp[6] - y_temp[2]*y_temp[4] + y_temp[1]*y_temp[7] ) + y_temp[15] * ( y_temp[0]*y_temp[7] - y_temp[1]*y_temp[6] + y_temp[2]*y_temp[5] ) ) + 0.25 / mu13 * y_temp[8] * R2; 
	dy[1] = 0.25 / p[2].m * ( y_temp[12] * ( y_temp[0]*y_temp[5] - y_temp[1]*y_temp[4] + y_temp[3]*y_temp[6] ) + y_temp[13] * ( y_temp[0]*y_temp[4] + y_temp[1]*y_temp[5] + y_temp[3]*y_temp[7] ) + y_temp[14] * ( y_temp[1]*y_temp[6] - y_temp[0]*y_temp[7] + y_temp[3]*y_temp[4] ) - y_temp[15] * ( y_temp[0]*y_temp[6] + y_temp[1]*y_temp[7] - y_temp[3]*y_temp[5] ) ) + 0.25 / mu13 * y_temp[9] * R2; 
	dy[2] = 0.25 / p[2].m * (-y_temp[12] * ( y_temp[2]*y_temp[4] - y_temp[0]*y_temp[6] + y_temp[3]*y_temp[5] ) + y_temp[13] * ( y_temp[0]*y_temp[7] + y_temp[2]*y_temp[5] - y_temp[3]*y_temp[4] ) + y_temp[14] * ( y_temp[0]*y_temp[4] - y_temp[2]*y_temp[6] + y_temp[3]*y_temp[7] ) + y_temp[15] * ( y_temp[0]*y_temp[5] - y_temp[2]*y_temp[7] + y_temp[3]*y_temp[6] ) ) + 0.25 / mu13 * y_temp[10] * R2;
	dy[3] = 0.25 / p[2].m * ( y_temp[12] * ( y_temp[1]*y_temp[6] - y_temp[2]*y_temp[5] + y_temp[3]*y_temp[4] ) - y_temp[13] * ( y_temp[2]*y_temp[4] - y_temp[1]*y_temp[7] + y_temp[3]*y_temp[5] ) + y_temp[14] * ( y_temp[1]*y_temp[4] + y_temp[2]*y_temp[7] - y_temp[3]*y_temp[6] ) + y_temp[15] * ( y_temp[1]*y_temp[5] + y_temp[2]*y_temp[6] + y_temp[3]*y_temp[7] ) ) + 0.25 / mu13 * y_temp[11] * R2; 

	dy[4] = 0.25 / p[2].m * ( y_temp[4] * ( y_temp[8]*y_temp[0] - y_temp[9]*y_temp[1] - y_temp[10]*y_temp[2] + y_temp[11]*y_temp[3] ) + y_temp[5] * ( y_temp[8]*y_temp[1] + y_temp[9]*y_temp[0] - y_temp[10]*y_temp[3] - y_temp[11]*y_temp[2] ) + y_temp[6] * ( y_temp[8]*y_temp[2] + y_temp[10]*y_temp[0] + y_temp[9]*y_temp[3] + y_temp[11]*y_temp[1] ) ) + 0.25 / mu23 * y_temp[12] * R1; 
	dy[5] = 0.25 / p[2].m * ( y_temp[4] * ( y_temp[8]*y_temp[1] + y_temp[9]*y_temp[0] - y_temp[10]*y_temp[3] - y_temp[11]*y_temp[2] ) - y_temp[5] * ( y_temp[8]*y_temp[0] - y_temp[9]*y_temp[1] - y_temp[10]*y_temp[2] + y_temp[11]*y_temp[3] ) + y_temp[7] * ( y_temp[8]*y_temp[2] + y_temp[10]*y_temp[0] + y_temp[9]*y_temp[3] + y_temp[11]*y_temp[1] ) ) + 0.25 / mu23 * y_temp[13] * R1;
	dy[6] = 0.25 / p[2].m * ( y_temp[4] * ( y_temp[8]*y_temp[2] + y_temp[10]*y_temp[0] + y_temp[9]*y_temp[3] + y_temp[11]*y_temp[1] ) - y_temp[6] * ( y_temp[8]*y_temp[0] - y_temp[9]*y_temp[1] - y_temp[10]*y_temp[2] + y_temp[11]*y_temp[3] ) - y_temp[7] * ( y_temp[8]*y_temp[1] + y_temp[9]*y_temp[0] - y_temp[10]*y_temp[3] - y_temp[11]*y_temp[2] ) ) + 0.25 / mu23 * y_temp[14] * R1;
	dy[7] = 0.25 / p[2].m * ( y_temp[5] * ( y_temp[8]*y_temp[2] + y_temp[10]*y_temp[0] + y_temp[9]*y_temp[3] + y_temp[11]*y_temp[1] ) - y_temp[6] * ( y_temp[8]*y_temp[1] + y_temp[9]*y_temp[0] - y_temp[10]*y_temp[3] - y_temp[11]*y_temp[2] ) + y_temp[7] * ( y_temp[8]*y_temp[0] - y_temp[9]*y_temp[1] - y_temp[10]*y_temp[2] + y_temp[11]*y_temp[3] ) ) + 0.25 / mu23 * y_temp[15] * R1;

	dy[16] = R1*R2;
}

void threebody::gpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy)
{
	throw string("The gpu_calc_dy() is not implemented.");
}

void threebody::load(string& path)
{
	ifstream input;

	cout << "Loading " << path << " ";

	data_rep_t repres = (file::get_extension(path) == "txt" ? DATA_REPRESENTATION_ASCII : DATA_REPRESENTATION_BINARY);
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		input.open(path.c_str());
		if (input) 
		{
			load_ascii(input);
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	case DATA_REPRESENTATION_BINARY:
		input.open(path.c_str(), ios::binary);
		if (input) 
		{
			load_binary(input);
		}
		else 
		{
			throw string("Cannot open " + path + ".");
		}
		break;
	}
	input.close();

	cout << " done" << endl;
}

void threebody::load_ascii(ifstream& input)
{
	threebody_t::param_t* p = (threebody_t::param_t*)h_p;

	for (uint32_t i = 0; i < n_obj; i++)
	{
		load_ascii_record(input, &h_epoch[i], &h_md[i], &p[i], &h_y[i], &h_y[i+3]);
	}
}

void threebody::load_ascii_record(ifstream& input, ttt_t* t, threebody_t::metadata_t *md, threebody_t::param_t* p, var_t* r, var_t* v)
{
	string name;

	// epoch
	input >> *t;
	// name
	input >> name;
	if (name.length() > 30)
	{
		name = name.substr(0, 30);
	}
	obj_names.push_back(name);
	// id
	input >> md->id;
	// mu = k^2*(m1 + m2)
	input >> p->m;

	// position
	var3_t* _r = (var3_t*)r;
	input >> _r->x >> _r->y >> _r->z;
	// velocity
	var3_t* _v = (var3_t*)v;
	input >> _v->x >> _v->y >> _v->z;
}

void threebody::load_binary(ifstream& input)
{
	throw string("The load_binary() is not implemented.");
}

void threebody::print_result(ofstream** sout, data_rep_t repres)
{
	calc_energy();
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		print_result_ascii(*sout[OUTPUT_NAME_DATA]);
		print_integral_data_ascii(*sout[OUTPUT_NAME_INTEGRAL]);
		break;
	case DATA_REPRESENTATION_BINARY:
		print_result_binary(*sout[OUTPUT_NAME_DATA]);
		print_integral_data_binary(*sout[OUTPUT_NAME_INTEGRAL]);
		break;
	}
}

void threebody::print_result_ascii(ofstream& sout)
{
	static uint32_t int_t_w  =  8;
	static uint32_t var_t_w  = 25;

	sout.precision(16);
	sout.setf(ios::right);
	sout.setf(ios::scientific);

	for (uint32_t i = 0; i < n_obj; i++)
    {
		uint32_t orig_idx = h_md[i].id - 1;

		sout << setw(var_t_w) << t << SEP                       /* time of the record [day] (double)           */
			 << setw(     30) << obj_names[orig_idx] << SEP     /* name of the body         (string = 30 char) */ 
		// Print the metadata for each object
        << setw(int_t_w) << h_md[i].id << SEP;

		// Print the parameters for each object
		for (uint16_t j = 0; j < n_ppo; j++)
		{
			uint32_t param_idx = i * n_ppo + j;
			sout << setw(var_t_w) << h_p[param_idx] << SEP;
		}
		// Print the variables for each object
		for (uint16_t j = 0; j < n_vpo; j++)
		{
			uint32_t var_idx = i * n_vpo + j;
			sout << setw(var_t_w) << h_y[var_idx];
			if (j < n_vpo - 1)
			{
				sout << SEP;
			}
			else
			{
				sout << endl;
			}
		}
	}
	sout.flush();
}

void threebody::print_result_binary(ofstream& sout)
{
	throw string("The print_result_binary() is not implemented.");
}

void threebody::print_integral_data(ofstream& sout, data_rep_t repres)
{
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		print_integral_data_ascii(sout);
		break;
	case DATA_REPRESENTATION_BINARY:
		print_integral_data_binary(sout);
		break;
	}
}

void threebody::print_integral_data_ascii(ofstream& sout)
{
	static uint32_t int_t_w  =  8;
	static uint32_t var_t_w  = 25;

	sout.precision(16);
	sout.setf(ios::right);
	sout.setf(ios::scientific);

	sout << setw(var_t_w) << t << SEP                       /* time of the record [day] (double)           */
		 << h << endl;

	sout. flush();
}

void threebody::print_integral_data_binary(ofstream& sout)
{
	throw string("The print_integral_data_binary() is not implemented.");
}
