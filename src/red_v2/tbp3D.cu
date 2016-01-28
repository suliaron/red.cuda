#include <iostream>
#include <iomanip>
#include <fstream>

#include "tbp3D.h"

#include "redutilcu.h"
#include "red_constants.h"

using namespace std;
using namespace redutilcu;


tbp3D::tbp3D(uint16_t n_ppo, computing_device_t comp_dev) :
	ode(3, 6, n_ppo, 1, comp_dev)
{
	initialize();
	allocate_storage();
}

tbp3D::~tbp3D()
{
	deallocate_storage();
}

void tbp3D::initialize()
{
	h_md    = 0x0;
	h_epoch = 0x0;

	h       = 0.0;
}

void tbp3D::allocate_storage()
{
	allocate_host_storage();
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		allocate_device_storage();
	}
}

void tbp3D::allocate_host_storage()
{
	ALLOCATE_HOST_VECTOR((void**)&(h_md),    n_obj * sizeof(tbp3D_t::metadata_t));
	ALLOCATE_HOST_VECTOR((void**)&(h_epoch), n_obj * sizeof(var_t));
}

void tbp3D::allocate_device_storage()
{
	ALLOCATE_DEVICE_VECTOR((void**)&(d_md),    n_obj * sizeof(tbp3D_t::metadata_t));
	ALLOCATE_DEVICE_VECTOR((void**)&(d_epoch), n_obj * sizeof(var_t));
}

void tbp3D::deallocate_storage()
{
	deallocate_host_storage();
	if (COMPUTING_DEVICE_GPU == comp_dev)
	{
		deallocate_device_storage();
	}
}

void tbp3D::deallocate_host_storage()
{
	FREE_HOST_VECTOR((void **)&(h_md));
	FREE_HOST_VECTOR((void **)&(h_epoch));
}

void tbp3D::deallocate_device_storage()
{
	FREE_DEVICE_VECTOR((void **)&(h_md));
	FREE_DEVICE_VECTOR((void **)&(h_epoch));
}

void tbp3D::calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy)
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

void tbp3D::calc_energy()
{
	const tbp3D_t::param_t* p = (tbp3D_t::param_t*)h_p;

	var_t r  = sqrt( SQR(h_y[0]) + SQR(h_y[1]) + SQR(h_y[2]) );
	var_t v2 = SQR(h_y[3]) + SQR(h_y[4]) + SQR(h_y[5]);

	h = 0.5 * v2 - p[0].mu / r;
}

void tbp3D::cpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy)
{
	const tbp3D_t::param_t* p = (tbp3D_t::param_t*)h_p;

	var_t r = sqrt( SQR(y_temp[0]) + SQR(y_temp[1]) + SQR(y_temp[2]) );
	var_t r3 = r*r*r;

	dy[0] = y_temp[3];                    // dx1 / dt = x4
	dy[1] = y_temp[4];                    // dx2 / dt = x5
	dy[2] = y_temp[5];                    // dx3 / dt = x6

	dy[3] = -p[0].mu / r3 * y_temp[0];    // dx4 / dt = -(mu/r^3) * x1
	dy[4] = -p[0].mu / r3 * y_temp[1];    // dx5 / dt = -(mu/r^3) * x2
	dy[5] = -p[0].mu / r3 * y_temp[2];    // dx6 / dt = -(mu/r^3) * x3
}

void tbp3D::gpu_calc_dy(uint16_t stage, ttt_t curr_t, const var_t* y_temp, var_t* dy)
{
	throw string("The gpu_calc_dy() is not implemented.");
}

void tbp3D::load(string& path)
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

void tbp3D::load_ascii(ifstream& input)
{
	tbp3D_t::param_t* p = (tbp3D_t::param_t*)h_p;

	for (uint32_t i = 0; i < n_obj; i++)
	{
		load_ascii_record(input, &h_epoch[i], &h_md[i], &p[i], &h_y[i], &h_y[i+3]);
	}
}

void tbp3D::load_ascii_record(ifstream& input, ttt_t* t, tbp3D_t::metadata_t *md, tbp3D_t::param_t* p, var_t* r, var_t* v)
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
	input >> p->mu;

	// position
	var3_t* _r = (var3_t*)r;
	input >> _r->x >> _r->y >> _r->z;
	// velocity
	var3_t* _v = (var3_t*)v;
	input >> _v->x >> _v->y >> _v->z;
}

void tbp3D::load_binary(ifstream& input)
{
	throw string("The load_binary() is not implemented.");
}

void tbp3D::print_result(ofstream& sout, data_rep_t repres)
{
	switch (repres)
	{
	case DATA_REPRESENTATION_ASCII:
		print_result_ascii(sout);
		break;
	case DATA_REPRESENTATION_BINARY:
		print_result_binary(sout);
		break;
	}
}

void tbp3D::print_result_ascii(ofstream& sout)
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

void tbp3D::print_result_binary(ofstream& sout)
{
	throw string("The print_result_binary() is not implemented.");
}

void tbp3D::print_integral_data(ofstream& sout, data_rep_t repres)
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

void tbp3D::print_integral_data_ascii(ofstream& sout)
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

void tbp3D::print_integral_data_binary(ofstream& sout)
{
	throw string("The print_integral_data_binary() is not implemented.");
}

