// includes, system 
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include <stdint.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>

// includes, project
#include "distribution.h"
#include "nebula.h"
#include "red_constants.h"
#include "red_type.h"
#include "redutilcu.h"
#include "util_init.h"

using namespace std;
using namespace redutilcu;

int generate_pp_disk(const string &path, body_disk_t& body_disk)
{
	ofstream	output(path.c_str(), ios_base::out);
	if (output)
	{
		for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_PADDINGPARTICLE; body_type++)
		{
			output << body_disk.nBody[body_type] << SEP;
		}
		output << endl;

		var_t	t = 0.0;
		vec_t	rVec = {0.0, 0.0, 0.0, 0.0};
		vec_t	vVec = {0.0, 0.0, 0.0, 0.0};

        ttt_t           epoch = 0.0;
		param_t	        param0, param;
        body_metadata_t body_md;

        orbelem_t	oe;

        // The id of each body must be larger than 0 in order to indicate inactive body with negative id (ie. zero is not good)
        int bodyIdx = 0;
		int bodyId = 1;
		for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_N; body_type++)
		{
			for (int i = 0; i < body_disk.nBody[body_type]; i++, bodyIdx++, bodyId++)
			{
    			epoch = 0.0;
				if (body_type == BODY_TYPE_STAR)
				{
					body_md.id = bodyId;
					body_md.body_type = BODY_TYPE_STAR;

					generate_pp(&body_disk.pp_d[body_type], param0);
					body_md.mig_type = body_disk.mig_type[bodyIdx];
					body_md.mig_stop_at = body_disk.stop_at[bodyIdx];

printf("Printing body %s to file ... ", body_disk.names[bodyIdx].c_str());
					file::print_body_record(output, body_disk.names[bodyIdx], epoch, &param0, &body_md, &rVec, &vVec);
printf("done\r");
				} /* if */
				else 
				{
					body_md.id = bodyId;
					body_md.body_type = static_cast<body_type_t>(body_type);

					generate_oe(&body_disk.oe_d[body_type], oe);
					generate_pp(&body_disk.pp_d[body_type], param);
					body_md.mig_type = body_disk.mig_type[bodyIdx];
					body_md.mig_stop_at = body_disk.stop_at[bodyIdx];

					var_t mu = K2*(param0.mass + param.mass);
					int_t ret_code = tools::calculate_phase(mu, &oe, &rVec, &vVec);
					if (ret_code == 1) {
						cerr << "Could not calculate the phase." << endl;
						return ret_code;
					}

printf("Printing body %s to file ... ", body_disk.names[bodyIdx].c_str());
					file::print_body_record(output, body_disk.names[bodyIdx], t, &param, &body_md, &rVec, &vVec);
printf("done\r");
				} /* else */
			} /* for */
		} /* for */
		output.flush();
		output.close();
	}
	else
	{
		cerr << "Cannot open " << path << ".";
		exit(0);
	}

	return 0;
}

void set_parameters_of_Two_body_disk(body_disk_t& disk)
{
	const var_t m1 = 1.0 /* Jupiter */ * constants::JupiterToSolar;
	const var_t rho1 = 1.3 * constants::GramPerCm3ToSolarPerAu3;

	initialize(disk);

	disk.nBody[BODY_TYPE_STAR       ] = 1;
	disk.nBody[BODY_TYPE_GIANTPLANET] = 1;

	int_t nBodies = calculate_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[nBodies];
	disk.stop_at = new var_t[nBodies];

    int bodyIdx = 0;
	int type = BODY_TYPE_STAR;

	disk.names.push_back("star");

	disk.pp_d[type].item[MASS]       = new uniform_distribution(1, 1.0, 1.0);

	var_t tmp = constants::SolarRadiusToAu;
	disk.pp_d[type].item[RADIUS]     = new uniform_distribution(3, tmp, tmp);

	disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(5, 0.0, 0.0);

	disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
	disk.stop_at[bodyIdx] = 0.0;

	type = BODY_TYPE_GIANTPLANET;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new uniform_distribution(4, 1.0, 1.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new uniform_distribution(4, 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(4, 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(4, 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(4, 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(4, 0.0, 0.0);

		disk.pp_d[type].item[MASS      ] = new uniform_distribution(4, m1, m1);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(4, rho1, rho1);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(4, 0.0, 0.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}
}

void set_parameters_of_Dvorak_disk(nebula& n, body_disk_t& disk)
{
	const var_t rhoBasalt = 2.7 /* g/cm^3 */ * constants::GramPerCm3ToSolarPerAu3;

	disk.nBody[BODY_TYPE_STAR        ] = 1;
	disk.nBody[BODY_TYPE_PROTOPLANET ] = 2000;

	int_t nBodies = calculate_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[nBodies];
	disk.stop_at  = new var_t[nBodies];

    int bodyIdx = 0;
	int type = BODY_TYPE_STAR;

	disk.names.push_back("star");
	disk.pp_d[type].item[MASS]       = new uniform_distribution(1, 1.0, 1.0);
	disk.pp_d[type].item[RADIUS]     = new uniform_distribution(3, constants::SolarRadiusToAu, constants::SolarRadiusToAu);
	disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(5, 0.0, 0.0);

	disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
	disk.stop_at[bodyIdx] = 0.0;

	type = BODY_TYPE_PROTOPLANET;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new power_law_distribution(7, n.solid_c.get_r_1(), n.solid_c.get_r_2(), n.solid_c.get_p());

		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new rayleigh_distribution(11, 0.01);
		disk.oe_d[type].range[ORBITAL_ELEMENT_ECC ].x = 0.0;
		disk.oe_d[type].range[ORBITAL_ELEMENT_ECC ].y = 0.2;

		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(13, 0.0, 0.0);

		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(17, 0.0, 2.0 * PI);

		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(19, 0.0, 0.0);

		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(23, 0.0, 2.0 * PI);

		disk.pp_d[type].item[MASS      ] = new lognormal_distribution(27, 0.1, 3.0, 0.0, 0.25);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(35, rhoBasalt, rhoBasalt);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(1, 0.0, 0.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i + 1, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}
}

void populate_Dvorak_disk(body_disk_t& disk, sim_data_t *sd)
{
    ttt_t           epoch = 0.0;
	param_t	        param;
    body_metadata_t body_md;

    orbelem_t oe = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // The id of each body must be larger than 0 in order to indicate inactive body with negative id (ie. zero is not good)
    int bodyIdx = 0;
	int bodyId = 1;
	for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_N; body_type++)
	{
		for (int i = 0; i < disk.nBody[body_type]; i++, bodyIdx++, bodyId++)
		{
			body_md.id = bodyId;
			body_md.body_type = static_cast<body_type_t>(body_type);
			body_md.mig_type = disk.mig_type[bodyIdx];
			body_md.mig_stop_at = disk.stop_at[bodyIdx];

			generate_pp(&disk.pp_d[body_type], param);

			if (BODY_TYPE_STAR == body_type)
			{
				oe.sma = oe.ecc = oe.inc = oe.peri = oe.node = oe.mean = 0.0;
			}
			else
			{
				generate_oe(&disk.oe_d[body_type], oe);
			}

			sd->epoch[  bodyIdx] = epoch;
			sd->body_md[bodyIdx] = body_md;
			sd->p[      bodyIdx] = param;
			sd->h_oe[   bodyIdx] = oe;
		} /* for */
	} /* for */
}

void create_Dvorak_disk(string dir, string filename, body_disk_t& disk)
{
	initialize(disk);

	// Create a MMSN with gas component and solids component
	var_t r_1  =  0.025;  /* inner rim of the disk [AU] */
	var_t r_2  = 33.0;    /* outer rim of the disk [AU] */
	var_t r_SL =  2.7;    /* distance of the snowline [AU] */
	var_t f_neb = 3.0;
	var_t f_ice = 4.2;    /* ice condensation coefficient beyond the snowline */
	var_t Sigma_1 = 7.0;  /* Surface density of solids at r = 1 AU */
	var_t f_gas = 240.0;  /* gas to dust ratio */
	var_t p = -3.0/2.0;   /* profile index of the power-law function */

	Sigma_1 *= constants::GramPerCm2ToSolarPerAu2;
	gas_component gas_c(r_1, r_2, r_SL, f_neb, Sigma_1, f_gas, p);

	r_1 = 0.9;
	r_2 = 2.7;
	solid_component solid_c(r_1, r_2, r_SL, f_neb, Sigma_1, f_ice, p);
	nebula mmsn(gas_c, solid_c);

	var_t m_gas   = mmsn.gas_c.calc_mass();
	var_t m_solid = mmsn.solid_c.calc_mass();

	set_parameters_of_Dvorak_disk(mmsn, disk);

	sim_data_t* sim_data = new sim_data_t;
	int nBodies = calculate_number_of_bodies(disk);
	allocate_host_storage(sim_data, nBodies);

	populate_Dvorak_disk(disk, sim_data);

	// Scale the masses in order to get the required mass transform_mass()
	{
		var_t m_total_pp = tools::get_total_mass(nBodies, BODY_TYPE_PROTOPLANET, sim_data);
		var_t f = m_solid / m_total_pp;
		for (int i = 0; i < nBodies; i++)
		{
			// Only the masses of the protoplanets will be scaled
			if (sim_data->body_md[i].body_type == BODY_TYPE_PROTOPLANET)
			{
				sim_data->p[i].mass *= f;
			}
		}
		m_total_pp = tools::get_total_mass(nBodies, BODY_TYPE_PROTOPLANET, sim_data);
		if (fabs(m_total_pp - m_solid) > 1.0e-15)
		{
			cerr << "The required mass was not reached." << endl;
			exit(0);
		}
	}

	// Computes the physical quantities with the new mass
	{
		int bodyIdx = 0;
		for (int type = BODY_TYPE_STAR; type < BODY_TYPE_N; type++)
		{
			for (int i = 0; i < disk.nBody[type]; i++, bodyIdx++)
			{
				if (sim_data->p[bodyIdx].mass > 0.0)
				{
					if (disk.pp_d[type].item[DENSITY] == 0x0 && disk.pp_d[type].item[RADIUS] != 0x0)
					{
						sim_data->p[bodyIdx].density = tools::calculate_density(sim_data->p[bodyIdx].mass, sim_data->p[bodyIdx].radius);
					}
					if (disk.pp_d[type].item[RADIUS] == 0x0 && disk.pp_d[type].item[DENSITY] != 0x0)
					{
						sim_data->p[bodyIdx].radius = tools::calculate_radius(sim_data->p[bodyIdx].mass, sim_data->p[bodyIdx].density);
					}
				}
			}
		}
	}

	// Calculate coordinates and velocities
	{
		// The mass of the central star
		var_t m0 = sim_data->p[0].mass;
		vec_t rVec = {0.0, 0.0, 0.0, 0.0};
		vec_t vVec = {0.0, 0.0, 0.0, 0.0};

		// The coordinates of the central star
		sim_data->y[0][0] = rVec;
		sim_data->y[1][0] = vVec;
		for (int i = 1; i < nBodies; i++)
		{
			var_t mu = K2 *(m0 + sim_data->p[i].mass);
			int ret_code = tools::calculate_phase(mu, &sim_data->h_oe[i], &rVec, &vVec);
			if (1 == ret_code)
			{
				cerr << "Could not calculate the phase." << endl;
				exit(0);
			}
			sim_data->y[0][i] = rVec;
			sim_data->y[1][i] = vVec;
		}
	}

	tools::transform_to_bc(nBodies, sim_data);

	string path = file::combine_path(dir, filename) + ".oe.txt";
	print(path, nBodies, sim_data);

	path = file::combine_path(dir, filename) + ".txt";
	print(path, disk, sim_data);

	deallocate_host_storage(sim_data);

	delete sim_data;
}

int parse_options(int argc, const char **argv, string &outDir, string &filename)
{
	int i = 1;

	while (i < argc) {
		string p = argv[i];

		if (     p == "-o") {
			i++;
			outDir = argv[i];
		}
		else if (p == "-f") {
			i++;
			filename = argv[i];
		}
		else {
			cerr << "Invalid switch on command-line.";
			return 1;
		}
		i++;
	}

	return 0;
}

//-o D:\Work\Projects\solaris.cuda\TestRun\Dvorak_disk -f Dvorak_disk.txt
//-o C:\Work\Projects\red.cuda\TestRun\InputTest\Release\TwoBody -f TwoBody.txt
//-o C:\Work\Projects\red.cuda.TestRun\Emese_Dvorak -f collision-testdata-N10001-vecelem-binary.dat
//-o C:\Work\Projects\red.cuda\TestRun\InputTest\Release\N1_massive_N3_test -f N1_massive_N3_test.txt
//-o C:\Work\Projects\red.cuda\TestRun\DvorakDisk\Run_cf_5 -f Run_cf_5.txt
int main(int argc, const char **argv)
{
	body_disk_t disk;
	string outDir;
	string filename;
	string output_path;

	parse_options(argc, argv, outDir, filename);

	//{
	//	string input_path = file::combine_path(outDir, filename);
	//	string output_path = file::combine_path(outDir, file::get_filename_without_ext(filename) + ".txt");
	//	Emese_data_format_to_red_cuda_format(input_path, output_path);
	//	return 0;
	//}

	//{
	//	_set_parameters_of_Two_body_disk(disk);
	//	output_path = file::combine_path(outDir, filename);
	//	generate_pp_disk(output_path, disk);
	//	return 0;
	//}

	{
		create_Dvorak_disk(outDir, filename, disk);
		return 0;
	}

	//{
	//	set_parameters_of_N1_massive_N3_test(disk);
	//	output_path = file::combine_path(outDir, filename);
	//	generate_pp_disk(output_path, disk);
    //	return 0;
	//}

	return 0;
}
