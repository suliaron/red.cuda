// includes, system 
#include <cmath>
#include <fstream>		// ofstream
#include <string>		// string
#include <sstream>		// ostringstream

// includes, project
#include "util_init.h"
#include "distribution.h"
#include "red_constants.h"
#include "redutilcu.h"

using namespace std;
using namespace redutilcu;

string create_name(int i, int type)
{
	static string body_type_names[] = {"star", "giant", "rocky", "proto", "superpl", "pl", "testp"}; 

	ostringstream convert;	// stream used for the conversion
	string i_str;			// string which will contain the number
	string name;

	convert << i;			// insert the textual representation of 'i' in the characters in the stream
	i_str = convert.str();  // set 'i_str' to the contents of the stream
	name = body_type_names[type] + i_str;

	return name;
}

void initialize(body_disk &disk)
{
	for (int type = BODY_TYPE_STAR; type < BODY_TYPE_N; type++)
	{
		disk.nBody[type] = 0;
        for (int i = 0; i < ORBITAL_ELEMENT_N; i++) 
		{
			disk.oe_d[type].range[i].x = disk.oe_d[type].range[i].y = 0.0;
			disk.oe_d[type].item[i] = 0x0;
		}
		for (int i = 0; i < 4; i++) 
		{
			disk.pp_d[type].range[i].x = disk.pp_d[type].range[i].y = 0.0;
			disk.pp_d[type].item[i] = 0x0;
		}
	}
	disk.mig_type = 0x0;
	disk.stop_at  = 0x0;
}

int	calculate_number_of_bodies(body_disk &bd)
{
	int_t result = 0;
	for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_N; body_type++)
	{
		result += bd.nBody[body_type];
	}
	return result;
}

void generate_oe(orbelem_name_t name, oe_dist_t *oe_d, var_t* oe)
{
	var_t x_min = oe_d->range[name].x;
	var_t x_max = oe_d->range[name].y;
	var_t dx = fabs(x_max - x_min);
	do
	{
		*oe = oe_d->item[name]->get_next();
	} while (dx > 0.0 && x_min > *oe && x_max < *oe);
}

void generate_oe(oe_dist_t *oe_d, orbelem_t& oe)
{
	if (0x0 != oe_d->item[ORBITAL_ELEMENT_SMA])
	{
		generate_oe(ORBITAL_ELEMENT_SMA,  oe_d, &oe.sma);
	}
	else
	{
		oe.sma = 0.0;
	}
	if (0x0 != oe_d->item[ORBITAL_ELEMENT_ECC])
	{
		generate_oe(ORBITAL_ELEMENT_ECC,  oe_d, &oe.ecc);
	}
	else
	{
		oe.ecc = 0.0;
	}
	if (0x0 != oe_d->item[ORBITAL_ELEMENT_INC])
	{
		generate_oe(ORBITAL_ELEMENT_INC,  oe_d, &oe.inc);
	}
	else
	{
		oe.inc = 0.0;
	}
	if (0x0 != oe_d->item[ORBITAL_ELEMENT_PERI])
	{
		generate_oe(ORBITAL_ELEMENT_PERI, oe_d, &oe.peri);
	}
	else
	{
		oe.peri = 0.0;
	}
	if (0x0 != oe_d->item[ORBITAL_ELEMENT_NODE])
	{
		generate_oe(ORBITAL_ELEMENT_NODE, oe_d, &oe.node);
	}
	else
	{
		oe.node = 0.0;
	}
	if (0x0 != oe_d->item[ORBITAL_ELEMENT_MEAN])
	{
		generate_oe(ORBITAL_ELEMENT_MEAN, oe_d, &oe.mean);
	}
	else
	{
		oe.mean = 0.0;
	}
}

void generate_pp(phys_prop_name_t name, phys_prop_dist_t* pp_d, var_t* value)
{
	var_t x_min = pp_d->range[name].x;
	var_t x_max = pp_d->range[name].y;
	var_t dx = fabs(x_max - x_min);
	do
	{
		*value = pp_d->item[name]->get_next();
	} while (dx > 0.0 && x_min > *value && x_max < *value);
}

void generate_pp(phys_prop_dist_t *pp_d, param_t& param)
{
	param.mass = param.radius = param.density = param.cd = 0.0;

	if (0x0 != pp_d->item[MASS])
	{
		generate_pp(MASS, pp_d, &param.mass);
	}
	if (0x0 != pp_d->item[RADIUS])
	{
		generate_pp(RADIUS, pp_d, &param.radius);
	}
	if (0x0 != pp_d->item[DENSITY])
	{
		generate_pp(DENSITY, pp_d, &param.density);
	}
	if (0x0 != pp_d->item[DRAG_COEFF])
	{
		generate_pp(DRAG_COEFF, pp_d, &param.cd);
	}
}

void print(string &path, body_disk_t& disk, sim_data_t* sd, input_format_name_t format)
{
	ofstream	output(path.c_str(), ios_base::out);
	if (output)
	{
		int_t nBodies = calculate_number_of_bodies(disk);

		if (INPUT_FORMAT_RED == format)
		{
			for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_PADDINGPARTICLE; body_type++)
			{
				output << disk.nBody[body_type] << SEP;
			}
			output << endl;
		}
		if (INPUT_FORMAT_NONAME == format)
		{
			output << "Time: " << sd->h_epoch[0] << endl;
			output << "Number of bodies: " << nBodies << endl;
			output << "Coordinate system: barycentric" << endl;
			output << endl;
			output << "id                       name                     type                     cx [AU]                  cy [AU]                  cz [AU]                  vx [AU/day]              vy [AU/day]              vz [AU/day]              mass [sol]               radius [AU]" << endl;
			output << endl;
		}
		if (INPUT_FORMAT_HIPERION == format)
		{
			output << "# HEADER_BEGIN" << endl;
			output << "# CountSnapshot   = 0" << endl;
			output << "# Iter            = 0" << endl;
			output << "# Time            = 0" << endl;
			output << "# NewDt           = 0" << endl;
			output << "# Ntot            = " << nBodies << endl;
			output << "# Ngrav           = " << disk.nBody[BODY_TYPE_STAR] + disk.nBody[BODY_TYPE_GIANTPLANET] + disk.nBody[BODY_TYPE_ROCKYPLANET] + disk.nBody[BODY_TYPE_PROTOPLANET] << endl;
			output << "# Nst             = " << disk.nBody[BODY_TYPE_STAR] << endl;
			output << "# Npl             = " << disk.nBody[BODY_TYPE_GIANTPLANET] + disk.nBody[BODY_TYPE_ROCKYPLANET] << endl;
			output << "# Nplms           = " << disk.nBody[BODY_TYPE_PROTOPLANET] << endl;
			output << "# Ndust           = " << disk.nBody[BODY_TYPE_PLANETESIMAL] + disk.nBody[BODY_TYPE_TESTPARTICLE] << endl;
			output << "# NCstr           = 0" << endl;
			output << "# NCpl            = 0" << endl;
			output << "# NCplms          = 0" << endl;
			output << "# Ekin0           = -1.000000" << endl;
			output << "# Epot0           = -1.000000" << endl;
			output << "# Eloss           = 0.000000" << endl;
			output << "# dE              = 0" << endl;
			output << "# OrbitalPeriod   = 6.2831854820251465e+00" << endl;
			output << "# TimeCPU         = 0"  << endl;
			output << "# HEADER_END" << endl;
		}

		for (int i = 0; i < nBodies; i++)
		{
			switch (format)
			{
			case INPUT_FORMAT_RED:
				file::print_body_record(         output, disk.names[i], sd->h_epoch[i], &sd->h_p[i], &sd->h_body_md[i], &sd->h_y[0][i], &sd->h_y[1][i]);
				break;
			case INPUT_FORMAT_NONAME:
				file::print_body_record_Emese(   output, disk.names[i], sd->h_epoch[i], &sd->h_p[i], &sd->h_body_md[i], &sd->h_y[0][i], &sd->h_y[1][i]);
				break;
			case INPUT_FORMAT_HIPERION:
				file::print_body_record_HIPERION(output, disk.names[i], sd->h_epoch[i], &sd->h_p[i], &sd->h_body_md[i], &sd->h_y[0][i], &sd->h_y[1][i]);
				break;
			default:
				throw string("Invalid format in print().");
				break;
			}
		}
		output.flush();
		output.close();
	}
	else
	{
		throw string("Cannot open " + path + "!");
	}
}

void print(string &path, int n, sim_data_t *sd)
{
	ofstream	output(path.c_str(), ios_base::out);
	if (output)
	{		
		for (int i = 0; i < n; i++)
		{
			file::print_oe_record(output, &sd->h_oe[i]);
		}
		output.close();
	}
	else
	{
		throw string("Cannot open " + path + "!");
	}
}
