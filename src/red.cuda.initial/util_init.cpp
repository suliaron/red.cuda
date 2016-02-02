// includes, system 
#include <iomanip>      // setw()
#include <cmath>
#include <fstream>		// ofstream
#include <string>		// string
#include <sstream>		// ostringstream
#include <stdlib.h>		// atof()

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

uint32_t calc_number_of_bodies(body_disk &bd)
{
	uint32_t result = 0;
	for (uint16_t body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_N; body_type++)
	{
		result += bd.nBody[body_type];
	}
	return result;
}

uint32_t calc_number_of_bodies(body_disk &bd, body_type_t bt)
{
	uint32_t result = 0;
	for (uint16_t body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_N; body_type++)
	{
		if (bt == body_type)
		{
			result += bd.nBody[body_type];
		}
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

void generate_pp(phys_prop_dist_t *pp_d, pp_disk_t::param_t& param)
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

void print_uint32_t(string &path, uint32_t n)
{
	printf("Writing %s to disk .", path.c_str());

	ofstream sout(path.c_str(), ios_base::out);
	if (sout)
	{
		sout << n;
		sout.close();
		printf(" done\n");
	}
	else
	{
		throw string("Cannot open " + path + "!");
	}
}

template <typename T>
void print_number(string& path, T number)
{
	printf("Writing %s to disk .", path.c_str());

	ofstream sout(path.c_str(), ios_base::out);
	if (sout)
	{
		sout << number;
		sout.close();
		printf(" done\n");
	}
	else
	{
		throw string("Cannot open " + path + "!");
	}
}
template void print_number<char>(         string& path, char number);
template void print_number<unsigned char>(string& path, unsigned char number);
template void print_number<int>(          string& path, int number);
template void print_number<uint32_t>(     string& path, uint32_t number);
template void print_number<long>(         string& path, long number);
template void print_number<unsigned long>(string& path, unsigned long number);

void print_data(string &path, body_disk_t& disk, pp_disk_t::sim_data_t* sd, input_format_name_t format)
{
	printf("Writing %s to disk .", path.c_str());

	ofstream sout(path.c_str(), ios_base::out);
	if (sout)
	{
		uint32_t n_body = calc_number_of_bodies(disk);
		if (INPUT_FORMAT_RED == format)
		{
			;
		}
		if (INPUT_FORMAT_NONAME == format)
		{
			sout << "Time: " << sd->h_epoch[0] << endl;
			sout << "Number of bodies: " << n_body << endl;
			sout << "Coordinate system: barycentric" << endl;
			sout << endl;
			sout << "id                       name                     type                     cx [AU]                  cy [AU]                  cz [AU]                  vx [AU/day]              vy [AU/day]              vz [AU/day]              mass [sol]               radius [AU]" << endl;
			sout << endl;
		}
		if (INPUT_FORMAT_HIPERION == format)
		{
			sout << "# HEADER_BEGIN" << endl;
			sout << "# CountSnapshot   = 0" << endl;
			sout << "# Iter            = 0" << endl;
			sout << "# Time            = 0" << endl;
			sout << "# NewDt           = 0" << endl;
			sout << "# Ntot            = " << n_body << endl;
			sout << "# Ngrav           = " << disk.nBody[BODY_TYPE_STAR] + disk.nBody[BODY_TYPE_GIANTPLANET] + disk.nBody[BODY_TYPE_ROCKYPLANET] + disk.nBody[BODY_TYPE_PROTOPLANET] << endl;
			sout << "# Nst             = " << disk.nBody[BODY_TYPE_STAR] << endl;
			sout << "# Npl             = " << disk.nBody[BODY_TYPE_GIANTPLANET] + disk.nBody[BODY_TYPE_ROCKYPLANET] << endl;
			sout << "# Nplms           = " << disk.nBody[BODY_TYPE_PROTOPLANET] << endl;
			sout << "# Ndust           = " << disk.nBody[BODY_TYPE_PLANETESIMAL] + disk.nBody[BODY_TYPE_TESTPARTICLE] << endl;
			sout << "# NCstr           = 0" << endl;
			sout << "# NCpl            = 0" << endl;
			sout << "# NCplms          = 0" << endl;
			sout << "# Ekin0           = -1.000000" << endl;
			sout << "# Epot0           = -1.000000" << endl;
			sout << "# Eloss           = 0.000000" << endl;
			sout << "# dE              = 0" << endl;
			sout << "# OrbitalPeriod   = 6.2831854820251465e+00" << endl;
			sout << "# TimeCPU         = 0"  << endl;
			sout << "# HEADER_END" << endl;
		}

		int pcd = 1;
		for (uint32_t i = 0; i < n_body; i++)
		{
			if (pcd <= (int)((((var_t)i/(var_t)n_body))*100.0))
			{
				printf(".");
				pcd++;
			}
			switch (format)
			{
			case INPUT_FORMAT_RED:
				file::print_body_record_ascii_RED(     sout, disk.names[i], &sd->h_p[i], &sd->h_body_md[i], &sd->h_y[0][i], &sd->h_y[1][i]);
				break;
			case INPUT_FORMAT_NONAME:
				file::print_body_record_Emese(   sout, disk.names[i], sd->h_epoch[i], &sd->h_p[i], &sd->h_body_md[i], &sd->h_y[0][i], &sd->h_y[1][i]);
				break;
			case INPUT_FORMAT_HIPERION:
				file::print_body_record_HIPERION(sout, disk.names[i], sd->h_epoch[i], &sd->h_p[i], &sd->h_body_md[i], &sd->h_y[0][i], &sd->h_y[1][i]);
				break;
			default:
				throw string("Invalid format in print_data().");
				break;
			}
		}
		sout.close();
		printf(" done\n");
	}
	else
	{
		throw string("Cannot open " + path + "!");
	}
}

void print_data_info(string &path, ttt_t t, ttt_t dt, body_disk_t& disk, input_format_name_t format)
{
	printf("Writing %s to disk .", path.c_str());

	ofstream sout(path.c_str(), ios_base::out);
	if (sout)
	{
		n_objects_t *n_bodies = new n_objects_t(disk.nBody[BODY_TYPE_STAR], disk.nBody[BODY_TYPE_GIANTPLANET], disk.nBody[BODY_TYPE_ROCKYPLANET], disk.nBody[BODY_TYPE_PROTOPLANET], disk.nBody[BODY_TYPE_SUPERPLANETESIMAL], disk.nBody[BODY_TYPE_PLANETESIMAL], disk.nBody[BODY_TYPE_TESTPARTICLE]);
		if (INPUT_FORMAT_RED == format)
		{
			file::print_data_info_record_ascii_RED(sout, t, dt, n_bodies);
		}
		if (INPUT_FORMAT_NONAME == format)
		{
			;
		}
		if (INPUT_FORMAT_HIPERION == format)
		{
			;
		}
		sout.close();
		printf(" done\n");
	}
	else
	{
		throw string("Cannot open " + path + "!");
	}
}

void print_oe(string &path, uint32_t n, ttt_t t, pp_disk_t::sim_data_t *sd)
{
	printf("Writing %s to disk .", path.c_str());

	ofstream sout(path.c_str(), ios_base::out);
	if (sout)
	{		
		int pcd = 1;
		for (uint32_t i = 0; i < n; i++)
		{
			file::print_oe_record(sout, t, &sd->h_oe[i], &sd->h_p[i], &sd->h_body_md[i]);
			if (pcd <= (int)((((var_t)(i+1)/(var_t)n))*100.0))
			{
				printf(".");
				pcd++;
			}
		}
		sout.close();
		printf(" done\n");
	}
	else
	{
		throw string("Cannot open " + path + "!");
	}
}

/*
    JDCT     Epoch Julian Date, Coordinate Time
      EC     Eccentricity, e                                                   
      QR     Periapsis distance, q (AU)                                        
      IN     Inclination w.r.t xy-plane, i (degrees)                           
      OM     Longitude of Ascending Node, OMEGA, (degrees)                     
      W      Argument of Perifocus, w (degrees)                                
      Tp     Time of periapsis (Julian day number)                             
      N      Mean motion, n (degrees/day)                                      
      MA     Mean anomaly, M (degrees)                                         
      TA     True anomaly, nu (degrees)                                        
      A      Semi-major axis, a (AU)                                           
      AD     Apoapsis distance (AU)                                            
      PR     Sidereal orbit period (day)
JDCT ,   ,                                          EC,                     QR,                     IN,                     OM,                     W,                      Tp,                     N,                      MA,                     TA,                     A,                      AD,                     PR
2457153.500000000, A.D. 2015-May-11 00:00:00.0000,  2.056283170656704E-01,  3.075005270000471E-01,  7.004033081870772E+00,  4.831135275416084E+01,  2.917018579009503E+01,  2.457132325703553E+06,  4.092332803985572E+00,  8.665226794919347E+01,  1.098801777958040E+02,  3.870990540148295E-01,  4.666975810296119E-01,  8.796938500441402E+01,
*/
ttt_t extract_from_horizon_output(string &data, orbelem_t& oe)
{
	int start = 0;
	int len = start + 17;
	string s = data.substr(start, len);

	ttt_t epoch = atof(s.c_str());

	start = 52;
	len = 21;
	s = data.substr(start, len);
	oe.ecc = atof(s.c_str());

	start = 100;
	len = 21;
	s = data.substr(start, len);
	oe.inc = atof(s.c_str()) * constants::DegreeToRadian;

	start = 124;
	len = 21;
	s = data.substr(start, len);
	oe.node = atof(s.c_str()) * constants::DegreeToRadian;

	start = 148;
	len = 21;
	s = data.substr(start, len);
	oe.peri = atof(s.c_str()) * constants::DegreeToRadian;

	start = 220;
	len = 21;
	s = data.substr(start, len);
	oe.mean = atof(s.c_str()) * constants::DegreeToRadian;

	start = 268;
	len = 21;
	s = data.substr(start, len);
	oe.sma = atof(s.c_str());

	return epoch;
}