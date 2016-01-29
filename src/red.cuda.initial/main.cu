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
#include "red.cuda.initial.type.h"

#if defined(__WIN32__) || defined(_WIN32) || defined(WIN32) || defined(__WINDOWS__) || defined(__TOS_WIN__)

#include <windows.h>

using namespace std;
using namespace redutilcu;

typedef unsigned long ulong;

inline void delay(ulong ms)
{
	Sleep( ms );
}
#else  /* presume POSIX */

#include <unistd.h>

inline void delay(ulong ms)
{
	usleep( ms * 1000 );
}
#endif 


namespace ephemeris_major_planets
{
/*
	Symbol meaning [1 au=149597870.700 km, 1 day=86400.0 s]:

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
*/
	namespace date_20150511
	{
		string mercury_oe = "2457153.500000000, A.D. 2015-May-11 00:00:00.0000,  2.056283170656704E-01,  3.075005270000471E-01,  7.004033081870772E+00,  4.831135275416084E+01,  2.917018579009503E+01,  2.457132325703553E+06,  4.092332803985572E+00,  8.665226794919347E+01,  1.098801777958040E+02,  3.870990540148295E-01,  4.666975810296119E-01,  8.796938500441402E+01,";
		string venus_oe   = "2457153.500000000, A.D. 2015-May-11 00:00:00.0000,  6.761247965610893E-03,  7.184381867467915E-01,  3.394467689974794E+00,  7.663950855912078E+01,  5.463888630600258E+01,  2.457130873225775E+06,  1.602141893419786E+00,  3.625130289912041E+01,  3.671259128576724E+01,  7.233287920706470E-01,  7.282193973945026E-01,  2.246991989152577E+02,";
		string earth_oe   = "2457153.500000000, A.D. 2015-May-11 00:00:00.0000,  1.706151645376220E-02,  9.828818510541610E-01,  2.832604859699792E-03,  1.991276585462631E+02,  2.644902808333621E+02,  2.457027106783887E+06,  9.856943344121811E-01,  1.245850770307868E+02,  1.261752194211184E+02,  9.999423845001243E-01,  1.017002917946088E+00,  3.652247836188346E+02,";
		string mars_oe    = "2457153.500000000, A.D. 2015-May-11 00:00:00.0000,  9.345598920376896E-02,  1.381200747391636E+00,  1.848403968432629E+00,  4.951276588949865E+01,  2.865318030175783E+02,  2.457003817487961E+06,  5.240856597101560E-01,  7.844645806929105E+01,  8.912748635050312E+01,  1.523589291796774E+00,  1.665977836201911E+00,  6.869106096112168E+02,";
		string jupiter_oe = "2457153.500000000, A.D. 2015-May-11 00:00:00.0000,  4.895926045495331E-02,  4.946812500213837E+00,  1.303927595561599E+00,  1.005220570364609E+02,  2.736235124666692E+02,  2.455634711874421E+06,  8.312310735603498E-02,  1.262463884135522E+02,  1.306085971088266E+02,  5.201472759810757E+00,  5.456133019407678E+00,  4.330925676996638E+03,";
		string saturn_oe  = "2457153.500000000, A.D. 2015-May-11 00:00:00.0000,  5.370451604891303E-02,  9.040758523537329E+00,  2.486273622454862E+00,  1.136214918787423E+02,  3.402344971420399E+02,  2.452848192133344E+06,  3.338097180639689E-02,  1.437153605146956E+02,  1.471679382461214E+02,  9.553843040430950E+00,  1.006692755732457E+01,  1.078458716204937E+04,";
		string uranus_oe  = "2457153.500000000, A.D. 2015-May-11 00:00:00.0000,  4.928987553618186E-02,  1.821116226013173E+01,  7.717997843738144E-01,  7.400078543435393E+01,  9.677390970582711E+01,  2.470049485736452E+06,  1.175653216407963E-02,  2.083879289018873E+02,  2.058417203584891E+02,  1.915532588905842E+01,  2.009948951798512E+01,  3.062127462211412E+04,";
		string neptune_oe = "2457153.500000000, A.D. 2015-May-11 00:00:00.0000,  8.258758489207289E-03,  2.973094026694364E+01,  1.765657591232885E+00,  1.317164628477979E+02,  2.932421741299731E+02,  2.471558726412109E+06,  6.004818034445084E-03,  2.734992366503029E+02,  2.725540643806312E+02,  2.997852567031729E+01,  3.022611107369094E+01,  5.995185831359972E+04,";
	} /* date_20150511 */

	namespace date_
	{
		string mercury_oe = "";
		string venus_oe   = "";
		string earth_oe   = "";
		string mars_oe    = "";
		string jupiter_oe = "";
		string saturn_oe  = "";
		string uranus_oe  = "";
		string neptune_oe = "";
	} /* date_ */
} /* ephemeris */

void populate_solar_system(body_disk_t& disk, pp_disk_t::sim_data_t *sd)
{
    ttt_t epoch = 2457153.5;
	pp_disk_t::param_t param = {0.0, 0.0, 0.0, 0.0};
	pp_disk_t::body_metadata_t body_md = {0, 0, 0.0, MIGRATION_TYPE_NO};
	orbelem_t oe = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // The id of each body must be larger than 0 in order to indicate inactive body with negative id (ie. zero is not good)
    uint32_t bodyIdx = 0;
	uint32_t bodyId = 1;

	// Star: Sun
	{
		disk.names.push_back("Sun");

		body_md.id          = bodyId;
		body_md.body_type   = BODY_TYPE_STAR;

		param.mass          = 1.0;
		param.radius        = 1.0 * constants::SolarRadiusToAu;
		param.density       = tools::calc_density(param.mass, param.radius);

		sd->h_epoch[  bodyIdx] = epoch;
		sd->h_body_md[bodyIdx] = body_md;
		sd->h_p[      bodyIdx] = param;
		sd->h_oe[     bodyIdx] = oe;
		bodyId++, bodyIdx++;
	}

	// Rocky planets: Mercury, Venus, Earth, Mars
	{
		disk.names.push_back("Mercury");

		body_md.id          = bodyId;
		body_md.body_type   = BODY_TYPE_ROCKYPLANET;

		param.mass          = 1.0 * constants::MercuryToSolar;
		param.radius        = 2439.7 * constants::KilometerToAu;
		param.density       = tools::calc_density(param.mass, param.radius);

		epoch = extract_from_horizon_output(ephemeris_major_planets::date_20150511::mercury_oe, oe);

		sd->h_epoch[  bodyIdx] = epoch;
		sd->h_body_md[bodyIdx] = body_md;
		sd->h_p[      bodyIdx] = param;
		sd->h_oe[     bodyIdx] = oe;
		bodyId++, bodyIdx++;

		disk.names.push_back("Venus");

		body_md.id          = bodyId;

		param.mass          = 1.0 * constants::VenusToSolar;
		param.radius        = 6051.8 * constants::KilometerToAu;
		param.density       = tools::calc_density(param.mass, param.radius);

		epoch = extract_from_horizon_output(ephemeris_major_planets::date_20150511::venus_oe, oe);

		sd->h_epoch[  bodyIdx] = epoch;
		sd->h_body_md[bodyIdx] = body_md;
		sd->h_p[      bodyIdx] = param;
		sd->h_oe[     bodyIdx] = oe;
		bodyId++, bodyIdx++;

		disk.names.push_back("Earth");

		body_md.id          = bodyId;

		param.mass          = 1.0 * constants::EarthToSolar;
		param.radius        = 6371.0 * constants::KilometerToAu;
		param.density       = tools::calc_density(param.mass, param.radius);

		epoch = extract_from_horizon_output(ephemeris_major_planets::date_20150511::earth_oe, oe);

		sd->h_epoch[  bodyIdx] = epoch;
		sd->h_body_md[bodyIdx] = body_md;
		sd->h_p[      bodyIdx] = param;
		sd->h_oe[     bodyIdx] = oe;
		bodyId++, bodyIdx++;

		disk.names.push_back("Mars");

		body_md.id          = bodyId;

		param.mass          = 1.0 * constants::MarsToSolar;
		param.radius        = 3389.5 * constants::KilometerToAu;
		param.density       = tools::calc_density(param.mass, param.radius);

		epoch = extract_from_horizon_output(ephemeris_major_planets::date_20150511::mars_oe, oe);

		sd->h_epoch[  bodyIdx] = epoch;
		sd->h_body_md[bodyIdx] = body_md;
		sd->h_p[      bodyIdx] = param;
		sd->h_oe[     bodyIdx] = oe;
		bodyId++, bodyIdx++;
	}

	// Giant planets: Jupiter, Saturn, Uranus, Neptune
	{
		disk.names.push_back("Jupiter");

		body_md.id          = bodyId;
		body_md.body_type   = BODY_TYPE_GIANTPLANET;

		param.mass          = 1.0 * constants::JupiterToSolar;
		param.radius        = 71492.0 * constants::KilometerToAu;
		param.density       = tools::calc_density(param.mass, param.radius);

		epoch = extract_from_horizon_output(ephemeris_major_planets::date_20150511::jupiter_oe, oe);

		sd->h_epoch[  bodyIdx] = epoch;
		sd->h_body_md[bodyIdx] = body_md;
		sd->h_p[      bodyIdx] = param;
		sd->h_oe[     bodyIdx] = oe;
		bodyId++, bodyIdx++;

		disk.names.push_back("Saturn");

		body_md.id          = bodyId;

		param.mass          = 1.0 * constants::SaturnToSolar;
		param.radius        = 60268.0 * constants::KilometerToAu;
		param.density       = tools::calc_density(param.mass, param.radius);

		epoch = extract_from_horizon_output(ephemeris_major_planets::date_20150511::saturn_oe, oe);

		sd->h_epoch[  bodyIdx] = epoch;
		sd->h_body_md[bodyIdx] = body_md;
		sd->h_p[      bodyIdx] = param;
		sd->h_oe[     bodyIdx] = oe;
		bodyId++, bodyIdx++;

		disk.names.push_back("Uranus");

		body_md.id          = bodyId;

		param.mass          = 1.0 * constants::UranusToSolar;
		param.radius        = 25559.0 * constants::KilometerToAu;
		param.density       = tools::calc_density(param.mass, param.radius);

		epoch = extract_from_horizon_output(ephemeris_major_planets::date_20150511::uranus_oe, oe);

		sd->h_epoch[  bodyIdx] = epoch;
		sd->h_body_md[bodyIdx] = body_md;
		sd->h_p[      bodyIdx] = param;
		sd->h_oe[     bodyIdx] = oe;
		bodyId++, bodyIdx++;

		disk.names.push_back("Neptune");

		body_md.id          = bodyId;

		param.mass          = 1.0 * constants::NeptuneToSolar;
		param.radius        = 24766.0 * constants::KilometerToAu;
		param.density       = tools::calc_density(param.mass, param.radius);

		epoch = extract_from_horizon_output(ephemeris_major_planets::date_20150511::neptune_oe, oe);

		sd->h_epoch[  bodyIdx] = epoch;
		sd->h_body_md[bodyIdx] = body_md;
		sd->h_p[      bodyIdx] = param;
		sd->h_oe[     bodyIdx] = oe;
		bodyId++, bodyIdx++;
	}
}

void populate_disk(ttt_t epoch, body_disk_t& disk, pp_disk_t::sim_data_t *sd)
{
	pp_disk_t::param_t param = {0.0, 0.0, 0.0, 0.0};
	pp_disk_t::body_metadata_t body_md = {0, 0, 0.0, MIGRATION_TYPE_NO};
	orbelem_t oe = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // The id of each body must be larger than 0 in order to indicate inactive body with negative id (ie. zero is not good)
    uint32_t bodyIdx = 0;
	uint32_t bodyId  = 1;
	for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_N; body_type++)
	{
		for (int i = 0; i < disk.nBody[body_type]; i++, bodyIdx++, bodyId++)
		{
			body_md.id          = bodyId;
			body_md.body_type   = static_cast<body_type_t>(body_type);
			body_md.mig_type    = disk.mig_type[bodyIdx];
			body_md.mig_stop_at = disk.stop_at[bodyIdx];

			generate_pp(&disk.pp_d[body_type], param);
			if (BODY_TYPE_SUPERPLANETESIMAL != body_type)
			{
				if (0x0 == disk.pp_d[body_type].item[MASS] && 0x0 != disk.pp_d[body_type].item[RADIUS] && 0x0 != disk.pp_d[body_type].item[DENSITY])
				{
					param.mass = tools::calc_mass(param.radius, param.density);
				}
				else if (0x0 == disk.pp_d[body_type].item[RADIUS] && 0x0 != disk.pp_d[body_type].item[MASS] && 0x0 != disk.pp_d[body_type].item[DENSITY])
				{
					param.radius = tools::calc_radius(param.mass, param.density);
				}
				else if (0x0 == disk.pp_d[body_type].item[DENSITY] && 0x0 != disk.pp_d[body_type].item[MASS] && 0x0 != disk.pp_d[body_type].item[RADIUS])
				{
					param.density = tools::calc_density(param.mass, param.radius);
				}
				else
				{
					throw string("Missing physical parameters!");
				}
			}

			if (BODY_TYPE_STAR == body_type)
			{
				oe.sma = oe.ecc = oe.inc = oe.peri = oe.node = oe.mean = 0.0;
			}
			else
			{
				generate_oe(&disk.oe_d[body_type], oe);
			}

            sd->h_epoch[  bodyIdx] = epoch;
			sd->h_body_md[bodyIdx] = body_md;
			sd->h_p[      bodyIdx] = param;
			sd->h_oe[     bodyIdx] = oe;
		} /* for */
	} /* for */
}

// Calculate coordinates, velocities and minimal orbital period from the orbital elements
void calculate_phase(pp_disk_t::sim_data_t* sim_data, uint32_t n_body, ttt_t &dt)
{
	var_t min_P = DBL_MAX;
	// The mass of the central star
	var_t m0    = sim_data->h_p[0].mass;
	var4_t rVec = {0.0, 0.0, 0.0, 0.0};
	var4_t vVec = {0.0, 0.0, 0.0, 0.0};

	// The coordinates of the central star
	sim_data->h_y[0][0] = rVec;
	sim_data->h_y[1][0] = vVec;
	for (uint32_t i = 1; i < n_body; i++)
	{
		var_t mu = K2 *(m0 + sim_data->h_p[i].mass);
		tools::calc_phase(mu, &sim_data->h_oe[i], &rVec, &vVec);
		sim_data->h_y[0][i] = rVec;
		sim_data->h_y[1][i] = vVec;

		ttt_t P = tools::calc_orbital_period(mu, sim_data->h_oe[i].sma);
		if (min_P > P)
		{
			min_P = P;
		}
	}
	dt = min_P / 1000.0;
}

void print_all_input_data(string& dir, string& filename, uint32_t n_body, uint32_t seed, ttt_t t0, ttt_t dt, body_disk_t& disk, pp_disk_t::sim_data_t* sim_data)
{
	string path = file::combine_path(dir, filename) + ".seed.txt";
	print_number(path, seed);

	path = file::combine_path(dir, filename) + ".info.txt";
	print_data_info(path, t0, dt, disk, INPUT_FORMAT_RED);

	path = file::combine_path(dir, filename) + ".oe.txt";
	print_oe(path, n_body, t0, sim_data);

	path = file::combine_path(dir, filename) + ".data.txt";
	print_data(path, disk, sim_data, INPUT_FORMAT_RED);

	//path = file::combine_path(dir, filename) + "_NONMAE.txt";
	//print_data(path, disk, sim_data, INPUT_FORMAT_NONAME);

	//path = file::combine_path(dir, filename) + "_HIPERION.txt";
	//print_data(path, disk, sim_data, INPUT_FORMAT_HIPERION);
}

namespace set_parameters
{
uint32_t Chambers2001(nebula& n, body_disk_t& disk)
{
	uint32_t seed = (uint32_t)time(NULL);
	cout << "The seed number is " << seed << endl;
	//The pseudo-random number generator is initialized using the argument passed as seed.
	srand(seed);

	const var_t rho_solid = 3.0 /* g/cm^3 */ * constants::GramPerCm3ToSolarPerAu3;

	disk.nBody[BODY_TYPE_STAR        ] = 1;
	disk.nBody[BODY_TYPE_PROTOPLANET ] = 153;

	uint32_t n_body = calc_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[n_body];
	disk.stop_at  = new var_t[n_body];

    uint32_t bodyIdx = 0;
	int type = BODY_TYPE_STAR;

	disk.names.push_back("star");
	disk.pp_d[type].item[MASS]       = new uniform_distribution(rand(), 1.0, 1.0);
	disk.pp_d[type].item[RADIUS]     = new uniform_distribution(rand(), constants::SolarRadiusToAu, constants::SolarRadiusToAu);
	disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

	disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
	disk.stop_at[bodyIdx] = 0.0;

	type = BODY_TYPE_PROTOPLANET;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new power_law_distribution(rand(), n.solid_c.get_r_1(), n.solid_c.get_r_2(), n.solid_c.get_p());

		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new rayleigh_distribution(rand(), 0.01);
		disk.oe_d[type].range[ORBITAL_ELEMENT_ECC ].x = 0.0;
		disk.oe_d[type].range[ORBITAL_ELEMENT_ECC ].y = 0.2;

		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 2.0 * PI);

		disk.pp_d[type].item[MASS      ] = new uniform_distribution(rand(), 1.0/60.0 * constants::EarthToSolar, 1.0/60.0 * constants::EarthToSolar);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(rand(), rho_solid, rho_solid);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i + 1, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	return seed;
}

uint32_t solar_system(body_disk_t& disk)
{
	uint32_t seed = (uint32_t)time(NULL);
	cout << "The seed number is " << seed << endl;
	//The pseudo-random number generator is initialized using the argument passed as seed.
	srand(seed);

	disk.nBody[BODY_TYPE_STAR       ] = 1;
	disk.nBody[BODY_TYPE_ROCKYPLANET] = 4;
	disk.nBody[BODY_TYPE_GIANTPLANET] = 4;

	uint32_t n_body = calc_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[n_body];
	disk.stop_at = new var_t[n_body];

	return seed;
}

uint32_t pl_to_test_anal_gd(body_disk_t& disk)
{
	uint32_t seed = (uint32_t)time(NULL);
	cout << "The seed number is " << seed << endl;
	//The pseudo-random number generator is initialized using the argument passed as seed.
	srand(seed);

	disk.nBody[BODY_TYPE_STAR        ] = 1;
	disk.nBody[BODY_TYPE_PLANETESIMAL] = 10;

	uint32_t n_body = calc_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[n_body];
	disk.stop_at = new var_t[n_body];

    uint32_t bodyIdx = 0;
	int type = BODY_TYPE_STAR;

	disk.names.push_back("star");
	disk.pp_d[type].item[MASS]       = new uniform_distribution(rand(), 1.0, 1.0);
	disk.pp_d[type].item[RADIUS]     = new uniform_distribution(rand(), 1.0*constants::SolarRadiusToAu, 1.0*constants::SolarRadiusToAu);
	disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

	disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
	disk.stop_at[bodyIdx] = 0.0;

	type = BODY_TYPE_PLANETESIMAL;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new uniform_distribution(rand(), 1.0, 10.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 0.0);

		//disk.pp_d[type].item[MASS      ] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.pp_d[type].item[RADIUS    ] = new uniform_distribution(rand(), 1.0*constants::MeterToAu, 1.0*constants::MeterToAu);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(rand(), 2.7*constants::GramPerCm3ToSolarPerAu3, 2.7*constants::GramPerCm3ToSolarPerAu3);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 1.0, 1.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	return seed;
}

uint32_t coll_stat_run(nebula& n, body_disk_t& disk)
{
	uint32_t seed = (uint32_t)time(NULL);
	cout << "The seed number is " << seed << endl;
	//The pseudo-random number generator is initialized using the argument passed as seed.
	srand(seed);

	const var_t rhoSilicate = 2.0 /* g/cm^3 */ * constants::GramPerCm3ToSolarPerAu3;

	disk.nBody[BODY_TYPE_STAR        ] = 1;
	disk.nBody[BODY_TYPE_PROTOPLANET ] = 10000;

	uint32_t n_body = calc_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[n_body];
	disk.stop_at  = new var_t[n_body];

    uint32_t bodyIdx = 0;
	int type = BODY_TYPE_STAR;

	disk.names.push_back("star");
	disk.pp_d[type].item[MASS]       = new uniform_distribution(rand(), 1.0, 1.0);
	disk.pp_d[type].item[RADIUS]     = new uniform_distribution(rand(), constants::SolarRadiusToAu, constants::SolarRadiusToAu);
	disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

	disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
	disk.stop_at[bodyIdx]  = 0.0;

	type = BODY_TYPE_PROTOPLANET;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new power_law_distribution(rand(), n.solid_c.get_r_1(), n.solid_c.get_r_2(), n.solid_c.get_p());

		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new rayleigh_distribution(rand(), 0.02);
		disk.oe_d[type].range[ORBITAL_ELEMENT_ECC].x = 0.0;
		disk.oe_d[type].range[ORBITAL_ELEMENT_ECC].y = 0.2;

		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 2.0 * PI);

		disk.pp_d[type].item[MASS      ] = new uniform_distribution(rand(), 1.0, 1.0);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(rand(), rhoSilicate, rhoSilicate);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i + 1, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	return seed;
}

uint32_t Dvorak(nebula& n, body_disk_t& disk)
{
	uint32_t seed = (uint32_t)time(NULL);
	cout << "The seed number is " << seed << endl;
	//The pseudo-random number generator is initialized using the argument passed as seed.
	srand(seed);

	const var_t rhoBasalt = 2.7 /* g/cm^3 */ * constants::GramPerCm3ToSolarPerAu3;

	disk.nBody[BODY_TYPE_STAR        ] = 1;
	disk.nBody[BODY_TYPE_PROTOPLANET ] = 2000;

	uint32_t n_body = calc_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[n_body];
	disk.stop_at  = new var_t[n_body];

    uint32_t bodyIdx = 0;
	int type = BODY_TYPE_STAR;

	disk.names.push_back("star");
	disk.pp_d[type].item[MASS]       = new uniform_distribution(rand(), 1.0, 1.0);
	disk.pp_d[type].item[RADIUS]     = new uniform_distribution(rand(), constants::SolarRadiusToAu, constants::SolarRadiusToAu);
	disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

	disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
	disk.stop_at[bodyIdx] = 0.0;

	type = BODY_TYPE_PROTOPLANET;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new power_law_distribution(rand(), n.solid_c.get_r_1(), n.solid_c.get_r_2(), n.solid_c.get_p());

		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new rayleigh_distribution(rand(), 0.01);
		disk.oe_d[type].range[ORBITAL_ELEMENT_ECC ].x = 0.0;
		disk.oe_d[type].range[ORBITAL_ELEMENT_ECC ].y = 0.2;

		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 2.0 * PI);

		disk.pp_d[type].item[MASS      ] = new lognormal_distribution(rand(), 0.1, 3.0, 0.0, 0.25);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(  rand(), rhoBasalt, rhoBasalt);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(  rand(), 0.0, 0.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i + 1, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	return seed;
}

uint32_t Hansen_2009(body_disk_t& disk)
{
	uint32_t seed = (uint32_t)time(NULL);
	cout << "The seed number is " << seed << endl;
	//The pseudo-random number generator is initialized using the argument passed as seed.
	srand(seed);

	const var_t rhoBasalt = 2.7 /* g/cm^3 */ * constants::GramPerCm3ToSolarPerAu3;

	disk.nBody[BODY_TYPE_STAR        ] = 1;
	disk.nBody[BODY_TYPE_GIANTPLANET ] = 1;
	disk.nBody[BODY_TYPE_PROTOPLANET ] = 400;

	uint32_t n_body = calc_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[n_body];
	disk.stop_at  = new var_t[n_body];

    uint32_t bodyIdx = 0;
	int type = BODY_TYPE_STAR;

	disk.names.push_back("star");
	disk.pp_d[type].item[MASS]       = new uniform_distribution(rand(), 1.0, 1.0);
	disk.pp_d[type].item[RADIUS]     = new uniform_distribution(rand(), constants::SolarRadiusToAu, constants::SolarRadiusToAu);
	disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

	disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
	disk.stop_at[bodyIdx] = 0.0;

	type = BODY_TYPE_GIANTPLANET;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new uniform_distribution(rand(), 5.2, 5.2);
		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new uniform_distribution(rand(), 0.05, 0.05);
		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 2.0 * PI);

		disk.pp_d[type].item[MASS      ] = new uniform_distribution(rand(), 1.0 * constants::JupiterToSolar, 1.0 * constants::JupiterToSolar);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(rand(), 1.326 * constants::GramPerCm3ToSolarPerAu3, 1.326 * constants::GramPerCm3ToSolarPerAu3);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i+1, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	type = BODY_TYPE_PROTOPLANET;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new uniform_distribution(rand(), 0.7, 1.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 0.5 * constants::DegreeToRadian);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 2.0 * PI);

		disk.pp_d[type].item[MASS      ] = new uniform_distribution(rand(), 5.0e-3 * constants::EarthToSolar, 5.0e-3 * constants::EarthToSolar);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(rand(), rhoBasalt, rhoBasalt);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i + 1, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	return seed;
}

uint32_t Two_body(body_disk_t& disk)
{
	uint32_t seed = (uint32_t)time(NULL);
	cout << "The seed number is " << seed << endl;
	//The pseudo-random number generator is initialized using the argument passed as seed.
	srand(seed);

	disk.nBody[BODY_TYPE_STAR       ] = 1;
	disk.nBody[BODY_TYPE_ROCKYPLANET] = 1;

	uint32_t n_body = calc_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[n_body];
	disk.stop_at = new var_t[n_body];

    uint32_t bodyIdx = 0;
	int type = BODY_TYPE_STAR;

	disk.names.push_back("star");
	disk.pp_d[type].item[MASS]       = new uniform_distribution(rand(), 1.0, 1.0);
	disk.pp_d[type].item[RADIUS]     = new uniform_distribution(rand(), 1.0*constants::SolarRadiusToAu, 1.0*constants::SolarRadiusToAu);
	disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

	disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
	disk.stop_at[bodyIdx] = 0.0;

	type = BODY_TYPE_ROCKYPLANET;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new uniform_distribution(rand(), 1.0, 1.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 0.0);

		disk.pp_d[type].item[MASS      ] = new uniform_distribution(rand(),	1.0*constants::EarthToSolar, 1.0*constants::EarthToSolar);
		//disk.pp_d[type].item[RADIUS    ] = new uniform_distribution(rand(), 1.0*constants::MeterToAu, 1.0*constants::MeterToAu);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(rand(), 2.7*constants::GramPerCm3ToSolarPerAu3, 2.7*constants::GramPerCm3ToSolarPerAu3);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 1.0, 1.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	return seed;
}

uint32_t n_gp(body_disk_t& disk)
{
	uint32_t seed = (uint32_t)time(NULL);
	cout << "The seed number is " << seed << endl;
	//The pseudo-random number generator is initialized using the argument passed as seed.
	srand(seed);

	disk.nBody[BODY_TYPE_STAR       ] = 1;
	disk.nBody[BODY_TYPE_GIANTPLANET] = 10;

	uint32_t n_body = calc_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[n_body];
	disk.stop_at = new var_t[n_body];

    uint32_t bodyIdx = 0;
	int type = BODY_TYPE_STAR;

	disk.names.push_back("star");
	disk.pp_d[type].item[MASS]       = new uniform_distribution(rand(), 1.0, 1.0);

	var_t tmp = constants::SolarRadiusToAu;
	disk.pp_d[type].item[RADIUS]     = new uniform_distribution(rand(), tmp, tmp);

	disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

	disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
	disk.stop_at[bodyIdx] = 0.0;

	type = BODY_TYPE_GIANTPLANET;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new uniform_distribution(rand(), 1.0, 10.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new uniform_distribution(rand(), 0.0, 0.1);
		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 10.0 * constants::DegreeToRadian);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 2.0 * PI);

		disk.pp_d[type].item[MASS      ] = new uniform_distribution(rand(), 0.1 * constants::SaturnToSolar, 1.0 * constants::SaturnToSolar);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(rand(), 1.0 * constants::GramPerCm3ToSolarPerAu3, 2.0 * constants::GramPerCm3ToSolarPerAu3);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i+1, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	return seed;
}

uint32_t n_pp(body_disk_t& disk)
{
	uint32_t seed = (uint32_t)time(NULL);
	cout << "The seed number is " << seed << endl;
	//The pseudo-random number generator is initialized using the argument passed as seed.
	srand(seed);

	disk.nBody[BODY_TYPE_STAR       ] = 1;
	disk.nBody[BODY_TYPE_PROTOPLANET] = 500;
	//disk.nBody[BODY_TYPE_PROTOPLANET] = 2000;
	//disk.nBody[BODY_TYPE_PROTOPLANET] = 10000;

	uint32_t n_body = calc_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[n_body];
	disk.stop_at = new var_t[n_body];

    uint32_t bodyIdx = 0;
	int type = BODY_TYPE_STAR;

	disk.names.push_back("star");
	disk.pp_d[type].item[MASS]       = new uniform_distribution(rand(), 1.0, 1.0);
	disk.pp_d[type].item[RADIUS]     = new uniform_distribution(rand(), constants::SolarRadiusToAu, constants::SolarRadiusToAu);
	disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

	disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
	disk.stop_at[bodyIdx] = 0.0;

	type = BODY_TYPE_PROTOPLANET;
	{
  		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new uniform_distribution(rand(), 0.6, 2.6);
		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new uniform_distribution(rand(), 0.0, 0.1);
		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 0.0 * constants::DegreeToRadian);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 0.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 2.0 * PI);

		disk.pp_d[type].item[MASS      ] = new uniform_distribution(rand(), 0.5 * constants::MoonToSolar, 2.0 * constants::MoonToSolar);
		//disk.pp_d[type].item[RADIUS    ] = new uniform_distribution(rand(), 1.0e2 * constants::KilometerToAu, 1.0e2 * constants::KilometerToAu);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(rand(), 2.7 * constants::GramPerCm3ToSolarPerAu3, 2.7 * constants::GramPerCm3ToSolarPerAu3);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i+1, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	return seed;
}

uint32_t n_spl(body_disk_t& disk)
{
	uint32_t seed = (uint32_t)time(NULL);
	cout << "The seed number is " << seed << endl;
	//The pseudo-random number generator is initialized using the argument passed as seed.
	srand(seed);

	disk.nBody[BODY_TYPE_STAR             ] = 1;
	disk.nBody[BODY_TYPE_SUPERPLANETESIMAL] = 10;

	uint32_t n_body = calc_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[n_body];
	disk.stop_at = new var_t[n_body];

    uint32_t bodyIdx = 0;

	int type = BODY_TYPE_STAR;
	{
		disk.names.push_back("star");
		disk.pp_d[type].item[MASS]       = new uniform_distribution(rand(), 1.0, 1.0);

		var_t tmp = constants::SolarRadiusToAu;
		disk.pp_d[type].item[RADIUS]     = new uniform_distribution(rand(), tmp, tmp);

		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
		disk.stop_at[bodyIdx] = 0.0;
	}

	type = BODY_TYPE_SUPERPLANETESIMAL;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new uniform_distribution(rand(), 1.0, 10.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new uniform_distribution(rand(), 0.0, 0.1);
		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 10.0 * constants::DegreeToRadian);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 2.0 * PI);

		disk.pp_d[type].item[MASS      ] = new uniform_distribution(rand(), 1.0e-10, 1.0e-9);
		disk.pp_d[type].item[RADIUS    ] = new uniform_distribution(rand(), 10.0 * constants::KilometerToAu, 30.0 * constants::KilometerToAu);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(rand(), 1.0 * constants::GramPerCm3ToSolarPerAu3, 3.0 * constants::GramPerCm3ToSolarPerAu3);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 1.0, 2.7);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i+1, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	return seed;
}

uint32_t n_pl(body_disk_t& disk)
{
	uint32_t seed = (uint32_t)time(NULL);
	cout << "The seed number is " << seed << endl;
	//The pseudo-random number generator is initialized using the argument passed as seed.
	srand(seed);

	disk.nBody[BODY_TYPE_STAR        ] = 1;
	disk.nBody[BODY_TYPE_PLANETESIMAL] = 10;

	uint32_t n_body = calc_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[n_body];
	disk.stop_at = new var_t[n_body];

    uint32_t bodyIdx = 0;

	int type = BODY_TYPE_STAR;
	{
		disk.names.push_back("star");
		disk.pp_d[type].item[MASS]       = new uniform_distribution(rand(), 1.0, 1.0);

		var_t tmp = constants::SolarRadiusToAu;
		disk.pp_d[type].item[RADIUS]     = new uniform_distribution(rand(), tmp, tmp);

		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
		disk.stop_at[bodyIdx] = 0.0;
	}

	type = BODY_TYPE_PLANETESIMAL;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new uniform_distribution(rand(), 1.0, 10.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new uniform_distribution(rand(), 0.0, 0.1);
		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 10.0 * constants::DegreeToRadian);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 2.0 * PI);

		disk.pp_d[type].item[MASS      ] = new uniform_distribution(rand(), 1.0e-3 * constants::CeresToSolar, 1.0e-2 * constants::CeresToSolar);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(rand(), 1.0 * constants::GramPerCm3ToSolarPerAu3, 3.0 * constants::GramPerCm3ToSolarPerAu3);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i+1, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	return seed;
}

uint32_t n_tp(body_disk_t& disk)
{
	uint32_t seed = (uint32_t)time(NULL);
	cout << "The seed number is " << seed << endl;
	//The pseudo-random number generator is initialized using the argument passed as seed.
	srand(seed);

	disk.nBody[BODY_TYPE_STAR        ] = 1;
	disk.nBody[BODY_TYPE_TESTPARTICLE] = 10;

	uint32_t n_body = calc_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[n_body];
	disk.stop_at = new var_t[n_body];

    uint32_t bodyIdx = 0;

	int type = BODY_TYPE_STAR;
	{
		disk.names.push_back("star");
		disk.pp_d[type].item[MASS]       = new uniform_distribution(rand(), 1.0, 1.0);

		var_t tmp = constants::SolarRadiusToAu;
		disk.pp_d[type].item[RADIUS]     = new uniform_distribution(rand(), tmp, tmp);

		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
		disk.stop_at[bodyIdx] = 0.0;
	}

	type = BODY_TYPE_TESTPARTICLE;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new uniform_distribution(rand(), 1.0, 10.0);
		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new uniform_distribution(rand(), 0.0, 0.1);
		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 10.0 * constants::DegreeToRadian);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 2.0 * PI);

		disk.pp_d[type].item[MASS      ] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.pp_d[type].item[RADIUS    ] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(rand(), 0.0, 0.0);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i+1, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	return seed;
}

uint32_t GT_scenario(body_disk_t& disk)
{
	uint32_t seed = (uint32_t)time(NULL);
	cout << "The seed number is " << seed << endl;
	//The pseudo-random number generator is initialized using the argument passed as seed.
	srand(seed);

	disk.nBody[BODY_TYPE_STAR        ] = 1;
	disk.nBody[BODY_TYPE_GIANTPLANET ] = 2;
	disk.nBody[BODY_TYPE_PROTOPLANET ] = 2000;

	uint32_t n_body = calc_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[n_body];
	disk.stop_at = new var_t[n_body];

    uint32_t bodyIdx = 0;
	int type = BODY_TYPE_STAR;
	{
		disk.names.push_back("star");
		disk.pp_d[type].item[MASS]       = new uniform_distribution(rand(), 1.0, 1.0);

		var_t tmp = constants::SolarRadiusToAu;
		disk.pp_d[type].item[RADIUS]     = new uniform_distribution(rand(), tmp, tmp);

		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
		disk.stop_at[bodyIdx] = 0.0;
	}

	type = BODY_TYPE_GIANTPLANET;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new uniform_distribution(rand(), 1.0, 2.00);
		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new uniform_distribution(rand(), 0.0, 0.05);
		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 1.0 * constants::DegreeToRadian);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 2.0 * PI);

		disk.pp_d[type].item[MASS      ] = new uniform_distribution(rand(), 1.0 * constants::JupiterToSolar, 0.0);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(rand(), 2.7 * constants::GramPerCm3ToSolarPerAu3, 0.0);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i+1, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	type = BODY_TYPE_PROTOPLANET;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new normal_distribution(rand(), 1.45 /* AU */, 0.35);
		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new uniform_distribution(rand(), 0.0, 0.05);
		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 1.0 * constants::DegreeToRadian);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 2.0 * PI);

		disk.pp_d[type].item[MASS      ] = new uniform_distribution(rand(), 1.0/2.0 * constants::MoonToEarth*constants::EarthToSolar, 1.0/2.0 * constants::MoonToEarth*constants::EarthToSolar);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(rand(), 2.7 * constants::GramPerCm3ToSolarPerAu3, 0.0);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i+1, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	return seed;
}

uint32_t GT_scenario_mod(body_disk_t& disk)
{
	uint32_t seed = (uint32_t)time(NULL);
	cout << "The seed number is " << seed << endl;
	//The pseudo-random number generator is initialized using the argument passed as seed.
	srand(seed);

	disk.nBody[BODY_TYPE_STAR        ] = 1;
	disk.nBody[BODY_TYPE_GIANTPLANET ] = 3;
	disk.nBody[BODY_TYPE_PROTOPLANET ] = 2000;

	uint32_t n_body = calc_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[n_body];
	disk.stop_at = new var_t[n_body];

    uint32_t bodyIdx = 0;
	int type = BODY_TYPE_STAR;
	{
		disk.names.push_back("Sun");
		disk.pp_d[type].item[MASS]       = new uniform_distribution(rand(), 1.0, 1.0);

		var_t tmp = constants::SolarRadiusToAu;
		disk.pp_d[type].item[RADIUS]     = new uniform_distribution(rand(), tmp, tmp);

		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
		disk.stop_at[bodyIdx] = 0.0;
	}

	type = BODY_TYPE_GIANTPLANET;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new uniform_distribution(rand(), 1.0, 2.00);
		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new uniform_distribution(rand(), 0.0, 0.05);
		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 1.0 * constants::DegreeToRadian);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 2.0 * PI);

		disk.pp_d[type].item[MASS      ] = new uniform_distribution(rand(), 1.0 * constants::JupiterToSolar, 0.0);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(rand(), 2.7 * constants::GramPerCm3ToSolarPerAu3, 0.0);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i+1, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	type = BODY_TYPE_PROTOPLANET;
	{
		disk.oe_d[type].item[ORBITAL_ELEMENT_SMA ] = new exponential_distribution(rand(), 0.5);
		disk.oe_d[type].item[ORBITAL_ELEMENT_ECC ] = new uniform_distribution(rand(), 0.0, 0.05);
		disk.oe_d[type].item[ORBITAL_ELEMENT_INC ] = new uniform_distribution(rand(), 0.0, 1.0 * constants::DegreeToRadian);
		disk.oe_d[type].item[ORBITAL_ELEMENT_PERI] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_NODE] = new uniform_distribution(rand(), 0.0, 2.0 * PI);
		disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN] = new uniform_distribution(rand(), 0.0, 2.0 * PI);

		disk.pp_d[type].item[MASS      ] = new uniform_distribution(rand(), 1.0/2.0 * constants::MoonToEarth*constants::EarthToSolar, 1.0/2.0 * constants::MoonToEarth*constants::EarthToSolar);
		disk.pp_d[type].item[DENSITY   ] = new uniform_distribution(rand(), 2.7 * constants::GramPerCm3ToSolarPerAu3, 0.0);
		disk.pp_d[type].item[DRAG_COEFF] = new uniform_distribution(rand(), 0.0, 0.0);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i+1, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	return seed;
}
} /* set_parameters */

namespace create_disk
{
void Chambers2001(string& dir, string& filename)
{
	// Epoch for the disk's state
	ttt_t t0 = 0.0;

	body_disk_t disk;
	initialize(disk);

	// Inner disk: r = 0.3, ... 0.7: the surface density is prop to r
	{
		// Create a MMSN with gas component and solids component
		var_t r_1  =  0.025;    /* inner rim of the disk [AU] */
		var_t r_2  = 33.0;      /* outer rim of the disk [AU] */
		var_t r_SL =  2.7;      /* distance of the snowline [AU] */
		var_t f_neb = 1.0;	    
		var_t f_ice = 4.2;      /* ice condensation coefficient beyond the snowline */
		var_t Sigma_1 = 9.0;  /* Surface density of solids at r = 1 AU */
		var_t f_gas = 240.0;    /* gas to dust ratio */
		var_t p = 1.0;          /* profile index of the power-law function */

		Sigma_1 *= constants::GramPerCm2ToSolarPerAu2;
		gas_component gas_c(r_1, r_2, r_SL, f_neb, Sigma_1, f_gas, p);

		r_1 = 0.3;
		r_2 = 0.7;
		solid_component solid_c(r_1, r_2, r_SL, f_neb, Sigma_1, f_ice, p);
		nebula mmsn(gas_c, solid_c);

		var_t m_gas   = mmsn.gas_c.calc_mass();
		var_t m_solid = mmsn.solid_c.calc_mass();

		var_t m_pp = (1.0 / 60.0) * constants::EarthToSolar;
		int_t n_pp = (int)(m_solid / m_pp);

	}

	// Outer disk: r = 0.7, ... 2.0: the surface density is prop to r ^-3/2
	{
		// Create a MMSN with gas component and solids component
		var_t r_1  =  0.025;    /* inner rim of the disk [AU] */
		var_t r_2  = 33.0;      /* outer rim of the disk [AU] */
		var_t r_SL =  2.7;      /* distance of the snowline [AU] */
		var_t f_neb = 1.0;	    
		var_t f_ice = 4.2;      /* ice condensation coefficient beyond the snowline */
		var_t Sigma_1 = 9.0;  /* Surface density of solids at r = 1 AU */
		var_t f_gas = 240.0;    /* gas to dust ratio */
		var_t p = -3.0/2.0;     /* profile index of the power-law function */

		Sigma_1 *= constants::GramPerCm2ToSolarPerAu2;
		gas_component gas_c(r_1, r_2, r_SL, f_neb, Sigma_1, f_gas, p);

		r_1 = 0.7;
		r_2 = 2.0;
		solid_component solid_c(r_1, r_2, r_SL, f_neb, Sigma_1, f_ice, p);
		nebula mmsn(gas_c, solid_c);

		var_t m_gas   = mmsn.gas_c.calc_mass();
		var_t m_solid = mmsn.solid_c.calc_mass();

		var_t m_pp = (1.0 / 60.0) * constants::EarthToSolar;
		int_t n_pp = (int)(m_solid / m_pp);

		uint32_t seed = set_parameters::Chambers2001(mmsn, disk);

		pp_disk_t::sim_data_t* sim_data = new pp_disk_t::sim_data_t;
		uint32_t n_body = calc_number_of_bodies(disk);
		allocate_host_storage(sim_data, n_body);

		populate_disk(t0, disk, sim_data);

		// Scale the masses in order to get the required mass transform_mass()
		{
			var_t m_total_pp = tools::get_total_mass(n_body, BODY_TYPE_PROTOPLANET, sim_data);
			var_t f = m_solid / m_total_pp;
			for (uint32_t i = 0; i < n_body; i++)
			{
				// Only the masses of the protoplanets will be scaled
				if (sim_data->h_body_md[i].body_type == BODY_TYPE_PROTOPLANET)
				{
					sim_data->h_p[i].mass *= f;
				}
			}
			m_total_pp = tools::get_total_mass(n_body, BODY_TYPE_PROTOPLANET, sim_data);
			if (fabs(m_total_pp - m_solid) > 1.0e-15)
			{
				cerr << "The required mass was not reached." << endl;
				exit(0);
			}
		}

		// Computes the physical quantities with the new mass
		{
			uint32_t bodyIdx = 0;
			for (int type = BODY_TYPE_STAR; type < BODY_TYPE_N; type++)
			{
				for (int i = 0; i < disk.nBody[type]; i++, bodyIdx++)
				{
					if (sim_data->h_p[bodyIdx].mass > 0.0)
					{
						if (disk.pp_d[type].item[DENSITY] == 0x0 && disk.pp_d[type].item[RADIUS] != 0x0)
						{
							sim_data->h_p[bodyIdx].density = tools::calc_density(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].radius);
						}
						if (disk.pp_d[type].item[RADIUS] == 0x0 && disk.pp_d[type].item[DENSITY] != 0x0)
						{
							sim_data->h_p[bodyIdx].radius = tools::calc_radius(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].density);
						}
					}
				}
			}
		}

		ttt_t dt = 0.0;
		calculate_phase(sim_data, n_body, dt);
	}


	// Create a MMSN with gas component and solids component
	var_t r_1  =  0.025;  /* inner rim of the disk [AU] */
	var_t r_2  = 33.0;    /* outer rim of the disk [AU] */
	var_t r_SL =  2.7;    /* distance of the snowline [AU] */
	var_t f_neb = 1.0;
	var_t f_ice = 4.2;    /* ice condensation coefficient beyond the snowline */
	var_t Sigma_1 = 8.0;  /* Surface density of solids at r = 1 AU */
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

	uint32_t seed = set_parameters::Dvorak(mmsn, disk);

	pp_disk_t::sim_data_t* sim_data = new pp_disk_t::sim_data_t;
	uint32_t n_body = calc_number_of_bodies(disk);
	allocate_host_storage(sim_data, n_body);

	populate_disk(t0, disk, sim_data);

	// Scale the masses in order to get the required mass transform_mass()
	{
		var_t m_total_pp = tools::get_total_mass(n_body, BODY_TYPE_PROTOPLANET, sim_data);
		var_t f = m_solid / m_total_pp;
		for (uint32_t i = 0; i < n_body; i++)
		{
			// Only the masses of the protoplanets will be scaled
			if (sim_data->h_body_md[i].body_type == BODY_TYPE_PROTOPLANET)
			{
				sim_data->h_p[i].mass *= f;
			}
		}
		m_total_pp = tools::get_total_mass(n_body, BODY_TYPE_PROTOPLANET, sim_data);
		if (fabs(m_total_pp - m_solid) > 1.0e-15)
		{
			cerr << "The required mass was not reached." << endl;
			exit(0);
		}
	}

	// Computes the physical quantities with the new mass
	{
		uint32_t bodyIdx = 0;
		for (int type = BODY_TYPE_STAR; type < BODY_TYPE_N; type++)
		{
			for (int i = 0; i < disk.nBody[type]; i++, bodyIdx++)
			{
				if (sim_data->h_p[bodyIdx].mass > 0.0)
				{
					if (disk.pp_d[type].item[DENSITY] == 0x0 && disk.pp_d[type].item[RADIUS] != 0x0)
					{
						sim_data->h_p[bodyIdx].density = tools::calc_density(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].radius);
					}
					if (disk.pp_d[type].item[RADIUS] == 0x0 && disk.pp_d[type].item[DENSITY] != 0x0)
					{
						sim_data->h_p[bodyIdx].radius = tools::calc_radius(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].density);
					}
				}
			}
		}
	}

	// Calculate coordinates and velocities
	ttt_t dt = 0.0;
	calculate_phase(sim_data, n_body, dt);

	tools::transform_to_bc(n_body, false, sim_data);

	print_all_input_data(dir, filename, n_body, seed, t0, dt, disk, sim_data);

	deallocate_host_storage(sim_data);

	delete sim_data;
}

void coll_stat_run(string& dir, string& filename)
{
	// Epoch for the disk's state
	ttt_t t0 = 0.0;

	body_disk_t disk;
	initialize(disk);

	// Create a MMSN with gas component and solids component
	var_t r_1     =   0.5;      /* inner rim of the disk [AU]    */
	var_t r_2     =   1.5;      /* outer rim of the disk [AU]    */
	var_t r_SL    =   2.7;      /* distance of the snowline [AU] */
	var_t f_neb   =   1.0;
	var_t f_ice   =   4.2;      /* ice condensation coefficient beyond the snowline */
	var_t Sigma_1 =   7.0;      /* Surface density of solids at r = 1 AU            */
	var_t f_gas   = 240.0;      /* gas to dust ratio                                */
	var_t p       =  -3.0/2.0;  /* profile index of the power-law function          */

	Sigma_1 *= constants::GramPerCm2ToSolarPerAu2;
	gas_component gas_c(r_1, r_2, r_SL, f_neb, Sigma_1, f_gas, p);

	r_1 = 0.5;
	r_2 = 1.5;
	solid_component solid_c(r_1, r_2, r_SL, f_neb, Sigma_1, f_ice, p);
	nebula mmsn(gas_c, solid_c);

	var_t m_gas   = mmsn.gas_c.calc_mass();
	var_t m_solid = mmsn.solid_c.calc_mass();

	uint32_t seed = set_parameters::coll_stat_run(mmsn, disk);

	pp_disk_t::sim_data_t* sim_data = new pp_disk_t::sim_data_t;
	uint32_t n_body = calc_number_of_bodies(disk);
	allocate_host_storage(sim_data, n_body);

	populate_disk(t0, disk, sim_data);

	// Scale the masses in order to get the required mass transform_mass()
	{
		uint32_t n_pp = calc_number_of_bodies(disk, BODY_TYPE_PROTOPLANET);
		var_t m_pp = m_solid / n_pp;
		for (uint32_t i = 0; i < n_body; i++)
		{
			// Only the masses of the protoplanets will be scaled
			if (sim_data->h_body_md[i].body_type == BODY_TYPE_PROTOPLANET)
			{
				sim_data->h_p[i].mass = m_pp;
			}
		}
		var_t m_total_pp = tools::get_total_mass(n_body, BODY_TYPE_PROTOPLANET, sim_data);
		if (fabs(m_total_pp - m_solid) > 1.0e-15)
		{
			cerr << "The required mass was not reached." << endl;
			exit(0);
		}
	}

	// Computes the physical quantities with the new mass
	{
		uint32_t bodyIdx = 0;
		for (int type = BODY_TYPE_STAR; type < BODY_TYPE_N; type++)
		{
			for (int i = 0; i < disk.nBody[type]; i++, bodyIdx++)
			{
				if (0.0 < sim_data->h_p[bodyIdx].mass)
				{
					if (disk.pp_d[type].item[DENSITY] == 0x0 && disk.pp_d[type].item[RADIUS] != 0x0)
					{
						sim_data->h_p[bodyIdx].density = tools::calc_density(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].radius);
					}
					if (disk.pp_d[type].item[RADIUS] == 0x0 && disk.pp_d[type].item[DENSITY] != 0x0)
					{
						sim_data->h_p[bodyIdx].radius = tools::calc_radius(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].density);
					}
				}
			}
		}
	}

	// Calculate coordinates, velocities and minimal orbital period from the orbital elements
	ttt_t dt = 0.0;
	calculate_phase(sim_data, n_body, dt);

	tools::transform_to_bc(n_body, false, sim_data);

	print_all_input_data(dir, filename, n_body, seed, t0, dt, disk, sim_data);

	deallocate_host_storage(sim_data);

	delete sim_data;
}

void Dvorak(string& dir, string& filename)
{
	// Epoch for the disk's state
	ttt_t t0 = 0.0;

	body_disk_t disk;
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

	uint32_t seed = set_parameters::Dvorak(mmsn, disk);

	pp_disk_t::sim_data_t* sim_data = new pp_disk_t::sim_data_t;
	uint32_t n_body = calc_number_of_bodies(disk);
	allocate_host_storage(sim_data, n_body);

	populate_disk(t0, disk, sim_data);

	// Scale the masses in order to get the required mass transform_mass()
	{
		var_t m_total_pp = tools::get_total_mass(n_body, BODY_TYPE_PROTOPLANET, sim_data);
		var_t f = m_solid / m_total_pp;
		for (uint32_t i = 0; i < n_body; i++)
		{
			// Only the masses of the protoplanets will be scaled
			if (sim_data->h_body_md[i].body_type == BODY_TYPE_PROTOPLANET)
			{
				sim_data->h_p[i].mass *= f;
			}
		}
		m_total_pp = tools::get_total_mass(n_body, BODY_TYPE_PROTOPLANET, sim_data);
		if (fabs(m_total_pp - m_solid) > 1.0e-15)
		{
			cerr << "The required mass was not reached." << endl;
			exit(0);
		}
	}

	// Computes the physical quantities with the new mass
	{
		uint32_t bodyIdx = 0;
		for (int type = BODY_TYPE_STAR; type < BODY_TYPE_N; type++)
		{
			for (int i = 0; i < disk.nBody[type]; i++, bodyIdx++)
			{
				if (sim_data->h_p[bodyIdx].mass > 0.0)
				{
					if (disk.pp_d[type].item[DENSITY] == 0x0 && disk.pp_d[type].item[RADIUS] != 0x0)
					{
						sim_data->h_p[bodyIdx].density = tools::calc_density(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].radius);
					}
					if (disk.pp_d[type].item[RADIUS] == 0x0 && disk.pp_d[type].item[DENSITY] != 0x0)
					{
						sim_data->h_p[bodyIdx].radius = tools::calc_radius(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].density);
					}
				}
			}
		}
	}

	ttt_t dt = 0.0;
	calculate_phase(sim_data, n_body, dt);

	tools::transform_to_bc(n_body, false, sim_data);

	print_all_input_data(dir, filename, n_body, seed, t0, dt, disk, sim_data);

	deallocate_host_storage(sim_data);

	delete sim_data;
}

void Hansen_2009(string& dir, string& filename)
{
	// Epoch for the disk's state
	ttt_t t0 = 0.0;

	body_disk_t disk;
	initialize(disk);

	uint32_t seed = set_parameters::Hansen_2009(disk);

	pp_disk_t::sim_data_t* sim_data = new pp_disk_t::sim_data_t;
	uint32_t n_body = calc_number_of_bodies(disk);
	allocate_host_storage(sim_data, n_body);

	populate_disk(t0, disk, sim_data);

	// Calculate coordinates, velocities and minimal orbital period from the orbital elements
	ttt_t dt = 0.0;
	{
		var_t min_P = DBL_MAX;
		// The mass of the central star
		var_t m0 = sim_data->h_p[0].mass;
		var4_t rVec = {0.0, 0.0, 0.0, 0.0};
		var4_t vVec = {0.0, 0.0, 0.0, 0.0};

		// The coordinates of the central star
		sim_data->h_y[0][0] = rVec;
		sim_data->h_y[1][0] = vVec;
		int gp_counter = 0;
		for (uint32_t i = 1; i < n_body; i++)
		{
			if (BODY_TYPE_GIANTPLANET == sim_data->h_body_md[i].body_type && gp_counter < 1)
			{				
				if (0 == gp_counter)
				{
					disk.names[i] = "Jupiter";
					sim_data->h_p[i].mass          = 1.0 * constants::JupiterToSolar;
					sim_data->h_p[i].radius        = 71492.0 * constants::KilometerToAu;
					sim_data->h_p[i].density       = tools::calc_density(sim_data->h_p[i].mass, sim_data->h_p[i].radius);
					ttt_t epoch = extract_from_horizon_output(ephemeris_major_planets::date_20150511::jupiter_oe, sim_data->h_oe[i]);
				}
				gp_counter++;
			}
			var_t mu = K2 *(m0 + sim_data->h_p[i].mass);
			tools::calc_phase(mu, &sim_data->h_oe[i], &rVec, &vVec);
			sim_data->h_y[0][i] = rVec;
			sim_data->h_y[1][i] = vVec;

			ttt_t P = tools::calc_orbital_period(mu, sim_data->h_oe[i].sma);
			if (min_P > P)
			{
				min_P = P;
			}
		}
		dt = min_P / 1000.0;
	}

	tools::transform_to_bc(n_body, false, sim_data);

	print_all_input_data(dir, filename, n_body, seed, t0, dt, disk, sim_data);

	deallocate_host_storage(sim_data);

	delete sim_data;
}

void GT_scenario(string& dir, string& filename)
{
	// Epoch for the disk's state
	ttt_t t0 = 0.0;

	body_disk_t disk;
	initialize(disk);

	uint32_t seed = set_parameters::GT_scenario(disk);

	pp_disk_t::sim_data_t* sim_data = new pp_disk_t::sim_data_t;
	uint32_t n_body = calc_number_of_bodies(disk);
	allocate_host_storage(sim_data, n_body);

	populate_disk(t0, disk, sim_data);

	// Scale the masses in order to get the required mass transform_mass()
	//{
	//	var_t m_total_pp = tools::get_total_mass(n_body, BODY_TYPE_PROTOPLANET, sim_data);
	//	var_t f = m_solid / m_total_pp;
	//	for (uint32_t i = 0; i < n_body; i++)
	//	{
	//		// Only the masses of the protoplanets will be scaled
	//		if (sim_data->h_body_md[i].body_type == BODY_TYPE_PROTOPLANET)
	//		{
	//			sim_data->h_p[i].mass *= f;
	//		}
	//	}
	//	m_total_pp = tools::get_total_mass(n_body, BODY_TYPE_PROTOPLANET, sim_data);
	//	if (fabs(m_total_pp - m_solid) > 1.0e-15)
	//	{
	//		cerr << "The required mass was not reached." << endl;
	//		exit(0);
	//	}
	//}

	// Computes the physical quantities with the new mass
	{
		uint32_t bodyIdx = 0;
		for (int type = BODY_TYPE_STAR; type < BODY_TYPE_N; type++)
		{
			for (int i = 0; i < disk.nBody[type]; i++, bodyIdx++)
			{
				if (sim_data->h_p[bodyIdx].mass > 0.0)
				{
					if (disk.pp_d[type].item[DENSITY] == 0x0 && disk.pp_d[type].item[RADIUS] != 0x0)
					{
						sim_data->h_p[bodyIdx].density = tools::calc_density(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].radius);
					}
					if (disk.pp_d[type].item[RADIUS] == 0x0 && disk.pp_d[type].item[DENSITY] != 0x0)
					{
						sim_data->h_p[bodyIdx].radius = tools::calc_radius(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].density);
					}
				}
			}
		}
	}

	// Calculate coordinates, velocities and minimal orbital period from the orbital elements
	ttt_t dt = 0.0;
	{
		var_t min_P = DBL_MAX;
		// The mass of the central star
		var_t m0 = sim_data->h_p[0].mass;
		var4_t rVec = {0.0, 0.0, 0.0, 0.0};
		var4_t vVec = {0.0, 0.0, 0.0, 0.0};

		// The coordinates of the central star
		sim_data->h_y[0][0] = rVec;
		sim_data->h_y[1][0] = vVec;
		int gp_counter = 0;
		for (uint32_t i = 1; i < n_body; i++)
		{
			if (BODY_TYPE_GIANTPLANET == sim_data->h_body_md[i].body_type && gp_counter < 2)
			{				
				if (0 == gp_counter)
				{
					disk.names[i] = "Jupiter";
					sim_data->h_p[i].mass          = 1.0 * constants::JupiterToSolar;
					sim_data->h_p[i].radius        = 71492.0 * constants::KilometerToAu;
					sim_data->h_p[i].density       = tools::calc_density(sim_data->h_p[i].mass, sim_data->h_p[i].radius);
					ttt_t epoch = extract_from_horizon_output(ephemeris_major_planets::date_20150511::jupiter_oe, sim_data->h_oe[i]);
				}
				if (1 == gp_counter)
				{
					disk.names[i] = "Saturn";
					sim_data->h_p[i].mass          = 1.0 * constants::SaturnToSolar;
					sim_data->h_p[i].radius        = 60268.0 * constants::KilometerToAu;
					sim_data->h_p[i].density       = tools::calc_density(sim_data->h_p[i].mass, sim_data->h_p[i].radius);
					ttt_t epoch = extract_from_horizon_output(ephemeris_major_planets::date_20150511::saturn_oe, sim_data->h_oe[i]);
				}
				gp_counter++;
			}
			if (0.4 > sim_data->h_oe[i].sma)
			{
				sim_data->h_oe[i].sma = 0.4;
			}

			var_t mu = K2 *(m0 + sim_data->h_p[i].mass);
			tools::calc_phase(mu, &sim_data->h_oe[i], &rVec, &vVec);
			sim_data->h_y[0][i] = rVec;
			sim_data->h_y[1][i] = vVec;

			ttt_t P = tools::calc_orbital_period(mu, sim_data->h_oe[i].sma);
			if (min_P > P)
			{
				min_P = P;
			}
		}
		dt = min_P / 1000.0;
	}

	tools::transform_to_bc(n_body, false, sim_data);

	print_all_input_data(dir, filename, n_body, seed, t0, dt, disk, sim_data);

	deallocate_host_storage(sim_data);

	delete sim_data;
}

void GT_scenario_mod(string& dir, string& filename)
{
	// Epoch for the disk's state
	ttt_t t0 = 0.0;

	body_disk_t disk;
	initialize(disk);

	uint32_t seed = set_parameters::GT_scenario_mod(disk);

	pp_disk_t::sim_data_t* sim_data = new pp_disk_t::sim_data_t;
	uint32_t n_body = calc_number_of_bodies(disk);
	allocate_host_storage(sim_data, n_body);

	populate_disk(t0, disk, sim_data);

	// Scale the masses in order to get the required mass transform_mass()
	//{
	//	var_t m_total_pp = tools::get_total_mass(n_body, BODY_TYPE_PROTOPLANET, sim_data);
	//	var_t f = m_solid / m_total_pp;
	//	for (uint32_t i = 0; i < n_body; i++)
	//	{
	//		// Only the masses of the protoplanets will be scaled
	//		if (sim_data->h_body_md[i].body_type == BODY_TYPE_PROTOPLANET)
	//		{
	//			sim_data->h_p[i].mass *= f;
	//		}
	//	}
	//	m_total_pp = tools::get_total_mass(n_body, BODY_TYPE_PROTOPLANET, sim_data);
	//	if (fabs(m_total_pp - m_solid) > 1.0e-15)
	//	{
	//		cerr << "The required mass was not reached." << endl;
	//		exit(0);
	//	}
	//}

	// Computes the physical quantities with the new mass
	{
		uint32_t bodyIdx = 0;
		for (int type = BODY_TYPE_STAR; type < BODY_TYPE_N; type++)
		{
			for (int i = 0; i < disk.nBody[type]; i++, bodyIdx++)
			{
				if (sim_data->h_p[bodyIdx].mass > 0.0)
				{
					if (disk.pp_d[type].item[DENSITY] == 0x0 && disk.pp_d[type].item[RADIUS] != 0x0)
					{
						sim_data->h_p[bodyIdx].density = tools::calc_density(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].radius);
					}
					if (disk.pp_d[type].item[RADIUS] == 0x0 && disk.pp_d[type].item[DENSITY] != 0x0)
					{
						sim_data->h_p[bodyIdx].radius = tools::calc_radius(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].density);
					}
				}
			}
		}
	}

	// Calculate coordinates, velocities and minimal orbital period from the orbital elements
	ttt_t dt = 0.0;
	{
		var_t min_P = DBL_MAX;
		// The mass of the central star
		var_t m0 = sim_data->h_p[0].mass;
		var4_t rVec = {0.0, 0.0, 0.0, 0.0};
		var4_t vVec = {0.0, 0.0, 0.0, 0.0};

		// The coordinates of the central star
		sim_data->h_y[0][0] = rVec;
		sim_data->h_y[1][0] = vVec;
		int gp_counter = 0;
		for (uint32_t i = 1; i < n_body; i++)
		{
			if (BODY_TYPE_GIANTPLANET == sim_data->h_body_md[i].body_type && gp_counter < 3)
			{				
				if (0 == gp_counter)
				{
					disk.names[i] = "Jupiter";
					sim_data->h_p[i].mass          = 1.0 * constants::JupiterToSolar;
					sim_data->h_p[i].radius        = 71492.0 * constants::KilometerToAu;
					sim_data->h_p[i].density       = tools::calc_density(sim_data->h_p[i].mass, sim_data->h_p[i].radius);
					ttt_t epoch = extract_from_horizon_output(ephemeris_major_planets::date_20150511::jupiter_oe, sim_data->h_oe[i]);
				}
				if (1 == gp_counter)
				{
					disk.names[i] = "Saturn";
					sim_data->h_p[i].mass          = 1.0 * constants::SaturnToSolar;
					sim_data->h_p[i].radius        = 60268.0 * constants::KilometerToAu;
					sim_data->h_p[i].density       = tools::calc_density(sim_data->h_p[i].mass, sim_data->h_p[i].radius);
					ttt_t epoch = extract_from_horizon_output(ephemeris_major_planets::date_20150511::saturn_oe, sim_data->h_oe[i]);
				}
				if (2 == gp_counter)
				{
					disk.names[i] = "Uranus";
					sim_data->h_p[i].mass          = 1.0 * constants::UranusToSolar;
					sim_data->h_p[i].radius        = 25362.0 * constants::KilometerToAu;
					sim_data->h_p[i].density       = tools::calc_density(sim_data->h_p[i].mass, sim_data->h_p[i].radius);
					ttt_t epoch = extract_from_horizon_output(ephemeris_major_planets::date_20150511::uranus_oe, sim_data->h_oe[i]);
				}
				gp_counter++;
			}
			if (0.4 > sim_data->h_oe[i].sma)
			{
				sim_data->h_oe[i].sma = 0.4;
			}

			var_t mu = K2 *(m0 + sim_data->h_p[i].mass);
			tools::calc_phase(mu, &sim_data->h_oe[i], &rVec, &vVec);
			sim_data->h_y[0][i] = rVec;
			sim_data->h_y[1][i] = vVec;
			ttt_t P = tools::calc_orbital_period(mu, sim_data->h_oe[i].sma);
			if (min_P > P)
			{
				min_P = P;
			}
		}
		dt = min_P / 1000.0;
	}

	tools::transform_to_bc(n_body, false, sim_data);

	print_all_input_data(dir, filename, n_body, seed, t0, dt, disk, sim_data);

	deallocate_host_storage(sim_data);

	delete sim_data;
} // end create_GT_scenariomod

void two_body(string& dir, string& filename)
{
	// Epoch for the disk's state
	ttt_t t0 = 0.0;

	body_disk_t disk;
	initialize(disk);

	uint32_t seed = set_parameters::Two_body(disk);

	pp_disk_t::sim_data_t* sim_data = new pp_disk_t::sim_data_t;
	uint32_t n_body = calc_number_of_bodies(disk);
	allocate_host_storage(sim_data, n_body);

	populate_disk(t0, disk, sim_data);

	ttt_t dt = 0.0;
	calculate_phase(sim_data, n_body, dt);

	tools::transform_to_bc(n_body, false, sim_data);

	print_all_input_data(dir, filename, n_body, seed, t0, dt, disk, sim_data);

	deallocate_host_storage(sim_data);

	delete sim_data;
}

void n_gp(string& dir, string& filename)
{
	// Epoch for the disk's state
	ttt_t t0 = 0.0;

	body_disk_t disk;
	initialize(disk);

	uint32_t seed = set_parameters::n_gp(disk);

	pp_disk_t::sim_data_t* sim_data = new pp_disk_t::sim_data_t;
	uint32_t n_body = calc_number_of_bodies(disk);
	allocate_host_storage(sim_data, n_body);

	populate_disk(t0, disk, sim_data);

	// Computes the physical quantities with the new mass
	{
		uint32_t bodyIdx = 0;
		for (int type = BODY_TYPE_STAR; type < BODY_TYPE_N; type++)
		{
			for (int i = 0; i < disk.nBody[type]; i++, bodyIdx++)
			{
				if (sim_data->h_p[bodyIdx].mass > 0.0)
				{
					if (disk.pp_d[type].item[DENSITY] == 0x0 && disk.pp_d[type].item[RADIUS] != 0x0)
					{
						sim_data->h_p[bodyIdx].density = tools::calc_density(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].radius);
					}
					if (disk.pp_d[type].item[RADIUS] == 0x0 && disk.pp_d[type].item[DENSITY] != 0x0)
					{
						sim_data->h_p[bodyIdx].radius = tools::calc_radius(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].density);
					}
				}
			}
		}
	}

	ttt_t dt = 0.0;
	calculate_phase(sim_data, n_body, dt);

	tools::transform_to_bc(n_body, false, sim_data);

	print_all_input_data(dir, filename, n_body, seed, t0, dt, disk, sim_data);

	deallocate_host_storage(sim_data);

	delete sim_data;
}

void n_pp(string& dir, string& filename)
{
	// Epoch for the disk's state
	ttt_t t0 = 0.0;

	body_disk_t disk;
	initialize(disk);

	uint32_t seed = set_parameters::n_pp(disk);

	pp_disk_t::sim_data_t* sim_data = new pp_disk_t::sim_data_t;
	uint32_t n_body = calc_number_of_bodies(disk);
	allocate_host_storage(sim_data, n_body);

	populate_disk(t0, disk, sim_data);

	// Computes the physical quantities with the new mass
	//{
	//	uint32_t bodyIdx = 0;
	//	for (int type = BODY_TYPE_STAR; type < BODY_TYPE_N; type++)
	//	{
	//		for (int i = 0; i < disk.nBody[type]; i++, bodyIdx++)
	//		{
	//			if (sim_data->h_p[bodyIdx].mass > 0.0)
	//			{
	//				if (disk.pp_d[type].item[DENSITY] == 0x0 && disk.pp_d[type].item[RADIUS] != 0x0)
	//				{
	//					sim_data->h_p[bodyIdx].density = tools::calc_density(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].radius);
	//				}
	//				if (disk.pp_d[type].item[RADIUS] == 0x0 && disk.pp_d[type].item[DENSITY] != 0x0)
	//				{
	//					sim_data->h_p[bodyIdx].radius = tools::calc_radius(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].density);
	//				}
	//			}
	//		}
	//	}
	//}

	ttt_t dt = 0.0;
	calculate_phase(sim_data, n_body, dt);

	tools::transform_to_bc(n_body, false, sim_data);

	print_all_input_data(dir, filename, n_body, seed, t0, dt, disk, sim_data);

	deallocate_host_storage(sim_data);

	delete sim_data;
}

void n_spl(string& dir, string& filename)
{
	// Epoch for the disk's state
	ttt_t t0 = 0.0;

	body_disk_t disk;
	initialize(disk);

	uint32_t seed = set_parameters::n_spl(disk);

	pp_disk_t::sim_data_t* sim_data = new pp_disk_t::sim_data_t;
	uint32_t n_body = calc_number_of_bodies(disk);
	allocate_host_storage(sim_data, n_body);

	populate_disk(t0, disk, sim_data);

	ttt_t dt = 0.0;
	calculate_phase(sim_data, n_body, dt);

	tools::transform_to_bc(n_body, false, sim_data);

	print_all_input_data(dir, filename, n_body, seed, t0, dt, disk, sim_data);

	deallocate_host_storage(sim_data);

	delete sim_data;
}

void n_pl(string& dir, string& filename)
{
	// Epoch for the disk's state
	ttt_t t0 = 0.0;

	body_disk_t disk;
	initialize(disk);

	uint32_t seed = set_parameters::n_pl(disk);

	pp_disk_t::sim_data_t* sim_data = new pp_disk_t::sim_data_t;
	uint32_t n_body = calc_number_of_bodies(disk);
	allocate_host_storage(sim_data, n_body);

	populate_disk(t0, disk, sim_data);

	// Computes the physical quantities with the new mass
	{
		uint32_t bodyIdx = 0;
		for (int type = BODY_TYPE_STAR; type < BODY_TYPE_N; type++)
		{
			for (int i = 0; i < disk.nBody[type]; i++, bodyIdx++)
			{
				if (sim_data->h_p[bodyIdx].mass > 0.0)
				{
					if (disk.pp_d[type].item[DENSITY] == 0x0 && disk.pp_d[type].item[RADIUS] != 0x0)
					{
						sim_data->h_p[bodyIdx].density = tools::calc_density(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].radius);
					}
					if (disk.pp_d[type].item[RADIUS] == 0x0 && disk.pp_d[type].item[DENSITY] != 0x0)
					{
						sim_data->h_p[bodyIdx].radius = tools::calc_radius(sim_data->h_p[bodyIdx].mass, sim_data->h_p[bodyIdx].density);
					}
				}
			}
		}
	}

	ttt_t dt = 0.0;
	calculate_phase(sim_data, n_body, dt);

	tools::transform_to_bc(n_body, false, sim_data);

	print_all_input_data(dir, filename, n_body, seed, t0, dt, disk, sim_data);

	deallocate_host_storage(sim_data);

	delete sim_data;
}

void n_tp(string& dir, string& filename)
{
	// Epoch for the disk's state
	ttt_t t0 = 0.0;

	body_disk_t disk;
	initialize(disk);

	uint32_t seed = set_parameters::n_tp(disk);

	pp_disk_t::sim_data_t* sim_data = new pp_disk_t::sim_data_t;
	uint32_t n_body = calc_number_of_bodies(disk);
	allocate_host_storage(sim_data, n_body);

	populate_disk(t0, disk, sim_data);

	ttt_t dt = 0.0;
	calculate_phase(sim_data, n_body, dt);

	print_all_input_data(dir, filename, n_body, seed, t0, dt, disk, sim_data);

	deallocate_host_storage(sim_data);

	delete sim_data;
}

void n_pl_to_test_anal_gd(string& dir, string& filename)
{
	// Epoch for the disk's state
	ttt_t t0 = 0.0;

	body_disk_t disk;
	initialize(disk);

	uint32_t seed = set_parameters::pl_to_test_anal_gd(disk);

	pp_disk_t::sim_data_t* sim_data = new pp_disk_t::sim_data_t;
	uint32_t n_body = calc_number_of_bodies(disk);
	allocate_host_storage(sim_data, n_body);

	populate_disk(t0, disk, sim_data);

	ttt_t dt = 0.0;
	calculate_phase(sim_data, n_body, dt);

	tools::transform_to_bc(n_body, false, sim_data);

	print_all_input_data(dir, filename, n_body, seed, t0, dt, disk, sim_data);

	deallocate_host_storage(sim_data);

	delete sim_data;
}

void solar_system(string& dir, string& filename)
{
	// Epoch for the disk's state
	ttt_t t0 = 0.0;

	body_disk_t disk;
	initialize(disk);

	uint32_t seed = set_parameters::solar_system(disk);

	pp_disk_t::sim_data_t* sim_data = new pp_disk_t::sim_data_t;
	uint32_t n_body = calc_number_of_bodies(disk);
	allocate_host_storage(sim_data, n_body);

	populate_solar_system(disk, sim_data);

	ttt_t dt = 0.0;
	calculate_phase(sim_data, n_body, dt);

	tools::transform_to_bc(n_body, false, sim_data);

	print_all_input_data(dir, filename, n_body, seed, t0, dt, disk, sim_data);

	deallocate_host_storage(sim_data);

	delete sim_data;
}
} /* create_disk */


namespace project_collision_2D
{
void create_init_cond(string& out_dir)
{
	char buffer[4];
	// Iterates over the different initial seed
	for (uint32_t j = 0; j < 10; j++)
	{
		sprintf(buffer, "%02d", j+1);
		string postfix(buffer);
		string filename = "input_" + postfix;
		
		create_disk::n_pp(out_dir, filename);
		// This is needed to initialize the built-in random number generator with a new seed
		delay(1600);
	}
}
} /* project_collision_2D */

namespace project_collision_Rezso_2D
{
void create_init_cond(string& out_dir)
{
	char buffer[4];
	// Iterates over the different initial seed
	for (uint32_t j = 0; j < 1; j++)
	{
		sprintf(buffer, "%02d", j+1);
		string postfix(buffer);
		string filename = "input_" + postfix;
		
		create_disk::coll_stat_run(out_dir, filename);
		// This is needed to initialize the built-in random number generator with a new seed
		delay(1600);
	}
}
} /* project_collision_Rezso_2D */

int parse_options(int argc, const char **argv, string &outDir, string &filename)
{
	int i = 1;

	while (i < argc)
	{
		string p = argv[i];

		if (     p == "-o")
		{
			i++;
			outDir = argv[i];
		}
		else if (p == "-f")
		{
			i++;
			filename = argv[i];
		}
		else
		{
			throw string("Invalid switch on command-line.");
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
	string outDir;
	string filename;
	string output_path;

	parse_options(argc, argv, outDir, filename);

#if 0
	{
		string out_dir = "C:\\Work\\red.cuda.Results\\CollisionStatistics\\2D";
		project_collision_2D::create_init_cond(out_dir);
		return (EXIT_SUCCESS);
	}
#endif	

#if 1
	{
		project_collision_Rezso_2D::create_init_cond(outDir);
		return (EXIT_SUCCESS);
	}
#endif	


	try
	{
		//create_disk::solar_system(outDir, filename);
		//create_disk::Hansen_2009(outDir, filename);
		//create_disk::Chambers2001(outDir, filename);
		//create_disk::n_pl_to_test_anal_gd(outDir, filename);
		//create_disk::n_gp(outDir, filename);
		//create_disk::n_tp(outDir, filename);
		//create_disk::n_pl(outDir, filename);
		//create_disk::n_spl(outDir, filename);
		//create_disk::n_pp(outDir, filename);
		create_disk::two_body(outDir, filename);
		//create_disk::Dvorak(outDir, filename);
		//create_disk::GT_scenario(outDir, filename);
		//create_disk::GT_scenario_mod(outDir, filename);
	}
	catch (const string& msg)
	{
		cerr << "Error: " << msg << endl;
		return (EXIT_FAILURE);
	}

	return (EXIT_SUCCESS);
}
