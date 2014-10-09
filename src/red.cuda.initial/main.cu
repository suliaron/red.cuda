// includes, system 
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>

// includes, project
#include "pp_disk.h"
#include "red_constants.h"
#include "red_type.h"
#include "redutilcu.h"

#define MASS_STAR		1.0			// M_sol
#define MASS_JUPITER	1.0e-3		// M_sol
#define RAD_STAR		0.005		// AU

#define MASS_SUN		1.9891E+30	// kg
#define MASS_FACTOR		5e-7		// M_sol
#define MASS_MU			log(4.0)
#define MASS_S			0.3
#define MASS_MIN		1.0e-20		// M_sol
#define MASS_MAX		1.0e-19		// M_sol

#define DIST_MIN		4.5			// AU
#define DIST_MAX		15			// AU

//#define DENSITY			3000.0		// kg m-3
#define AU				149.6e9		// m

// It must be enclosed in parentheses in order to give correct results in
// the case of a division i.e. 1/SQR(x) -> 1/((x)*(x))
#define	SQR(x)			((x)*(x))
#define	CUBE(x)			((x)*(x)*(x))

typedef double ioT;

using namespace std;
using namespace redutilcu;


static string body_type_names[] = {"star", "giant", "rocky", "proto", "superpl", "pl", "testp"}; 

typedef struct orbelem
		{
			//! Semimajor-axis of the body
			var_t sma;
			//! Eccentricity of the body
			var_t ecc;
			//! Inclination of the body
			var_t inc;
			//! Argument of the pericenter
			var_t peri;
			//! Longitude of the ascending node
			var_t node;
			//! Mean anomaly
			var_t mean;
		} orbelem_t;

typedef enum orbelem_name
		{
			ORBITAL_ELEMENT_SMA,
			ORBITAL_ELEMENT_ECC,
			ORBITAL_ELEMENT_INC,
			ORBITAL_ELEMENT_PERI,
			ORBITAL_ELEMENT_NODE,
			ORBITAL_ELEMENT_MEAN,
            ORBITAL_ELEMENT_N
		} orbelem_name_t;

typedef enum phys_prop_name
	    {
		    MASS,
		    RADIUS,
		    DENSITY,
		    DRAG_COEFF
	    } phys_prop_name_t;

typedef struct distribution
		{
			var2_t	limits;
			var_t	(*pdf)(var_t);
		} distribution_t;

typedef struct oe_dist
		{
			distribution_t	item[ORBITAL_ELEMENT_N];
		} oe_dist_t;

typedef struct phys_prop_dist
		{
			distribution_t	item[4];
		} phys_prop_dist_t;

typedef struct body_disk
		{
			vector<string>		names;
			int_t				nBody[BODY_TYPE_N];
			oe_dist_t			oe_d[BODY_TYPE_N];
			phys_prop_dist_t	pp_d[BODY_TYPE_N];
			migration_type_t	*mig_type;
			var_t				*stop_at;
		} body_disk_t;


// Draw a number from a given distribution
var_t generate_random(var_t xmin, var_t xmax, var_t p(var_t))
{
	var_t x;
	var_t y;

	do
	{
		x = xmin + (var_t)rand() / RAND_MAX * (xmax - xmin);
		y = (var_t)rand() / RAND_MAX;
	}
	while (y > p(x));

	return x;
}

var_t pdf_mass_lognormal(var_t x)
{
	return 1.0 / sqrt(2 * PI) / MASS_S * exp(-pow(log(x) - MASS_MU, 2) / 2 / MASS_S / MASS_MU);
}

var_t pdf_distance_squared(var_t d)
{
	return d * d / DIST_MAX / DIST_MAX;
}

var_t pdf_distance_exp(var_t d)
{
	return exp(-d) * d * d;
}

var_t pdf_const(var_t x)
{
	return 1;
}

#define FOUR_PI_OVER_THREE	4.1887902047863909846168578443727
var_t calculate_radius(var_t m, var_t density)
{
	return pow(1.0/FOUR_PI_OVER_THREE * m/density, 1.0/3.0);
}

var_t calculate_density(var_t m, var_t R)
{
	if (R == 0.0)
	{
		return 0.0;
	}
	return m / (FOUR_PI_OVER_THREE * CUBE(R));
}

var_t caclulate_mass(var_t R, var_t density)
{
	return FOUR_PI_OVER_THREE * CUBE(R) * density;
}
#undef FOUR_PI_OVER_THREE

int_t	kepler_equation_solver(var_t ecc, var_t mean, var_t eps, var_t* E)
{
	if (ecc == 0.0 || mean == 0.0 || mean == PI) {
        *E = mean;
		return 0;
    }
    *E = mean + ecc * (sin(mean)) / (1.0 - sin(mean + ecc) + sin(mean));
    var_t E1 = 0.0;
    var_t error;
    int_t step = 0;
    do {
        E1 = *E - (*E - ecc * sin(*E) - mean) / (1.0 - ecc * cos(*E));
        error = fabs(E1 - *E);
        *E = E1;
    } while (error > eps && step++ <= 15);
	if (step > 15 ) {
		return 1;
	}

	return 0;
}

int_t	calculate_phase(var_t mu, const orbelem_t* oe, vec_t* rVec, vec_t* vVec)
{
    var_t ecc = oe->ecc;
	var_t E = 0.0;
	if (kepler_equation_solver(ecc, oe->mean, 1.0e-14, &E) == 1) {
		return 1;
	}
    var_t v = 2.0 * atan(sqrt((1.0 + ecc) / (1.0 - ecc)) * tan(E / 2.0));

    var_t p = oe->sma * (1.0 - SQR(ecc));
    var_t r = p / (1.0 + ecc * cos(v));
    var_t kszi = r * cos(v);
    var_t eta = r * sin(v);
    var_t vKszi = -sqrt(mu / p) * sin(v);
    var_t vEta = sqrt(mu / p) * (ecc + cos(v));

    var_t cw = cos(oe->peri);
    var_t sw = sin(oe->peri);
    var_t cO = cos(oe->node);
    var_t sO = sin(oe->node);
    var_t ci = cos(oe->inc);
    var_t si = sin(oe->inc);

    vec_t P;
	P.x = cw * cO - sw * sO * ci;
	P.y = cw * sO + sw * cO * ci;
	P.z = sw * si;
    vec_t Q;
	Q.x = -sw * cO - cw * sO * ci;
	Q.y = -sw * sO + cw * cO * ci;
	Q.z = cw * si;

	rVec->x = kszi * P.x + eta * Q.x;
	rVec->y = kszi * P.y + eta * Q.y;
	rVec->z = kszi * P.z + eta * Q.z;

	vVec->x = vKszi * P.x + vEta * Q.x;
	vVec->y = vKszi * P.y + vEta * Q.y;
	vVec->z = vKszi * P.z + vEta * Q.z;

	return 0;
}

void print_body_record(
    ofstream &output, 
    string name, 
    var_t epoch, 
    param_t *param, 
    body_metadata_t *body_md,
    vec_t *r, vec_t *v, int precision)
{
	static char sep = ' ';

	output << setw(6) << body_md->id << sep
			<< setw(16) << name << sep 
			<< setw(3) << body_md->body_type << sep 
			<< setw(20) << epoch << sep;
	output << setw(precision + 6) << setprecision(precision) << param->mass << sep 
			<< setw(precision + 6) << param->radius << sep 
			<< setw(precision + 6) << param->density << sep 
			<< setw(precision + 6) << param->cd << sep;
	output << setw(3) << body_md->mig_type << sep
			<< setw(precision + 6) << body_md->mig_stop_at << sep;
	output << setw(precision + 6) << r->x << sep 
			<< setw(precision + 6) << r->y << sep
			<< setw(precision + 6) << r->z << sep;
	output << setw(precision + 6) << v->x << sep
			<< setw(precision + 6) << v->y << sep
			<< setw(precision + 6) << v->z << sep;
	output << endl;

    output.flush();
}

void Emese_data_format_to_red_cuda_format(const string& input_path, const string& output_path)
{
	ifstream input(input_path, ios::in | ios::binary);
	ofstream output(output_path, ios_base::out);

	if (!input)
	{
		cerr << "Cannot open " << input_path << "." << endl;
	}
	if (!output)
	{
		cerr << "Cannot open " << output_path << "." << endl;
	}

	output << "1 0 0 5000 0 0 5000" << endl;
	if (input && output) 
	{
		ioT time = 0;        
		int64_t nbodyfilein;
		int64_t lengthChar;      
		char buffer[64];
		ioT id = 0;
		string name;
		string reference;
		ioT x = 0;
		ioT y = 0;
		ioT z = 0;
		ioT vx = 0;
		ioT vy = 0;
		ioT vz = 0;
		ioT m = 0;
		ioT rad = 0;

		input.read(reinterpret_cast<char *>(&time), sizeof(time));
		input.read(reinterpret_cast<char *>(&nbodyfilein), sizeof(nbodyfilein));
		for (int i = 0; i < nbodyfilein; i++)
		{
			input.read(reinterpret_cast<char *>(&id), sizeof(id));

			lengthChar = 0;
			input.read(reinterpret_cast<char *>(&lengthChar), sizeof(lengthChar));            
			input.read(buffer, lengthChar);
			buffer[lengthChar] = 0;
			name = buffer;
			replace(name.begin(), name.end(), ' ', '_'); // replace all ' ' to '_'

			lengthChar = 0;
			input.read(reinterpret_cast<char *>(&lengthChar), sizeof(lengthChar));
			input.read(buffer, lengthChar);
			buffer[lengthChar] = 0;
			reference = buffer; 

			input.read(reinterpret_cast<char *>(& x), sizeof( x));
			input.read(reinterpret_cast<char *>(& y), sizeof( y));
			input.read(reinterpret_cast<char *>(& z), sizeof( z));
			input.read(reinterpret_cast<char *>(&vx), sizeof(vx));
			input.read(reinterpret_cast<char *>(&vy), sizeof(vy));
			input.read(reinterpret_cast<char *>(&vz), sizeof(vz));
			input.read(reinterpret_cast<char *>(& m), sizeof( m));
			input.read(reinterpret_cast<char *>(& rad), sizeof( rad));

			vec_t	rVec = {x, y, z, 0.0};
			vec_t	vVec = {vx, vy, vz, 0.0};

			param_t	        param;
            body_metadata_t body_md;

			// red.cuda: id starts from 1
			body_md.id = ++id;
			if (1 == body_md.id)
			{
				body_md.body_type = BODY_TYPE_STAR;
			}
			if (1 < body_md.id)
			{
				if (0.0 < m)
				{
					body_md.body_type = BODY_TYPE_PROTOPLANET;
				}
				else
				{
					body_md.body_type = BODY_TYPE_TESTPARTICLE;
				}
			}

			param.cd = 0.0;
			param.mass = m;
			param.radius = rad;
			param.density = calculate_density(m, rad);
			body_md.mig_stop_at = 0.0;
			body_md.mig_type = MIGRATION_TYPE_NO;

			print_body_record(output, name, time, &param,&body_md, &rVec, &vVec, 15);
		}
		input.close();
		output.close();
	}
	else
	{
		exit(1);
	}
}

void set(distribution_t& d, var_t x, var_t (*pdf)(var_t))
{
	d.limits.x = d.limits.y = x;
	d.pdf = pdf;
}

void set(distribution_t& d, var_t x, var_t y, var_t (*pdf)(var_t))
{
	d.limits.x = x;
	d.limits.y = y;
	d.pdf = pdf;
}

string create_name(int i, int type)
{
	ostringstream convert;	// stream used for the conversion
	string i_str;			// string which will contain the number
	string name;

	convert << i;			// insert the textual representation of 'i' in the characters in the stream
	i_str = convert.str();  // set 'i_str' to the contents of the stream
	name = body_type_names[type] + i_str;

	return name;
}

void set_default(body_disk &bd)
{
	for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_N; body_type++)
	{
		bd.nBody[body_type] = 0;
        for (int i = 0; i < ORBITAL_ELEMENT_N; i++) 
		{
			set(bd.oe_d[body_type].item[i], 0.0, 0.0, pdf_const);
		}
		for (int i = 0; i < 4; i++) 
		{
			set(bd.pp_d[body_type].item[i], 0.0, 0.0, pdf_const);
		}
	}
}

int_t	calculate_number_of_bodies(body_disk &bd)
{
	int_t result = 0;
	for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_N; body_type++)
	{
		result += bd.nBody[body_type];
	}
	return result;
}

void generate_oe(oe_dist_t *oe_d, orbelem_t& oe)
{
    oe.sma  = generate_random(oe_d->item[ORBITAL_ELEMENT_SMA].limits.x,  oe_d->item[ORBITAL_ELEMENT_SMA].limits.y,  oe_d->item[ORBITAL_ELEMENT_SMA].pdf);
	oe.ecc  = generate_random(oe_d->item[ORBITAL_ELEMENT_ECC].limits.x,  oe_d->item[ORBITAL_ELEMENT_ECC].limits.y,  oe_d->item[ORBITAL_ELEMENT_ECC].pdf);
	oe.inc  = generate_random(oe_d->item[ORBITAL_ELEMENT_INC].limits.x,  oe_d->item[ORBITAL_ELEMENT_INC].limits.y,  oe_d->item[ORBITAL_ELEMENT_INC].pdf);
	oe.peri = generate_random(oe_d->item[ORBITAL_ELEMENT_PERI].limits.x, oe_d->item[ORBITAL_ELEMENT_PERI].limits.y, oe_d->item[ORBITAL_ELEMENT_PERI].pdf);
	oe.node = generate_random(oe_d->item[ORBITAL_ELEMENT_NODE].limits.x, oe_d->item[ORBITAL_ELEMENT_NODE].limits.y, oe_d->item[ORBITAL_ELEMENT_NODE].pdf);
    oe.mean = generate_random(oe_d->item[ORBITAL_ELEMENT_MEAN].limits.x, oe_d->item[ORBITAL_ELEMENT_MEAN].limits.y, oe_d->item[ORBITAL_ELEMENT_MEAN].pdf);
}

void generate_pp(phys_prop_dist_t *pp_d, param_t& param)
{
	param.mass = generate_random(pp_d->item[MASS].limits.x, pp_d->item[MASS].limits.y, pp_d->item[MASS].pdf);

	if (	 pp_d->item[DENSITY].limits.x == 0.0 && pp_d->item[DENSITY].limits.y == 0.0 &&
			 pp_d->item[RADIUS].limits.x == 0.0 && pp_d->item[RADIUS].limits.y == 0.0 )
	{
		param.radius = 0.0;
		param.density = 0.0;
	}
	else if (pp_d->item[DENSITY].limits.x == 0.0 && pp_d->item[DENSITY].limits.y == 0.0 &&
			 pp_d->item[RADIUS].limits.x > 0.0 && pp_d->item[RADIUS].limits.y > 0.0 )
	{
		param.radius = generate_random(pp_d->item[RADIUS].limits.x, pp_d->item[RADIUS].limits.y, pp_d->item[RADIUS].pdf);
		param.density = calculate_density(param.mass, param.radius);
	}
	else if (pp_d->item[DENSITY].limits.x > 0.0 && pp_d->item[DENSITY].limits.y > 0.0 &&
			 pp_d->item[RADIUS].limits.x == 0.0 && pp_d->item[RADIUS].limits.y == 0.0 )
	{
		param.density = generate_random(pp_d->item[DENSITY].limits.x, pp_d->item[DENSITY].limits.y, pp_d->item[DENSITY].pdf);
		param.radius = calculate_radius(param.mass, param.density);
	}
	else
	{
		param.radius = generate_random(pp_d->item[RADIUS].limits.x, pp_d->item[RADIUS].limits.y, pp_d->item[RADIUS].pdf);
		param.density = generate_random(pp_d->item[DENSITY].limits.x, pp_d->item[DENSITY].limits.y, pp_d->item[DENSITY].pdf);
	}

	param.cd = generate_random(pp_d->item[DRAG_COEFF].limits.x, pp_d->item[DRAG_COEFF].limits.y, pp_d->item[DRAG_COEFF].pdf);
}

int generate_pp_disk(const string &path, body_disk_t& body_disk)
{
	static char sep = ' ';
	static const int precision = 10;

	ofstream	output(path, ios_base::out);
	if (output)
	{
		for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_N; body_type++)
		{
			output << body_disk.nBody[body_type] << sep;
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
			srand ((unsigned int)time(0));
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
					print_body_record(output, body_disk.names[bodyIdx], epoch, &param0, &body_md, &rVec, &vVec, precision);
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
					int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
					if (ret_code == 1) {
						cerr << "Could not calculate the phase." << endl;
						return ret_code;
					}

                    print_body_record(output, body_disk.names[bodyIdx], t, &param, &body_md, &rVec, &vVec, precision);
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

	set_default(disk);

	disk.nBody[BODY_TYPE_STAR       ] = 1;
	disk.nBody[BODY_TYPE_GIANTPLANET] = 1;

	int_t nBodies = calculate_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[nBodies];
	disk.stop_at = new var_t[nBodies];

    int bodyIdx = 0;
	int type = BODY_TYPE_STAR;

	disk.names.push_back("star");
	set(disk.pp_d[type].item[MASS], 1.0, pdf_const);
	set(disk.pp_d[type].item[RADIUS], 1.0 * constants::SolarRadiusToAu, pdf_const);
	set(disk.pp_d[type].item[DRAG_COEFF], 0.0, pdf_const);
	disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
	disk.stop_at[bodyIdx] = 0.0;

	type = BODY_TYPE_GIANTPLANET;
	{
        set(disk.oe_d[type].item[ORBITAL_ELEMENT_SMA], 1.0, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_ECC], 0.0, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_INC], 0.0, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_PERI], 0.0, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_NODE], 0.0, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN], 0.0, pdf_const);

		set(disk.pp_d[type].item[MASS], m1, pdf_const);
		set(disk.pp_d[type].item[DENSITY], 1.3 * constants::GramPerCm3ToSolarPerAu3, pdf_const);
		set(disk.pp_d[type].item[DRAG_COEFF], 0.0, pdf_const);

		for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	type = BODY_TYPE_ROCKYPLANET;

	type = BODY_TYPE_PROTOPLANET;

	type = BODY_TYPE_SUPERPLANETESIMAL;

	type = BODY_TYPE_PLANETESIMAL;

	type = BODY_TYPE_TESTPARTICLE;
}

void set_parameters_of_Dvorak_disk(body_disk_t& disk)
{
	const var_t mCeres	  = 9.43e20 /* kg */ * constants::KilogramToSolar;
	const var_t mMoon	  = 7.35e22 /* kg */ * constants::KilogramToSolar;
	const var_t rhoBasalt = 2.7 /* g/cm^3 */ * constants::GramPerCm3ToSolarPerAu3;

	set_default(disk);

	disk.nBody[BODY_TYPE_STAR        ] = 1;
	disk.nBody[BODY_TYPE_PROTOPLANET ] = 5000;
	disk.nBody[BODY_TYPE_TESTPARTICLE] = 5000;

	int_t nBodies = calculate_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[nBodies];
	disk.stop_at = new var_t[nBodies];

    int bodyIdx = 0;

    int type = BODY_TYPE_STAR;

	disk.names.push_back("star");
	set(disk.pp_d[type].item[MASS], 1.0, pdf_const);
	set(disk.pp_d[type].item[RADIUS], 1.0 * constants::SolarRadiusToAu, pdf_const);
	set(disk.pp_d[type].item[DRAG_COEFF], 0.0, pdf_const);
	disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
	disk.stop_at[bodyIdx] = 0.0;

	type = BODY_TYPE_GIANTPLANET;

	type = BODY_TYPE_ROCKYPLANET;

	type = BODY_TYPE_PROTOPLANET;
	{
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_SMA], 0.9, 2.5, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_ECC], 0.0, 0.3, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_INC], 0.0, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_PERI], 0.0, 360.0 * constants::DegreeToRadian, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_NODE], 0.0, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN], 0.0, 360.0 * constants::DegreeToRadian, pdf_const);

		set(disk.pp_d[type].item[MASS], mCeres, mMoon/10.0, pdf_mass_lognormal);
		set(disk.pp_d[type].item[DENSITY], rhoBasalt, pdf_const);
		set(disk.pp_d[type].item[DRAG_COEFF], 0.0, pdf_const);

        for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	type = BODY_TYPE_SUPERPLANETESIMAL;

	type = BODY_TYPE_PLANETESIMAL;

	type = BODY_TYPE_TESTPARTICLE;
	{
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_SMA], 0.5, 4.0, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_ECC], 0.0, 0.1, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_INC], 0.0, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_PERI], 0.0, 360.0 * constants::DegreeToRadian, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_NODE], 0.0, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN], 0.0, 360.0 * constants::DegreeToRadian, pdf_const);

		set(disk.pp_d[type].item[MASS], 0.0, pdf_const);
		set(disk.pp_d[type].item[DENSITY], 0.0, pdf_const);
		set(disk.pp_d[type].item[DRAG_COEFF], 0.0, pdf_const);

        for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}
}

void set_parameters_of_N1_massive_N3_test(body_disk_t& disk)
{
	const var_t mCeres	  = 9.43e20 /* kg */ * constants::KilogramToSolar;
	const var_t mMoon	  = 7.35e22 /* kg */ * constants::KilogramToSolar;
	const var_t rhoBasalt = 2.7 /* g/cm^3 */ * constants::GramPerCm3ToSolarPerAu3;

	set_default(disk);

	disk.nBody[BODY_TYPE_STAR        ] = 1;
	disk.nBody[BODY_TYPE_PROTOPLANET ] = 10;
	disk.nBody[BODY_TYPE_TESTPARTICLE] = 10;

	int_t nBodies = calculate_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[nBodies];
	disk.stop_at = new var_t[nBodies];

    int bodyIdx = 0;

    int type = BODY_TYPE_STAR;

	disk.names.push_back("star");
	set(disk.pp_d[type].item[MASS], 1.0, pdf_const);
	set(disk.pp_d[type].item[RADIUS], 1.0 * constants::SolarRadiusToAu, pdf_const);
	set(disk.pp_d[type].item[DRAG_COEFF], 0.0, pdf_const);
	disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
	disk.stop_at[bodyIdx] = 0.0;

	type = BODY_TYPE_GIANTPLANET;

	type = BODY_TYPE_ROCKYPLANET;

	type = BODY_TYPE_PROTOPLANET;
	{
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_SMA], 0.9, 2.5, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_ECC], 0.0, 0.3, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_INC], 0.0, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_PERI], 0.0, 360.0 * constants::DegreeToRadian, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_NODE], 0.0, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN], 0.0, 360.0 * constants::DegreeToRadian, pdf_const);

		set(disk.pp_d[type].item[MASS], mCeres, mMoon/10.0, pdf_mass_lognormal);
		set(disk.pp_d[type].item[DENSITY], rhoBasalt, pdf_const);
		set(disk.pp_d[type].item[DRAG_COEFF], 0.0, pdf_const);

        for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}

	type = BODY_TYPE_SUPERPLANETESIMAL;

	type = BODY_TYPE_PLANETESIMAL;

	type = BODY_TYPE_TESTPARTICLE;
	{
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_SMA], 0.9, 2.5, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_ECC], 0.0, 0.3, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_INC], 0.0, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_PERI], 0.0, 360.0 * constants::DegreeToRadian, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_NODE], 0.0, pdf_const);
		set(disk.oe_d[type].item[ORBITAL_ELEMENT_MEAN], 0.0, 360.0 * constants::DegreeToRadian, pdf_const);

		set(disk.pp_d[type].item[MASS], 0.0, pdf_const);
		set(disk.pp_d[type].item[DENSITY], 0.0, pdf_const);
		set(disk.pp_d[type].item[DRAG_COEFF], 0.0, pdf_const);

        for (int i = 0; i < disk.nBody[type]; i++) 
		{
            bodyIdx++;
			disk.names.push_back(create_name(i, type));
			disk.mig_type[bodyIdx] = MIGRATION_TYPE_NO;
			disk.stop_at[bodyIdx] = 0.0;
		}
	}
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

	{
		string input_path = file::combine_path(outDir, filename);
		string output_path = file::combine_path(outDir, file::get_filename_without_ext(filename) + ".txt");
		Emese_data_format_to_red_cuda_format(input_path, output_path);
		return 0;
	}

	//{
	//	set_parameters_of_Two_body_disk(disk);
	//	output_path = file::combine_path(outDir, filename);
	//	generate_pp_disk(output_path, disk);
	//	return 0;
	//}

	//{
	//	set_parameters_of_Dvorak_disk(disk);
	//	output_path = file::combine_path(outDir, filename);
	//	generate_pp_disk(output_path, disk);
	//	return 0;
	//}

	//{
	//	set_parameters_of_N1_massive_N3_test(disk);
	//	output_path = file::combine_path(outDir, filename);
	//	generate_pp_disk(output_path, disk);
 //   	return 0;
	//}


	ostringstream convert;	// stream used for the conversion
	string i_str;			// string which will contain the number
	string name;

	body_disk_t test_disk;
	set_default(test_disk);

	test_disk.nBody[BODY_TYPE_STAR] = 1;
	test_disk.nBody[BODY_TYPE_GIANTPLANET] = 2;
	test_disk.nBody[BODY_TYPE_ROCKYPLANET] = 0;
	test_disk.nBody[BODY_TYPE_PROTOPLANET] = 0;
	test_disk.nBody[BODY_TYPE_SUPERPLANETESIMAL] = 0;
	test_disk.nBody[BODY_TYPE_PLANETESIMAL] = 0;
	test_disk.nBody[BODY_TYPE_TESTPARTICLE] = 0;

	int_t nBodies = calculate_number_of_bodies(test_disk);
	test_disk.mig_type = new migration_type_t[nBodies];
	test_disk.stop_at = new var_t[nBodies];

	int	index_of_body = 0;

	test_disk.names.push_back("star");
	set(test_disk.pp_d[BODY_TYPE_STAR].item[MASS], 1.0, pdf_const);
	set(test_disk.pp_d[BODY_TYPE_STAR].item[RADIUS], 1.0 * constants::SolarRadiusToAu, pdf_const);
	set(test_disk.pp_d[BODY_TYPE_STAR].item[DRAG_COEFF], 0.0, pdf_const);
	test_disk.mig_type[index_of_body] = MIGRATION_TYPE_NO;
	test_disk.stop_at[index_of_body] = 0.0;

	index_of_body++;
	for (int i = 0; i < test_disk.nBody[BODY_TYPE_GIANTPLANET]; i++, index_of_body++)
	{
		convert << i;			// insert the textual representation of 'i' in the characters in the stream
		i_str = convert.str();  // set 'i_str' to the contents of the stream
		name = body_type_names[BODY_TYPE_GIANTPLANET] + i_str;		
		test_disk.names.push_back(name);
		convert.str("");

        set(test_disk.oe_d[BODY_TYPE_GIANTPLANET].item[ORBITAL_ELEMENT_SMA], 1.0, 4.0, pdf_const);
		set(test_disk.oe_d[BODY_TYPE_GIANTPLANET].item[ORBITAL_ELEMENT_ECC], 0.0, 0.2, pdf_const);
		set(test_disk.oe_d[BODY_TYPE_GIANTPLANET].item[ORBITAL_ELEMENT_INC], 0.0, 30.0 * constants::DegreeToRadian, pdf_const);
		set(test_disk.oe_d[BODY_TYPE_GIANTPLANET].item[ORBITAL_ELEMENT_PERI], 0.0, 360.0 * constants::DegreeToRadian, pdf_const);
		set(test_disk.oe_d[BODY_TYPE_GIANTPLANET].item[ORBITAL_ELEMENT_NODE], 0.0, 360.0 * constants::DegreeToRadian, pdf_const);
		set(test_disk.oe_d[BODY_TYPE_GIANTPLANET].item[ORBITAL_ELEMENT_MEAN], 0.0, 360.0 * constants::DegreeToRadian, pdf_const);

		set(test_disk.pp_d[BODY_TYPE_GIANTPLANET].item[MASS], 1.0 * constants::JupiterToSolar, 2.0 * constants::JupiterToSolar, pdf_mass_lognormal);
		set(test_disk.pp_d[BODY_TYPE_GIANTPLANET].item[DENSITY], 0.7 * constants::GramPerCm3ToSolarPerAu3, 1.2 * constants::GramPerCm3ToSolarPerAu3, pdf_mass_lognormal);
		set(test_disk.pp_d[BODY_TYPE_GIANTPLANET].item[DRAG_COEFF], 0.0, pdf_const);
		test_disk.mig_type[index_of_body] = MIGRATION_TYPE_TYPE_II;
		test_disk.stop_at[index_of_body] = 0.4;
	}

	outDir = "C:\\Work\\Projects\\solaris.cuda\\TestRun\\InputTest";
	generate_pp_disk(file::combine_path(outDir, "test_disk.txt"), test_disk);

	delete[] test_disk.mig_type;
	delete[] test_disk.stop_at;

	return 0;
}
