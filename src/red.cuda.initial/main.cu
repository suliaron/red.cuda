﻿// includes, system 
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
#include "red_constants.h"
#include "red_type.h"
#include "redutilcu.h"

using namespace std;
using namespace redutilcu;


static string body_type_names[] = {"star", "giant", "rocky", "proto", "superpl", "pl", "testp"}; 

namespace random_generator
{
	class red_random
	{
	public:
		red_random(unsigned int seed);
		~red_random();

		unsigned int rand();

		var_t uniform();
		int   uniform(int a, int b);
		var_t uniform(var_t a, var_t b);

		var_t normal(var_t m, var_t s);
		var_t exponential(var_t lambda);
		var_t rayleigh(var_t sigma);

		var_t power_law(var_t lower, var_t upper, var_t p);

	private:
		unsigned int idx;
		unsigned int I[624];
	};


	red_random::red_random(unsigned int seed) :
		idx(0)
	{
		I[  0] = seed & 0xffffffff;
		for(int i = 1; i < 624; ++i) 
		{
			I[i] = (1812433253*(I[i-1]^I[i-1]>>30)+i)&0xffffffff;
		}
	}

	red_random::~red_random()
	{ } 

	unsigned int red_random::rand()
	{
		unsigned int j = idx < 623 ? idx + 1 : 0;
		unsigned int y = I[idx]&0x80000000 | I[j]&0x7fffffff;
		y = I[idx] = I[idx < 227 ? idx + 397 : idx-227]^y>>1^(y&1)*0x9908b0df;
		idx = j;
		return y^(y^=(y^=(y^=y>>11)<<7&0x9d2c5680)<<15&0xefc60000)>>18;
	}

	var_t red_random::uniform()
    {
        return ((rand() + 1.0) / 4294967296.0);
    }

	int red_random::uniform(int a, int b)
    {
        return (a + (unsigned int)(rand()/4294967296.*(b - a + 1)));
    }

	var_t red_random::uniform(var_t a, var_t b)
    {
        return (a + (b - a)*(rand() + 1.0) / 4294967296.0);
    }

	var_t red_random::normal(var_t m, var_t s)
    {
        return (m + s*sqrt(-2.0*log((rand() + 1.0) / 4294967296.0)) * cos(1.4629180792671596E-9*(rand() + 1.0)));
    }

	var_t red_random::exponential(var_t lambda)
    {
        var_t u = uniform(); //(rand() + 1.0)/4294967296.0;
        return (-1.0 / lambda * log(u));
    }

    var_t red_random::rayleigh(var_t sigma)
    {
        var_t u = uniform();//(rand() + 1.0)/4294967296.0;
        return (sigma * sqrt(-2.0 * log(u)));
    }

	var_t red_random::power_law(var_t x_min, var_t x_max, var_t p)
	{
		assert(x_min < x_max);

		var_t y_min = pow(x_min, p);
		var_t y_max = pow(x_max, p);
		if (y_min > y_max)
		{
			swap(y_min, y_max);
		}

		var_t d_y = y_max - y_min;
		var_t d_x = x_max - x_min;

		var_t x, y;
		var_t area_max = d_x * d_y;

		do
		{
			x = uniform(0.0, area_max) / d_y + x_min;
			y = uniform(y_min, y_max);
		} while (y > pow(x, p));

		return x;
	}






	class pdf_base
	{
	public:
		pdf_base();
		~pdf_base();

		virtual var_t get_value(var_t x) = 0;
	};

	class pdf_constant : public pdf_base
	{
	public:
		pdf_constant();
		~pdf_constant();

		var_t get_value(var_t x);
	};

	class pdf_exponential : public pdf_base
	{
	public:
		pdf_exponential(var_t lambda);
		~pdf_exponential();

		var_t get_value(var_t x);
	private:
		var_t lambda;
	};

	pdf_base::pdf_base()
	{ }

	pdf_base::~pdf_base()
	{ }

	pdf_constant::pdf_constant()
	{ }

	pdf_constant::~pdf_constant()
	{ }

	var_t pdf_constant::get_value(var_t x)
	{
		return 1.0;
	}

	pdf_exponential::pdf_exponential(var_t lambda) :
		pdf_base(),
		lambda(lambda)
	{ }

	pdf_exponential::~pdf_exponential()
	{ }

	var_t pdf_exponential::get_value(var_t x)
	{
		return (lambda * exp(-lambda * x));
	}

	var_t generate(var2_t range, pdf_base* pdf)
	{
		var_t x;
		var_t y;

		do
		{
			x = range.x + (var_t)rand() / RAND_MAX * (range.y - range.x);
			y = (var_t)rand() / RAND_MAX;
		}
		while (y > pdf->get_value(x));

		return x;
	}

	void print_data(ostream& sout, vector<var_t>& data)
	{
		static int var_t_w = 17;

		sout.precision(8);
		sout.setf(ios::right);
		sout.setf(ios::scientific);

		for (int i = 0; i < data.size(); i++)
		{
			sout << setw(var_t_w) << data[i] << endl;
		}

		sout.flush();
	}


} /* random_generator */



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
			var2_t	range;
			var_t*	params;
			var_t	(*pdf)(var_t x);

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

typedef double ioT;
void Emese_data_format_to_red_cuda_format(const string& input_path, const string& output_path)
{
	ifstream input(input_path.c_str(), ios::in | ios::binary);
	ofstream output(output_path.c_str(), ios_base::out);

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

void set(distribution_t& d, var_t x0, var_t (*pdf)(var_t))
{
	d.range.x = d.range.y = x0;
	d.pdf = pdf;
}

void set(distribution_t& d, var_t x0, var_t x1, var_t* pdf_params, var_t (*pdf)(var_t))
{
	assert(x0 < x1);

	d.range.x = x0;
	d.range.y = x1;
	d.params = pdf_params;
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
			set(0.0, 0x0, pdf_const, bd.oe_d[body_type].item[i]);
		}
		for (int i = 0; i < 4; i++) 
		{
			set(0.0, 0x0, pdf_const, bd.pp_d[body_type].item[i]);
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
    oe.sma  = generate_random(oe_d->item[ORBITAL_ELEMENT_SMA].range.x,  oe_d->item[ORBITAL_ELEMENT_SMA].range.y,  oe_d->item[ORBITAL_ELEMENT_SMA].pdf);
	oe.ecc  = generate_random(oe_d->item[ORBITAL_ELEMENT_ECC].range.x,  oe_d->item[ORBITAL_ELEMENT_ECC].range.y,  oe_d->item[ORBITAL_ELEMENT_ECC].pdf);
	oe.inc  = generate_random(oe_d->item[ORBITAL_ELEMENT_INC].range.x,  oe_d->item[ORBITAL_ELEMENT_INC].range.y,  oe_d->item[ORBITAL_ELEMENT_INC].pdf);
	oe.peri = generate_random(oe_d->item[ORBITAL_ELEMENT_PERI].range.x, oe_d->item[ORBITAL_ELEMENT_PERI].range.y, oe_d->item[ORBITAL_ELEMENT_PERI].pdf);
	oe.node = generate_random(oe_d->item[ORBITAL_ELEMENT_NODE].range.x, oe_d->item[ORBITAL_ELEMENT_NODE].range.y, oe_d->item[ORBITAL_ELEMENT_NODE].pdf);
    oe.mean = generate_random(oe_d->item[ORBITAL_ELEMENT_MEAN].range.x, oe_d->item[ORBITAL_ELEMENT_MEAN].range.y, oe_d->item[ORBITAL_ELEMENT_MEAN].pdf);
}

void generate_pp(phys_prop_dist_t *pp_d, param_t& param)
{
	param.mass = generate_random(pp_d->item[MASS].range.x, pp_d->item[MASS].range.y, pp_d->item[MASS].pdf);

	if (	 pp_d->item[DENSITY].range.x == 0.0 && pp_d->item[DENSITY].range.y == 0.0 &&
			 pp_d->item[RADIUS].range.x == 0.0 && pp_d->item[RADIUS].range.y == 0.0 )
	{
		param.radius = 0.0;
		param.density = 0.0;
	}
	else if (pp_d->item[DENSITY].range.x == 0.0 && pp_d->item[DENSITY].range.y == 0.0 &&
			 pp_d->item[RADIUS].range.x > 0.0 && pp_d->item[RADIUS].range.y > 0.0 )
	{
		param.radius = generate_random(pp_d->item[RADIUS].range.x, pp_d->item[RADIUS].range.y, pp_d->item[RADIUS].pdf);
		param.density = calculate_density(param.mass, param.radius);
	}
	else if (pp_d->item[DENSITY].range.x > 0.0 && pp_d->item[DENSITY].range.y > 0.0 &&
			 pp_d->item[RADIUS].range.x == 0.0 && pp_d->item[RADIUS].range.y == 0.0 )
	{
		param.density = generate_random(pp_d->item[DENSITY].range.x, pp_d->item[DENSITY].range.y, pp_d->item[DENSITY].pdf);
		param.radius = calculate_radius(param.mass, param.density);
	}
	else
	{
		param.radius = generate_random(pp_d->item[RADIUS].range.x, pp_d->item[RADIUS].range.y, pp_d->item[RADIUS].pdf);
		param.density = generate_random(pp_d->item[DENSITY].range.x, pp_d->item[DENSITY].range.y, pp_d->item[DENSITY].pdf);
	}

	param.cd = generate_random(pp_d->item[DRAG_COEFF].range.x, pp_d->item[DRAG_COEFF].range.y, pp_d->item[DRAG_COEFF].pdf);
}

int generate_pp_disk(const string &path, body_disk_t& body_disk)
{
	static char sep = ' ';
	static const int precision = 10;

	ofstream	output(path.c_str(), ios_base::out);
	if (output)
	{
		for (int body_type = BODY_TYPE_STAR; body_type < BODY_TYPE_PADDINGPARTICLE; body_type++)
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

printf("Printing body %s to file ... ", body_disk.names[bodyIdx].c_str());
					print_body_record(output, body_disk.names[bodyIdx], epoch, &param0, &body_md, &rVec, &vVec, precision);
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
					int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
					if (ret_code == 1) {
						cerr << "Could not calculate the phase." << endl;
						return ret_code;
					}

printf("Printing body %s to file ... ", body_disk.names[bodyIdx].c_str());
					print_body_record(output, body_disk.names[bodyIdx], t, &param, &body_md, &rVec, &vVec, precision);
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

	set_default(disk);

	disk.nBody[BODY_TYPE_STAR       ] = 1;
	disk.nBody[BODY_TYPE_GIANTPLANET] = 1;

	int_t nBodies = calculate_number_of_bodies(disk);
	disk.mig_type = new migration_type_t[nBodies];
	disk.stop_at = new var_t[nBodies];

    int bodyIdx = 0;
	int type = BODY_TYPE_STAR;

	disk.names.push_back("star");
	set(1.0, 0x0, pdf_const, disk.pp_d[type].item[MASS]);
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

		set(disk.pp_d[type].item[MASS], mCeres, mMoon/10.0, pdf_const);
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
	/*
	 * Test the red_random class
	 */

	{
		const int N = 10000;
		string dir = "C:\\Work\\Projects\\red.cuda\\TestRandomGenerator\\";
		try
		{
			string path = dir + "power_law_-1.5.txt";
			ostream* data_f = new ofstream(path.c_str(), ios::out);

			random_generator::red_random my_random(1);

			vector<var_t>* data = new vector<var_t>[N];
			for (int i = 0; i < N; i++)
			{
				data->push_back(my_random.power_law(0.4, 4.0, -1.5));
			}
			random_generator::print_data(*data_f, *data);
		}
		catch (const exception ex)
		{
			cerr << "Error: " << ex.what() << endl;
		}

		return 0;
	}


	/*
	 * Test the random_generator namespace
	 */
	{
		const int N = 10000;
		string dir = "C:\\Work\\Projects\\red.cuda\\TestRandomGenerator\\";
		try
		{
			string path = dir + "pdf_exponential_10000.txt";
			ostream* data_f = new ofstream(path.c_str(), ios::out);

			var2_t range = {0.0, 5.0};
			random_generator::pdf_base* pdf_exp = new random_generator::pdf_exponential(2.0);

			vector<var_t>* data = new vector<var_t>[N];

			for (int i = 0; i < N; i++)
			{
				data->push_back(random_generator::generate(range, pdf_exp));
			}
			random_generator::print_data(*data_f, *data);

		}
		catch (const exception ex)
		{
			cerr << "Error: " << ex.what() << endl;
		}

		return 0;
	}


	body_disk_t disk;
	string outDir;
	string filename;
	string output_path;

	parse_options(argc, argv, outDir, filename);

/*
	{
		string input_path = file::combine_path(outDir, filename);
		string output_path = file::combine_path(outDir, file::get_filename_without_ext(filename) + ".txt");
		Emese_data_format_to_red_cuda_format(input_path, output_path);
		return 0;
	}
*/

	//{
	//	set_parameters_of_Two_body_disk(disk);
	//	output_path = file::combine_path(outDir, filename);
	//	generate_pp_disk(output_path, disk);
	//	return 0;
	//}

	{
		set_parameters_of_Dvorak_disk(disk);
		output_path = file::combine_path(outDir, filename);
		generate_pp_disk(output_path, disk);
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
