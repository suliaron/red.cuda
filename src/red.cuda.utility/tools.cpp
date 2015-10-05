// includes system
#include <cctype>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>

// includes project
#include "tools.h"
#include "red_constants.h"
#include "red_macro.h"
#include "red_type.h"

using namespace std;

namespace redutilcu
{
namespace tools
{
bool is_number(const string& str)
{
   for (size_t i = 0; i < str.length(); i++)
   {
	   if (std::isdigit(str[i]) || str[i] == 'e' || str[i] == 'E' || str[i] == '.' || str[i] == '-' || str[i] == '+')
	   {
           continue;
	   }
	   else
	   {
		   return false;
	   }
   }
   return true;
}

/// Removes all trailing white-space characters from the current std::string object.
void trim_right(string& str)
{
	// trim trailing spaces
	size_t endpos = str.find_last_not_of(" \t");
	if (string::npos != endpos )
	{
		str = str.substr(0, endpos + 1);
	}
}

/// Removes all trailing characters after the first c character
void trim_right(string& str, char c)
{
	// trim trailing spaces
	//size_t endpos = str.find(c);
	//if (string::npos != endpos )
	//{
	//	str = str.substr(0, endpos);
	//}

	//int idx = str.length() - 1;
	//printf("str[%2d] = '%c'\n", idx, str[idx]);
	//while (str[idx] == c)
	//{
	//	printf("str[%2d] = '%c'\n", idx, str[idx]);
	//	idx--;
	//}
	//str = str.substr(0, idx + 1);

	char *s = (char *)str.c_str();
	int idx = strlen(s) - 1;
	printf("s[%2d] = '%c'\n", idx, s[idx]);
	while (c == s[idx])
	{
		printf("s[%2d] = '%c'\n", idx, s[idx]);
		idx--;
	}
	s[idx + 1] = 0;
}

/// Removes all leading white-space characters from the current std::string object.
void trim_left(string& str)
{
	// trim leading spaces
	size_t startpos = str.find_first_not_of(" \t");
	if (string::npos != startpos )
	{
		str = str.substr( startpos );
	}
}

/// Removes all leading and trailing white-space characters from the current std::string object.
void trim(string& str)
{
	trim_right(str);
	trim_left(str);
}

string get_time_stamp(bool use_comma)
{
	static char time_stamp[20];

	time_t now = time(NULL);
	if (use_comma)
	{
		strftime(time_stamp, 20, "%Y-%m-%d,%H:%M:%S", localtime(&now));
	}
	else
	{
		strftime(time_stamp, 20, "%Y-%m-%d %H:%M:%S", localtime(&now));
	}

	return string(time_stamp);
}

string convert_time_t(time_t t)
{
	ostringstream convert;	// stream used for the conversion
	convert << t;			// insert the textual representation of 't' in the characters in the stream
	return convert.str();
}

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

var_t pdf_const(var_t x)
{
	return 1;
}

void populate_data(unsigned int* n_bodies, sim_data_t *sim_data)
{
	int idx = 0;
	int id = 1;

	// Create aliases
	vec_t* r             = sim_data->h_y[0];
	vec_t* v             = sim_data->h_y[1];
	param_t* p           = sim_data->h_p;
	orbelem_t* oe        = sim_data->h_oe;
	body_metadata_t* bmd = sim_data->h_body_md;
	ttt_t* epoch         = sim_data->h_epoch;

	int upper = n_bodies[BODY_TYPE_STAR];
	for (idx = 0; idx < upper; idx++)
	{
		bmd[idx].body_type = BODY_TYPE_STAR;
		bmd[idx].id = id++;

		p[idx].mass = 1.0;
		p[idx].radius = 1.0 * constants::SolarRadiusToAu;
		
		epoch[idx] = 0.0;

		r[idx].x = r[idx].y = r[idx].z = 0.0;
		v[idx].x = v[idx].y = v[idx].z = 0.0;
	}

	upper += n_bodies[BODY_TYPE_GIANTPLANET];
	for ( ; idx < upper; idx++)
	{
		bmd[idx].body_type = BODY_TYPE_GIANTPLANET;
		bmd[idx].id = id++;
		bmd[idx].mig_type = MIGRATION_TYPE_NO;
		bmd[idx].mig_stop_at = 0.0;

		p[idx].mass = generate_random(1.0, 10.0, pdf_const) * constants::JupiterToSolar;
		p[idx].radius = generate_random(8.0e4, 1.0e5, pdf_const) * constants::KilometerToAu;
		p[idx].density = calculate_density(p[idx].mass, p[idx].radius);
		p[idx].cd = 0.0;
		
		epoch[idx] = 0.0;

		oe[idx].sma = generate_random(1.0, 100.0, pdf_const);
		oe[idx].ecc = generate_random(0.0, 0.8, pdf_const);
		oe[idx].inc = generate_random(0.0, 20.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].peri = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].node = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].mean = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
	}

	upper += n_bodies[BODY_TYPE_ROCKYPLANET];
	for ( ; idx < upper; idx++)
	{
		bmd[idx].body_type = BODY_TYPE_ROCKYPLANET;
		bmd[idx].id = id++;
		bmd[idx].mig_type = MIGRATION_TYPE_NO;
		bmd[idx].mig_stop_at = 0.0;

		p[idx].mass = generate_random(1.0, 10.0, pdf_const) * constants::EarthToSolar;
		p[idx].radius = generate_random(5.0e3, 8.0e3, pdf_const) * constants::KilometerToAu;
		p[idx].density = calculate_density(p[idx].mass, p[idx].radius);
		p[idx].cd = 0.0;
		
		epoch[idx] = 0.0;

		oe[idx].sma = generate_random(1.0, 100.0, pdf_const);
		oe[idx].ecc = generate_random(0.0, 0.8, pdf_const);
		oe[idx].inc = generate_random(0.0, 20.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].peri = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].node = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].mean = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
	}

	upper += n_bodies[BODY_TYPE_PROTOPLANET];
	for ( ; idx < upper; idx++)
	{
		bmd[idx].body_type = BODY_TYPE_PROTOPLANET;
		bmd[idx].id = id++;
		bmd[idx].mig_type = MIGRATION_TYPE_NO;
		bmd[idx].mig_stop_at = 0.0;

		p[idx].mass = generate_random(1.0, 10.0, pdf_const) * constants::CeresToSolar;
		p[idx].radius = generate_random(1.0e3, 2.0e3, pdf_const) * constants::KilometerToAu;
		p[idx].density = calculate_density(p[idx].mass, p[idx].radius);
		p[idx].cd = 0.0;
		
		epoch[idx] = 0.0;

		oe[idx].sma = generate_random(1.0, 100.0, pdf_const);
		oe[idx].ecc = generate_random(0.0, 0.8, pdf_const);
		oe[idx].inc = generate_random(0.0, 20.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].peri = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].node = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].mean = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
	}

	upper += n_bodies[BODY_TYPE_SUPERPLANETESIMAL];
	for ( ; idx < upper; idx++)
	{
		bmd[idx].body_type = BODY_TYPE_SUPERPLANETESIMAL;
		bmd[idx].id = id++;
		bmd[idx].mig_type = MIGRATION_TYPE_NO;
		bmd[idx].mig_stop_at = 0.0;

		p[idx].mass = generate_random(1.0e-2, 1.0e-1, pdf_const) * constants::CeresToSolar;
		p[idx].radius = generate_random(1.0e1, 1.0e2, pdf_const) * constants::KilometerToAu;
		p[idx].density = generate_random(1.0, 3.0, pdf_const) * constants::GramPerCm3ToSolarPerAu3;
		p[idx].cd = 1.0;
		
		epoch[idx] = 0.0;

		oe[idx].sma = generate_random(1.0, 100.0, pdf_const);
		oe[idx].ecc = generate_random(0.0, 0.8, pdf_const);
		oe[idx].inc = generate_random(0.0, 20.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].peri = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].node = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].mean = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
	}

	upper += n_bodies[BODY_TYPE_PLANETESIMAL];
	for ( ; idx < upper; idx++)
	{
		bmd[idx].body_type = BODY_TYPE_PLANETESIMAL;
		bmd[idx].id = id++;
		bmd[idx].mig_type = MIGRATION_TYPE_NO;
		bmd[idx].mig_stop_at = 0.0;

		p[idx].mass = generate_random(1.0e-4, 1.0e-3, pdf_const) * constants::CeresToSolar;
		p[idx].radius = generate_random(1.0e1, 1.0e2, pdf_const) * constants::KilometerToAu;
		p[idx].density = calculate_density(p[idx].mass, p[idx].radius);
		p[idx].cd = generate_random(0.7, 2.0, pdf_const);
		
		epoch[idx] = 0.0;

		oe[idx].sma = generate_random(1.0, 100.0, pdf_const);
		oe[idx].ecc = generate_random(0.0, 0.8, pdf_const);
		oe[idx].inc = generate_random(0.0, 20.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].peri = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].node = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].mean = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
	}

	upper += n_bodies[BODY_TYPE_TESTPARTICLE];
	for ( ; idx < upper; idx++)
	{
		bmd[idx].body_type = BODY_TYPE_TESTPARTICLE;
		bmd[idx].id = id++;
		bmd[idx].mig_type = MIGRATION_TYPE_NO;
		bmd[idx].mig_stop_at = 0.0;

		p[idx].mass = 0.0;
		p[idx].radius = 0.0;
		p[idx].density = 0.0;
		p[idx].cd = 0.0;
		
		epoch[idx] = 0.0;

		oe[idx].sma = generate_random(1.0, 100.0, pdf_const);
		oe[idx].ecc = generate_random(0.0, 0.8, pdf_const);
		oe[idx].inc = generate_random(0.0, 20.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].peri = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].node = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
		oe[idx].mean = generate_random(0.0, 360.0, pdf_const) * constants::DegreeToRadian;
	}

	// Calculate coordinates and velocities
	{
		// The mass of the central star
		var_t m0 = sim_data->h_p[0].mass;
		vec_t rVec = {0.0, 0.0, 0.0, 0.0};
		vec_t vVec = {0.0, 0.0, 0.0, 0.0};

		// The coordinates of the central star
		sim_data->h_y[0][0] = rVec;
		sim_data->h_y[1][0] = vVec;
		for (int i = 1; i < upper; i++)
		{
			var_t mu = K2 *(m0 + sim_data->h_p[i].mass);
			tools::calculate_phase(mu, &sim_data->h_oe[i], &rVec, &vVec);
			sim_data->h_y[0][i] = rVec;
			sim_data->h_y[1][i] = vVec;
		}
	}
}

var_t get_total_mass(int n, body_type_t type, const sim_data_t *sim_data)
{
	var_t total_mass = 0.0;

	param_t* p = sim_data->h_p;
	for (int j = n - 1; j >= 0; j--)
	{
		if (sim_data->h_body_md[j].body_type == type)
		{
			total_mass += p[j].mass;
		}
	}

	return total_mass;
}

var_t get_total_mass(int n, const sim_data_t *sim_data)
{
	var_t total_mass = 0.0;

	param_t* p = sim_data->h_p;
	for (int j = n - 1; j >= 0; j--)
	{
		total_mass += p[j].mass;
	}

	return total_mass;
}

void compute_bc(int n, bool pts, const sim_data_t *sim_data, vec_t* R0, vec_t* V0)
{
	const param_t* p = sim_data->h_p;
	const vec_t* r = sim_data->h_y[0];
	const vec_t* v = sim_data->h_y[1];

	for (int j = n - 1; j >= 0; j-- )
	{
		var_t m = p[j].mass;
		R0->x += m * r[j].x;
		R0->y += m * r[j].y;
		R0->z += m * r[j].z;

		V0->x += m * v[j].x;
		V0->y += m * v[j].y;
		V0->z += m * v[j].z;
	}
	var_t M0 = get_total_mass(n, sim_data);

	R0->x /= M0;	R0->y /= M0;	R0->z /= M0;
	V0->x /= M0;	V0->y /= M0;	V0->z /= M0;

	if (pts)
	{
		cout << "   Position and velocity of the barycenter:" << endl;
		cout << "     R0: ";  print_vector(R0);
		cout << "     V0: ";  print_vector(V0);
	}
}

void transform_to_bc(int n, bool pts, const sim_data_t *sim_data)
{
	if (pts)
	{
		cout << "Transforming to barycentric system ... " << endl;
	}

	// Position and velocity of the system's barycenter
	vec_t R0 = {0.0, 0.0, 0.0, 0.0};
	vec_t V0 = {0.0, 0.0, 0.0, 0.0};

	compute_bc(n, pts, sim_data, &R0, &V0);

	vec_t* r = sim_data->h_y[0];
	vec_t* v = sim_data->h_y[1];
	// Transform the bodies coordinates and velocities
	for (int j = n - 1; j >= 0; j-- )
	{
		r[j].x -= R0.x;		r[j].y -= R0.y;		r[j].z -= R0.z;
		v[j].x -= V0.x;		v[j].y -= V0.y;		v[j].z -= V0.z;
	}

	if (pts)
	{
		cout << "done" << endl;
	}
}

var_t calculate_radius(var_t m, var_t density)
{
	static var_t four_pi_over_three = 4.1887902047863909846168578443727;

	return pow(1.0/four_pi_over_three * m/density, 1.0/3.0);
}

var_t calculate_density(var_t m, var_t R)
{
	static var_t four_pi_over_three = 4.1887902047863909846168578443727;

	if (R == 0.0)
	{
		return 0.0;
	}
	return m / (four_pi_over_three * CUBE(R));
}

var_t caclulate_mass(var_t R, var_t density)
{
	static var_t four_pi_over_three = 4.1887902047863909846168578443727;

	return four_pi_over_three * CUBE(R) * density;
}

void calc_position_after_collision(var_t m1, var_t m2, const vec_t* r1, const vec_t* r2, vec_t& r)
{
	const var_t M = m1 + m2;

	r.x = (m1 * r1->x + m2 * r2->x) / M;
	r.y = (m1 * r1->y + m2 * r2->y) / M;
	r.z = (m1 * r1->z + m2 * r2->z) / M;
}

void calc_velocity_after_collision(var_t m1, var_t m2, const vec_t* v1, const vec_t* v2, vec_t& v)
{
	const var_t M = m1 + m2;

	v.x = (m1 * v1->x + m2 * v2->x) / M;
	v.y = (m1 * v1->y + m2 * v2->y) / M;
	v.z = (m1 * v1->z + m2 * v2->z) / M;
}

void calc_physical_properties(const param_t &p1, const param_t &p2, param_t &p)
{
	// Calculate V = V1 + V2
	var_t volume = 4.188790204786391 * (CUBE(p1.radius) + CUBE(p2.radius));

	p.mass	  = p1.mass + p2.mass;
	p.density = p.mass / volume;
	p.radius  = calculate_radius(p.mass, p.density);
	p.cd      = p1.cd;
}

inline var_t norm(const vec_t* r)
{
	return sqrt(SQR(r->x) + SQR(r->y) + SQR(r->z));
}

inline var_t calc_kinetic_energy(const vec_t* v)
{
	return (SQR(v->x) + SQR(v->y) + SQR(v->z)) / 2.0;
}

inline var_t calc_pot_energy(var_t mu, const vec_t* r)
{
    return -mu / norm(r);
}

inline var_t calc_energy(var_t mu, const vec_t* r, const vec_t* v)
{
	return calc_kinetic_energy(v) + calc_pot_energy(mu, r);
}

void shift_into_range(var_t lower, var_t upper, var_t &value)
{
    var_t range = upper - lower;
    while (value >= upper)
    {
        value -= range;
    }
    while (value < lower)
    {
        value += range;
    }
}

void kepler_equation_solver(var_t ecc, var_t mean, var_t eps, var_t* E)
{
	if (ecc == 0.0 || mean == 0.0 || mean == PI)
	{
        *E = mean;
		return;
    }
    *E = mean + ecc * (sin(mean)) / (1.0 - sin(mean + ecc) + sin(mean));
    var_t E1 = 0.0;
    var_t error;
    int step = 0;
    do
	{
        E1 = *E - (*E - ecc * sin(*E) - mean) / (1.0 - ecc * cos(*E));
        error = fabs(E1 - *E);
        *E = E1;
    } while (error > eps && step++ <= 15);
	if (15 < step)
	{
		throw string("The kepler_equation_solver() failed: solution did not converge.");
	}
}

void calculate_phase(var_t mu, const orbelem_t* oe, vec_t* rVec, vec_t* vVec)
{
    var_t ecc = oe->ecc;
	var_t E = 0.0;
	kepler_equation_solver(ecc, oe->mean, 1.0e-14, &E);
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
}

void calculate_oe(var_t mu, const vec_t* rVec, const vec_t* vVec, orbelem_t* oe)
{
    const var_t sq2 = 1.0e-14;
    const var_t sq3 = 1.0e-14;

	var_t r_norm = norm(rVec);
	var_t v_norm = norm(vVec);

	// Calculate energy, h
    var_t h = calc_energy(mu, rVec, vVec);
    if (h >= 0.0)
    {
		throw string("The Kepler-energy is positive. calculate_oe() failed.");
    }

	vec_t cVec;
    cVec.x = rVec->y * vVec->z - rVec->z * vVec->y;
    cVec.y = rVec->z * vVec->x - rVec->x * vVec->z;
    cVec.z = rVec->x * vVec->y - rVec->y * vVec->x;
	cVec.w = 0.0;
	var_t c_norm = norm(&cVec);

	vec_t lVec;
	lVec.x = -mu / r_norm * rVec->x + vVec->y * cVec.z - vVec->z * cVec.y;
	lVec.y = -mu / r_norm * rVec->y + vVec->z * cVec.x - vVec->x * cVec.z;
	lVec.z = -mu / r_norm * rVec->z + vVec->x * cVec.y - vVec->y * cVec.x;
	lVec.w = 0.0;
	var_t l_norm = norm(&lVec);

	var_t rv = rVec->x * vVec->x + rVec->y * vVec->y + rVec->z * vVec->z;

    var_t e2 = 1.0 + 2.0 * SQR(c_norm) * h / (SQR(mu));
    if (abs(e2) < sq3)
    {
        e2 = 0.0;
    }
    var_t e = sqrt(e2);
    /*
    * Calculate semi-major axis, a
    */
    var_t a = -mu / (2.0 * h);
    /*
    * Calculate inclination, incl
    */
    var_t cosi = cVec.z / c_norm;
    var_t sini = sqrt(cVec.x * cVec.x + cVec.y * cVec.y) / c_norm;
    var_t incl = acos(cosi);
    if (incl < sq2)
    {
        incl = 0.0;
    }
    /*
    * Calculate longitude of node, O
    */
    var_t node = 0.0;
    if (incl != 0.0)
    {
        var_t tmpx = -cVec.y / (c_norm * sini);
        var_t tmpy =  cVec.x / (c_norm * sini);
		node = atan2(tmpy, tmpx);
		shift_into_range(0.0, 2.0*PI, node);
    }
    /*
    * Calculate argument of pericenter, w
    */
    var_t E = 0.0;
    var_t peri = 0.0;
    if (e2 != 0.0)
    {
        var_t tmpx = (lVec.x * cos(node) + lVec.y * sin(node)) / l_norm;
        var_t tmpy = (-lVec.x * sin(node) + lVec.y * cos(node)) / (l_norm * cosi);
        peri = atan2(tmpy, tmpx);
        shift_into_range(0.0, 2.0*PI, peri);

        tmpx = 1.0 / e * (1.0 - r_norm / a);
        tmpy = rv / (sqrt(mu * a) * e);
        E = atan2(tmpy, tmpx);
        shift_into_range(0.0, 2.0*PI, E);
    }
    else
    {
        peri = 0.0;
        E = atan2(rVec->y, rVec->x);
        shift_into_range(0, 2.0*PI, E);
    }
    /*
    * Calculate mean anomaly, M
    */
    var_t M = E - e * sin(E);
    shift_into_range(0, 2.0*PI, M);

	oe->sma  = a;
	oe->ecc	 = e;
	oe->inc  = incl;
	oe->peri = peri;
	oe->node = node;
	oe->mean = M;
}

void print_vector(const vec_t *v)
{
	static int var_t_w  = 25;

	cout.precision(16);
	cout.setf(ios::right);
	cout.setf(ios::scientific);

	cout << setw(var_t_w) << v->x 
		 << setw(var_t_w) << v->y
		 << setw(var_t_w) << v->z
		 << setw(var_t_w) << v->w << endl;
}

void print_parameter(const param_t *p)
{
	static int var_t_w  = 25;

	cout.precision(16);
	cout.setf(ios::right);
	cout.setf(ios::scientific);

	cout << setw(var_t_w) << p->mass 
		 << setw(var_t_w) << p->radius
		 << setw(var_t_w) << p->density
		 << setw(var_t_w) << p->cd << endl;
}

void print_body_metadata(const body_metadata_t *b)
{
	static int var_t_w  = 5;

	cout << setw(var_t_w) << b->id << endl;
	cout << (char)(48 + b->body_type) << " (" << body_type_name[b->body_type] << ")" << endl
		 << (char)(48 + b->mig_type) << " (" << migration_type_name[b->mig_type] << ")" << endl;
	cout.precision(16);
	cout.setf(ios::right);
	cout.setf(ios::scientific);
	cout << setw(var_t_w) << b->mig_stop_at << endl;
}

void print_body_metadata(const body_metadata_new_t *b)
{
	static int var_t_w  = 5;

	cout << setw(var_t_w) << b->id << endl;
	cout << (char)(48 + b->body_type) << " (" << body_type_name[b->body_type] << ")" << endl
		 << (char)(48 + b->mig_type) << " (" << migration_type_name[b->mig_type] << ")" << endl
		 << (char)(48 + b->active) << (b->active ? " (true)" : " (false)") << endl
		 << (char)(48 + b->unused) << (b->unused ? " (true)" : " (false)") << endl;
	cout.precision(16);
	cout.setf(ios::right);
	cout.setf(ios::scientific);
	cout << setw(var_t_w) << b->mig_stop_at << endl;
}
} /* tools */
} /* redutilcu */
